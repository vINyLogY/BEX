from functools import reduce
from math import prod
from typing import Callable, Iterable, Optional

from bex.libs.backend import (Array, OptArray, as_array, eye, moveaxis, opt_qr,
                              opt_tensordot, opt_array, zeros, np)
from bex.state.frame import End, Frame, Node, Point
from bex.libs.utils import depths


def triangular(n_list):
    """A Generator yields the natural number in a triangular order.
        """
    length = len(n_list)
    prod_list = [1]
    for n in n_list:
        prod_list.append(prod_list[-1] * n)
    prod_list = prod_list

    def key(case):
        return sum(n * i for n, i in zip(prod_list, case))

    combinations = {0: [[0] * length]}
    for m in range(prod_list[-1]):
        if m not in combinations:
            permutation = [
                case[:j] + [case[j] + 1] + case[j + 1:]
                for case in combinations[m - 1] for j in range(length)
                if case[j] + 1 < n_list[j]
            ]
            combinations[m] = []
            for case in permutation:
                if case not in combinations[m]:
                    combinations[m].append(case)
        for case in combinations[m]:
            yield key(case)


class Model:
    """A Model is a Frame with valuation for each node.
    """

    def __init__(self, frame: Frame) -> None:
        """
        Args:
            frame: Topology of the tensor network;
        """
        self.frame = frame
        self.ends = frame.ends
        self._dims = dict()  # type: dict[tuple[Point, int], int]
        self._valuation = dict()  # type: dict[Node, OptArray]
        return

    def __contains__(self, item):
        return item in self.frame.nodes

    def items(self):
        return self._valuation.items()

    def shape(self, p: Point) -> list[Optional[int]]:
        assert p in self.frame.world

        if isinstance(p, End):
            dim = self._dims.get((p, 0))
            node_shape = [dim, dim]
        else:
            # isinstance(p, Node)
            node_shape = [
                self._dims.get((p, i)) for i in range(self.frame.order(p))
            ]
        return node_shape

    def __getitem__(self, p: Node) -> OptArray:
        return self._valuation[p]

    def __setitem__(self, p: Node, array: Array) -> None:
        assert p in self.frame.nodes
        order = self.frame.order(p)
        assert array.ndim == order

        # Check confliction
        for i, dim in enumerate(array.shape):
            for pair in [(p, i), self.frame.dual(p, i)]:
                if pair in self._dims:
                    assert self._dims[pair] == dim
                else:
                    self._dims[pair] = dim

        self._valuation[p] = opt_array(array)
        return

    def __delitem__(self, p: Node) -> None:
        for i in range(self.frame.order(p)):
            del self._dims[(p, i)]
        del self._valuation[p]
        return

    def get_size(self) -> int:
        size = 0
        for p in self._valuation.keys():
            size += prod(self.shape(p))
        return size

    def opt_update(self, p: Node, array: OptArray) -> None:
        # assert p in self._valuation
        self._valuation[p] = array
        self._dims.update({(p, i): dim for i, dim in enumerate(array.shape)})
        return

    def fill_zeros(self,
                   dims: Optional[dict[tuple[Node, int], int]] = None,
                   default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the dimension for each Edge in dims (default is 1).
        """
        if dims is None:
            dims = dict()

        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        _dims = self._dims

        for p in self.frame.nodes:
            if p not in valuation:
                shape = [
                    _dims.get((p, i),
                              dims.get((p, i),
                                       dims.get(dual(p, i), default_dim)))
                    for i in range(order(p))
                ]
                self[p] = zeros(shape)

        return

    def get_axes(self, root: Point) -> dict[Point, Optional[int]]:
        assert root in self.frame.world

        ans = {root: None}
        for m, _, n, j in self.frame.point_link_visitor(root):
            if m in ans and n not in ans:
                ans[n] = j
        return ans

    def get_depths(self, root) -> dict[Node, int]:
        return depths(root, self.frame.near_nodes)

    def fill_eyes(self,
                  axes: dict[Node, int],
                  dims: Optional[dict[tuple[Node, int], int]] = None,
                  default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the each dimension tensors (default is 1).
        """
        if dims is None:
            dims = dict()
        nodes = self.frame.nodes
        order = self.frame.order
        dual = self.frame.dual
        valuation = self._valuation
        saved_dims = self._dims

        for p in nodes:
            if p not in valuation:
                ax = axes[p]
                shape = [
                    saved_dims.get((p, i),
                                   dims.get((p, i),
                                            dims.get(dual(p, i), default_dim)))
                    for i in range(order(p))
                ]
                if ax is None:
                    ans = zeros((prod(shape), ))
                    ans[0] = 1.0
                    ans = ans.reshape(shape)
                else:
                    _m = shape.pop(ax)
                    _n = prod(shape)
                    # Naive
                    # ans = moveaxis(eye(_m, _n).reshape([_m] + shape), 0, ax)

                    # Triangular
                    ans = zeros([_m, _n])
                    for n, v_i in zip(triangular(shape), ans):
                        v_i[n] = 1.0
                    ans = moveaxis(ans.reshape([_m] + shape), 0, ax)

                self[p] = ans
        return


class CannonialModel(Model):
    """
    State is a Network with Tree-shape and root.
    """

    def __init__(self, frame: Frame, root: Node) -> None:
        super().__init__(frame)

        self._root = root
        self._axes = None  # type: Optional[dict[Node, Optional[int]]]
        self._depths = None  # type: Optional[dict[Node, Optional[int]]]

        self._point_visitor = self.frame.point_visitor(start=self.root,
                                                       method='DFS')
        return

    def calc_densities(self) -> dict[Point, OptArray]:
        axes = self.axes
        dual = self.frame.dual
        order = self.frame.order
        densities = dict()  # type: dict[Point, OptArray]

        # From root to leaves
        for p in self._point_visitor:
            i = axes[p]
            if i is None:
                continue
            q, j = dual(p, i)
            k = axes[q]
            a_q = self[q]
            if k is not None:
                den_q = densities[q]
                a_q = opt_tensordot(den_q, a_q, ([1], [k])).moveaxis(0, k)
            ops = [_j for _j in range(order(q)) if _j != j]
            ans = opt_tensordot(a_q.conj(), a_q, (ops, ops))
            densities[p] = ans
        return densities

    @property
    def root(self) -> Node:
        return self._root

    @root.setter
    def root(self, value: Node) -> None:
        assert value in self.frame.nodes
        self._root = value
        self._axes = None
        self._depths = None
        return

    @property
    def axes(self) -> dict[Point, Optional[int]]:
        if self._axes is None:
            self._axes = self.get_axes(self.root)
        return self._axes

    @property
    def depths(self) -> dict[Node, int]:
        if self._depths is None:
            self._depths = self.get_depths(self.root)
        return self._depths

    def fill_eyes(self,
                  dims: Optional[dict[tuple[Node, int], int]] = None,
                  default_dim: int = 1) -> None:
        """
        Fill the unassigned part of model with proper shape arrays.
        Specify the each dimension tensors (default is 1).
        """
        super().fill_eyes(self.axes, dims=dims, default_dim=default_dim)
        return

    def split_unite_move(
            self,
            i: int,
            op: Optional[Callable[[OptArray], OptArray]] = None) -> None:
        m = self.root
        assert i < self.frame.order(m)

        n, j = self.frame.dual(m, i)
        dim = self._dims[(m, i)]
        shape = self.shape(m)
        shape.pop(i)

        mat_m = self[m].moveaxis(i, -1).reshape((-1, dim))
        q, mid = opt_qr(mat_m)
        array_m = q.reshape(shape + [-1]).moveaxis(-1, i)
        self.opt_update(m, array_m)
        self.root = n

        if op is not None:
            mid = op(mid)

        array_n = opt_tensordot(mid, self[n], ([1], [j])).moveaxis(0, j)
        self.opt_update(n, array_n)
        return

    def unite_split_move(self,
                         i: int,
                         op: Optional[Callable[[OptArray], OptArray]] = None,
                         compressed=False) -> None:
        m = self.root
        assert i < self.frame.order(m)

        n, j = self.frame.dual(m, i)
        shape_m = self.shape(m)
        shape_m.pop(i)
        shape_n = self.shape(n)
        shape_n.pop(j)

        mid = opt_tensordot(self[m], self[n], ([i], [j]))
        if op is not None:
            mid = op(mid)

        q, r = opt_qr(mid.reshape((prod(shape_m), prod(shape_n))),
                      compressed=compressed)
        array_m = q.reshape(shape_m + [-1]).moveaxis(-1, i)
        self.opt_update(m, array_m)
        self.root = n

        array_n = r.reshape([-1] + shape_n).moveaxis(0, j)
        self.opt_update(n, array_n)
        return

    def ev(self, op_dict) -> Array:
        from bex.operator.spo import MasterEqn, SumProdOp
        root = self.root
        order = self.frame.order(root)
        ops = SumProdOp([op_dict])
        cache = MasterEqn(ops, self)
        cache._get_mean_fields_type1()
        mfs = cache.mean_fields
        op_list = {i: mfs[root, i] for i in range(order)}
        right_array = ops.reduce(
            ops.transforms(ops.expand(self[root]), op_list))
        left_array = self[root].conj()
        ans = self.hs_ip(left_array, right_array)

        return ans

    def get_normsq(self) -> Array:
        return self.ev(self.identity)

    @staticmethod
    def hs_ip(tensor1: OptArray, tensor2: OptArray) -> OptArray:
        """Hilbert--Schmidt inner-product
        """
        # order = tensor1.ndim
        # assert 2 < order and order == tensor2.ndim
        # assert order + 1 < MAX_EINSUM_AXES

        # # A^\dagger_{ij} B_{ji} = A^*_{ji} B_{ji}
        # axes1 = [0, 1] + list(range(2, order))
        # axes2 = [0, 1] + list(range(2, order))
        # return opt_einsum(tensor1, axes1, tensor2, axes2)
        ans = tensor1.flatten() @ tensor2.flatten()
        return ans.cpu().numpy()

    @property
    def identity(self) -> dict[End, Array]:
        op = {_end: np.identity(self._dims[_end, 0]) for _end in self.ends}
        return op
