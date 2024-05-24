# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from itertools import chain
from math import prod

from bex.basis.dvr import SineDVR

from bex.bath.correlation import Correlation
from bex.libs.backend import MAX_EINSUM_AXES, Array, as_array, OptArray, arange, np, opt_einsum, opt_tensordot, opt_array, zeros, dtype
from bex.libs.utils import huffman_tree, Optional
from bex.operator.spo import SumProdOp
from bex.state.frame import End, Frame, Node, Point
from bex.state.model import CannonialModel

EPSILON = 1.0e-15


class Hierachy(CannonialModel):

    i_end = End(f'[EHEOM]i')
    j_end = End(f'[EHEOM]j')

    @staticmethod
    def a_end(identifier: int) -> End:
        return End(f'[EHEOM]A-{identifier}')

    @staticmethod
    def b_end(identifier: int) -> End:
        return End(f'[EHEOM]B-{identifier}')

    @staticmethod
    def node(identifier: ...) -> Node:
        return Node(f'[EHEOM]{identifier}')

    def __init__(
            self,
            rdo: Array,
            dims: list[int],
            frame: Frame,
            root: Node,
            rank: int = 1,
            decimation_rate: Optional[int] = None,
            spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:

        rdo = as_array(rdo)
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        all_dims = shape + dims + dims
        e_ends = [self.i_end, self.j_end]
        a_ends = [self.a_end(k) for k in range(len(dims))]
        b_ends = [self.b_end(k) for k in range(len(dims))]
        all_ends = e_ends + a_ends + b_ends

        assert root in frame.nodes
        assert frame.dual(root, 0)[0] == e_ends[0]
        assert frame.dual(root, 1)[0] == e_ends[1]
        assert set(all_ends) == frame.ends
        super().__init__(frame, root)

        axes = self.axes
        dim_dct = {frame.dual(e, 0): d for d, e in zip(all_dims, all_ends)}
        if decimation_rate is not None:
            for node in reversed(frame.node_visitor(root, 'BFS')):
                ax = axes[node]
                if ax is not None:
                    _ds = [
                        dim_dct[node, i] for i in range(frame.order(node))
                        if i != ax
                    ]
                    dim_dct[frame.dual(node, ax)] = max(
                        rank, int(prod(_ds) / decimation_rate))
        self.fill_eyes(dims=dim_dct, default_dim=rank)

        ext_shape = [k for i, k in enumerate(self.shape(root)) if i > 1]
        ext = zeros([prod(ext_shape)])
        ext[0] = 1.0
        _array = np.tensordot(rdo, ext, axes=0).reshape(self.shape(root))
        self[root] = _array

        # QFPE for defined k
        self.bases = dict()  # type: dict[End, SineDVR]
        tfmats = dict()  #type: dict[End, OptArray]
        if spaces is not None:
            for k, (x0, x1) in spaces.items():
                b = SineDVR(x0, x1, dims[k])
                tfmats[a_ends[k]] = opt_array(b.fock2dvr_mat)
                tfmats[b_ends[k]] = opt_array(b.fock2dvr_mat)
                self.bases[a_ends[k]] = b
                self.bases[b_ends[k]] = b

            for _q, tfmat in tfmats.items():
                _p, _i = frame.dual(_q, 0)
                _a = opt_tensordot(self[_p], tfmat,
                                   ([_i], [1])).moveaxis(-1, _i)
                self.opt_update(_p, _a)

        # add terminators
        self.terminators = {}  # type: dict[Point, OptArray]
        for _k, d in zip(a_ends + b_ends, dims + dims):
            if _k in tfmats:
                tm = (tfmats[_k].mH)[0]
            else:
                tm = zeros([d])
                tm[0] = 1.0
                tm = opt_array(tm)
            self.terminators[_k] = tm
        self.bfs_visitor = self.frame.node_visitor(root, method='BFS')
        return

    @staticmethod
    def terminate(tensor: OptArray, term_dict: dict[int, OptArray]):
        order = tensor.ndim
        n = len(term_dict)
        assert order + n - 1 < MAX_EINSUM_AXES

        ax_list = list(
            sorted(term_dict.keys(), key=(lambda ax: tensor.shape[ax])))
        vec_list = [term_dict[ax] for ax in ax_list]

        args = [(tensor, list(range(order)))]
        args.extend((vec_list[i], [ax_list[i]]) for i in range(n))
        ans_axes = [ax for ax in range(order) if ax not in ax_list]
        args.append((ans_axes, ))

        args = list(chain(*args))
        ans = opt_einsum(*args)
        return ans

    def get_rdo(self) -> OptArray:
        axes = self.axes
        root = self.root
        terminators = self.terminators
        terminate = self.terminate
        near = self.frame.near_points

        # Iterative from leaves to root (not include)
        for p in reversed(self.bfs_visitor[1:]):
            term_dict = {
                i: terminators[q]
                for i, q in enumerate(near(p)) if i != axes[p]
            }
            terminators[p] = terminate(self[p], term_dict)

        # root node: 0 and 1 observed for i and j
        term_dict = {
            i: terminators[q]
            for i, q in enumerate(near(root)) if i > 1
        }
        # print(torch.norm(term_dict[2]))
        _array = terminate(self[root], term_dict)
        dim = _array.shape[0]
        return _array.reshape((dim, dim, -1))[:, :, 0]


class NaiveHierachy(Hierachy):

    def __init__(
            self,
            rdo: Array,
            dims: list[int],
            spaces: Optional[dict[int, tuple[float, float]]] = None) -> None:
        ends = ([self.i_end, self.j_end] +
                [self.a_end(k) for k in range(len(dims))] +
                [self.b_end(k) for k in range(len(dims))])
        frame = Frame()
        root = self.node('root')
        for e in ends:
            frame.add_link(root, e)
        super().__init__(rdo, dims, frame, root, spaces=spaces)
        return


class HeomOp(SumProdOp):
    rm_zeroes = False

    @staticmethod
    def _fk_1(c_k):
        return 1.0

    @staticmethod
    def _fk_2(c_k):
        return np.sqrt(c_k, dtype=complex)

    @staticmethod
    def _fk_3(c_k):
        return c_k

    heom_type = 1

    def __init__(self, hierachy: Hierachy, sys_hamiltonian: Array,
                 sys_op: Array, corr: Correlation, dims: list[int]) -> None:
        self.bases = hierachy.bases
        self.i_end = hierachy.i_end
        self.j_end = hierachy.j_end
        self.a_end = hierachy.a_end
        self.b_end = hierachy.b_end
        self.h = sys_hamiltonian
        self.op = sys_op
        self.coefficients = corr.coefficients
        self.zeropoints = corr.zeropoints
        self.derivatives = corr.derivatives
        self.k_dims = dims

        super().__init__(self.op_list)
        return

    @property
    def op_list(self) -> list[dict[End, Array]]:
        ans = [{
            self.i_end: -1.0j * self.h
        }, {
            self.j_end: 1.0j * self.h.conj()
        }]
        for k in range(len(self.k_dims)):
            ans.extend(self.bcf_term(k, True))
            ans.extend(self.bcf_term(k, False))
        return ans

    def bcf_term(self, k: int, left: bool) -> list[dict[End, Array]]:
        if self.heom_type == 1:
            fk_func = self._fk_1
        elif self.heom_type == 2:
            fk_func = self._fk_2
        elif self.heom_type == 3:
            fk_func = self._fk_3
        else:
            raise RuntimeError(
                f'Not supported HEOM factor type {self.heom_type}.')

        if left:
            i_end = self.i_end
            j_end = self.j_end
            k_end = self.a_end(k)
            ck = self.coefficients[k]
            zk = self.zeropoints[k]
            dk = self.derivatives[k]
        else:
            i_end = self.j_end
            j_end = self.i_end
            k_end = self.b_end(k)
            ck = self.coefficients[k].conjugate()
            zk = self.zeropoints[k].conjugate()
            dk = {
                _k: _dk.conjugate()
                for _k, _dk in self.derivatives[k].items()
            }
        if k_end in self.bases:
            raiser = self.bases[k_end].creation_mat
            lower = self.bases[k_end].annihilation_mat
            numberer = self.bases[k_end].numberer_mat
        else:
            dim = self.k_dims[k]
            raiser = np.diag(np.sqrt(np.arange(1, dim, dtype=dtype)), k=-1)
            lower = np.diag(np.sqrt(np.arange(1, dim, dtype=dtype)), k=1)
            numberer = np.diag(arange(dim))

        if self.rm_zeroes and abs(zk * ck) < EPSILON:
            ans = []
        else:
            fk = fk_func(ck)
            print('f_k@', k_end, ': ', fk, flush=True)
            ans = [{
                i_end: -1.0j * self.op,
                k_end: zk / fk * raiser + ck * fk * lower
            }, {
                j_end: 1.0j * self.op,
                k_end: ck * fk * lower
            }]

        for k1, dk1 in dk.items():
            k1_end = self.a_end(k1) if left else self.b_end(k1)
            if k1_end is k_end:
                ans.append({k_end: dk1 * numberer})
            else:
                k1_lower = (self.bases[k1_end].annihilation_mat
                            if k1_end in self.bases else np.diag(np.sqrt(
                                np.arange(1, self.k_dims[k1], dtype=dtype)),
                                                                 k=1))
                ans.append({k_end: dk1 * raiser, k1_end: k1_lower})
        return ans
