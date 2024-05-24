# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.
"""

from itertools import chain
from math import prod
from typing import Literal

from bex.basis.dvr import DiscreteVariationalRepresentation
from bex.bath.correlation import Correlation
from bex.libs.backend import (MAX_EINSUM_AXES, NDArray, OptArray, Array,
                              arange, as_array, dtype, np, opt_einsum,
                              opt_tensordot, opt_array, zeros, ones)
from bex.libs.utils import Optional
from bex.operator.spo import MasterEqn, SumProdOp
from bex.state.frame import End, Frame, Node, Point
from bex.state.model import CannonialModel

EPSILON = 1.0e-14


class Hierachy(CannonialModel):

    @staticmethod
    def end(identifier: Literal['i', 'j'] | int) -> End:
        return End(f'[HEOM]{identifier}')

    @staticmethod
    def node(identifier: ...) -> Node:
        return Node(f'[HEOM]{identifier}')

    def __init__(
        self,
        rdo: NDArray,
        dims: list[int],
        frame: Frame,
        root: Node,
        rank: int = 1,
        decimation_rate: Optional[float] = None,
        dvr_bases: Optional[dict[int,
                                 DiscreteVariationalRepresentation]] = None
    ) -> None:

        rdo = as_array(rdo)
        print(rdo)
        shape = list(rdo.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        all_dims = shape + dims
        e_ends = [self.end('i'), self.end('j')]
        p_ends = [self.end(k) for k in range(len(dims))]
        all_ends = e_ends + p_ends
        self._e_ends = e_ends
        self._p_ends = p_ends

        assert root in frame.nodes
        assert frame.dual(root, 0)[0] == e_ends[0]
        assert frame.dual(root, 1)[0] == e_ends[1]
        assert set(all_ends) == frame.ends
        super().__init__(frame, root)

        dim_dct = {frame.dual(e, 0): d for d, e in zip(all_dims, all_ends)}
        axes = self.axes
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
        self.bases = dict(
        )  # type: dict[End, DiscreteVariationalRepresentation]
        tfmats = dict()  #type: dict[End, OptArray]
        if dvr_bases is not None:
            for k, b in dvr_bases.items():
                _t = opt_array(b.fock2dvr_mat)
                if _t[:, 0].real.sum() < 0:
                    _t = -_t
                tfmats[p_ends[k]] = _t
                self.bases[p_ends[k]] = b

            for _q, tf in tfmats.items():
                _p, _i = frame.dual(_q, 0)
                _a = opt_tensordot(self[_p], tf, ([_i], [1])).moveaxis(-1, _i)
                self.opt_update(_p, _a)

        # add terminators
        self.terminators = {}  # type: dict[Point, OptArray]
        for _k, d in zip(p_ends, dims):
            if _k in tfmats:
                _r = (tfmats[_k].mH)[0]
                if _r.real.sum() < 0:
                    _r = -_r
                tm = _r
            else:
                tm = zeros([d]).real
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

        args = [tensor, list(range(order))]
        for _v, _ax in zip(vec_list, ax_list):
            args += [_v, [_ax]]
        args.append([ax for ax in range(order) if ax not in ax_list])
        ans = opt_einsum(*args)
        return ans

    @staticmethod
    def transform(tensor: OptArray, op_dict: dict[int, OptArray]) -> OptArray:
        order = tensor.ndim
        n = len(op_dict)
        assert order + n - 1 < MAX_EINSUM_AXES

        ax_list = list(
            sorted(op_dict.keys(), key=(lambda ax: tensor.shape[ax])))
        # ax_list = list(sorted(op_dict.keys()))
        mat_list = [op_dict[ax] for ax in ax_list]

        args = [(tensor, list(range(order)))]
        args.extend((mat_list[i], [order + i, ax_list[i]]) for i in range(n))
        ans_axes = [
            order + ax_list.index(ax) if ax in ax_list else ax
            for ax in range(order)
        ]
        args.append((ans_axes, ))

        ans = opt_einsum(*chain(*args))
        return ans

    def get_rdo(self, terminators=None) -> Array:
        axes = self.axes
        root = self.root
        terminate = self.terminate
        near = self.frame.near_points

        if terminators is None:
            terminators = dict(self.terminators)

        # Iterative from leaves to root (not include)
        for p in reversed(self.bfs_visitor[1:]):
            term_dict = {
                i: terminators[q]
                for i, q in enumerate(near(p)) if i != axes[p]
            }
            terminators[p] = terminate(self[p], term_dict)

        # root node: 0 and 1 observed for i and j and n_left
        term_dict = {
            i: terminators[q]
            for i, q in enumerate(near(root)) if i >= 2
        }
        # print(torch.norm(term_dict[2]))
        _array = terminate(self[root], term_dict)
        return _array.cpu().numpy()

    def numberer_op(self, k, power=1) -> dict[End, Array]:
        op = self.identity
        k_end = self._p_ends[k]
        numberer = self.bases[
            k_end].numberer_mat if k_end in self.bases else np.diag(
                np.arange(0, self._dims[k_end, 0], dtype=dtype))

        op[k_end] = np.linalg.matrix_power(numberer, power)
        return op

    def position_op(self, k, power=1) -> dict[End, Array]:
        op = self.identity
        k_end = self._p_ends[k]
        ap = self.bases[k_end].creation_mat if k_end in self.bases else np.diag(
            np.sqrt(np.arange(1, self._dims[k_end, 0], dtype=dtype)), k=-1)
        am = self.bases[
            k_end].annihilation_mat if k_end in self.bases else np.diag(
                np.sqrt(np.arange(1, self._dims[k_end, 0], dtype=dtype)), k=1)

        op[k_end] = np.linalg.matrix_power((ap + am) / np.sqrt(2.0), power)
        return op

    def momentum_op(self, k, power=1) -> dict[End, Array]:
        op = self.identity
        k_end = self._p_ends[k]
        ap = self.bases[k_end].creation_mat if k_end in self.bases else np.diag(
            np.sqrt(np.arange(1, self._dims[k_end, 0], dtype=dtype)), k=-1)
        am = self.bases[
            k_end].annihilation_mat if k_end in self.bases else np.diag(
                np.sqrt(np.arange(1, self._dims[k_end, 0], dtype=dtype)), k=1)

        op[k_end] = np.linalg.matrix_power(1.0j * (am - ap) / np.sqrt(2.0),
                                           power)
        return op

    def population_op(self, k, n_k):
        op = self.identity
        k_end = self._p_ends[k]
        dim = self._dims[k_end, 0]
        proj = np.zeros((dim, dim))
        proj[n_k][n_k] = 1.0
        op[k_end] = proj

        return op


class NaiveHierachy(Hierachy):

    def __init__(
        self,
        rdo: NDArray,
        dims: list[int],
        dvr_bases: Optional[dict[int,
                                 DiscreteVariationalRepresentation]] = None
    ) -> None:
        ends = [self.end('i'), self.end('j')
                ] + [self.end(k) for k in range(len(dims))]
        frame = Frame()
        root = Node(f'0')
        for e in ends:
            frame.add_link(root, e)
        super().__init__(rdo, dims, frame, root, dvr_bases=dvr_bases)
        return


class HeomOp:
    # An artificial parameter in the definition of ADMs.
    metric = 're'
    metric_order = 0
    rm_zeroes = False
    soft_boundary = False

    def __init__(self,
                 hierachy: Hierachy,
                 sys_hamiltonian: NDArray,
                 sys_op: NDArray,
                 correlation: Correlation,
                 dims: list[int],
                 lindblad_rate: None | float = None) -> None:
        self.bases = hierachy.bases
        self.end = hierachy.end
        self.h = sys_hamiltonian
        self.op = sys_op
        self.coefficients = correlation.coefficients
        self.conj_coefficents = correlation.conj_coefficents
        self.zeropoints = correlation.zeropoints
        self.derivatives = correlation.derivatives
        self.dims = dims

        op_list = self.op_list()
        if lindblad_rate is not None:
            fluc = lindblad_rate - sum(c.real for c in self.coefficients)
            print(f'Add Lindblad term with rate: {fluc:.8f}', flush=True)
            op_list.extend(self._lindblad_term(fluc))
        self.ops = SumProdOp(op_list)
        return

    def _lindblad_term(self, fluc_left: float):
        sys_op = self.op
        _lamb = 0.5 * fluc_left * sys_op @ sys_op
        ans = [{
            self.end('i'): -_lamb,
        }, {
            self.end('j'): -_lamb.conj(),
        }, {
            self.end('i'): fluc_left * sys_op,
            self.end('j'): sys_op.conj(),
        }]

        return ans

    def op_list(self, include_heom: bool = True) -> list[dict[End, NDArray]]:
        ans = [{
            self.end('i'): -1.0j * self.h
        }, {
            self.end('j'): 1.0j * self.h.conj()
        }]
        if include_heom:
            for k in range(len(self.dims)):
                ans.extend(self.bcf_term(k))
        else:
            for k in range(len(self.dims)):
                ans.extend(self.bcf_term(k)[2:])
        return ans

    def _adm_factor(self, k: int):
        ck = self.coefficients[k]
        cck = self.conj_coefficents[k]
        if self.metric == 're':
            fk = np.sqrt(abs(ck.real + cck.real) / 2)
            if (ck.imag + cck.imag) > EPSILON:
                fk *= -1.0
        elif self.metric == 'abs':
            fk = np.sqrt((abs(ck) + abs(cck)) / 2)
        else:
            fk = complex(self.metric)
        if __debug__:
            print(f'For k = {k}: ', flush=True)
            print(
                f's:{(ck.real + cck.real) / 2:.8f} | '
                f'e:{(ck.real - cck.real) / 2:.8f} | '
                f'a:{ck.imag:.8f}',
                flush=True,
            )
            print(f'f:{fk:.8f} | f^2:{fk**2:.8f}', flush=True)
        return fk

    @staticmethod
    def _soft_dim(dim):
        return dim - 1 - int(dim // 5)

    def bcf_term(self, k: int) -> list[dict[End, NDArray]]:
        k_end = self.end(k)
        ck = self.coefficients[k]
        cck = self.conj_coefficents[k]
        zk = self.zeropoints[k]
        dk = self.derivatives[k]
        dim = self.dims[k]
        if k_end in self.bases:
            raiser = self.bases[k_end].creation_mat
            lower = self.bases[k_end].annihilation_mat
            numberer = self.bases[k_end].numberer_mat
        else:
            raiser = np.diag(np.arange(
                1, dim, dtype=dtype)**((1 + self.metric_order) / 2),
                             k=-1)
            lower = np.diag(np.arange(
                1, dim, dtype=dtype)**((1 - self.metric_order) / 2),
                            k=1)
            numberer = np.diag(arange(dim))
            print(f"self.metric_order : {self.metric_order}")

        if self.rm_zeroes and abs(zk * ck) < EPSILON and abs(
                zk * cck) < EPSILON:
            ans = []
        elif abs(ck) < EPSILON and abs(cck) < EPSILON:
            ans = []
        else:
            fk = self._adm_factor(k)
            ans = [{
                self.end('i'): -1.0j * self.op,
                k_end: (ck * zk / fk * raiser + fk * lower)
            }, {
                self.end('j'): 1.0j * self.op.conj(),
                k_end: (cck * zk / fk * raiser + fk * lower)
            }]

        for j, dj in dk.items():
            j_end = self.end(j)
            if j_end is k_end:
                if self.soft_boundary:
                    # soft_decay = np.zeros_like(numberer)
                    # sd = 1
                    soft_decay = (lower - raiser) / np.sqrt(2)
                    array = dj * numberer + 0.1 * dj.real * np.linalg.matrix_power(
                        soft_decay, 4)
                else:
                    array = dj * numberer
                ans.append({k_end: array})
            else:
                _dj = np.sqrt(dj, dtype=complex)
                fj = self._adm_factor(j)
                if j_end in self.bases:
                    # raiser_j = self.bases[j_end].creation_mat
                    lower_j = self.bases[j_end].annihilation_mat
                else:
                    # raiser_j = np.diag(np.sqrt(np.arange(1, self.dims[j], dtype=dtype)), k=-1)
                    lower_j = np.diag(
                        np.arange(1, self.dims[j],
                                  dtype=dtype)**((1 - self.metric_order) / 2),
                        k=1)
                ans.append({
                    k_end: _dj / fk * raiser,
                    j_end: _dj * fj * lower_j
                })
        return ans
