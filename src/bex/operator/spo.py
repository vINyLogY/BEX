# coding: utf-8
from __future__ import annotations
from itertools import chain
from math import prod
from typing import Callable, Generator, Literal, Optional

from bex.libs.backend import (MAX_EINSUM_AXES, Array, OptArray, eye, odeint,
                              opt_einsum, opt_sum, opt_array, opt_svd,
                              opt_pinv, opt_cat, opt_split, opt_stack, torch,
                              opt_dtype, np)
from bex.state.frame import End, Node, Point
from bex.state.model import CannonialModel
from bex.libs.utils import count_calls


class SumProdOp:

    def __init__(self, op_list: list[dict[End, Array]]) -> None:
        n_terms = len(op_list)

        dims = dict()
        for term in op_list:
            for e, a in term.items():
                assert a.ndim == 2 and a.shape[0] == a.shape[1]
                if e in dims:
                    assert dims[e] == a.shape[0]
                else:
                    dims[e] = a.shape[0]

        self.n_terms = n_terms
        self.op_list = op_list
        self.dims = dims  # type: dict[End, int]
        self._valuation = None
        return

    def _get_valuation(self):
        """Densify the op_list along DOFs to axis-0."""
        tensors = {
            e: [eye(dim, dim)] * self.n_terms
            for e, dim in self.dims.items()
        }
        for n, term in enumerate(self.op_list):
            for e, a in term.items():
                tensors[e][n] = a
        valuation = {
            e: opt_array(np.stack(a, axis=0))
            for e, a in tensors.items()
        }  # type: dict[End, OptArray]
        return valuation

    def __add__(self, other: SumProdOp) -> SumProdOp:
        cls = type(self)
        return cls(self.op_list + other.op_list)

    def __getitem__(self, key: End) -> OptArray:
        if self._valuation is None:
            self._valuation = self._get_valuation()

        return self._valuation[key]

    def dense(self, indices: Optional[list] = None):
        """dimension convension in the order of list 'indices'
        Slow and huge memory consumption!
        """
        if indices is None:
            indices = [
                key for key, _ in sorted(self.dims.items(), key=lambda x: x[1])
            ]
        else:
            assert set(indices) == set(self.dims.keys())
        dim = prod([self.dims[i] for i in indices])

        n = len(indices)
        args = [(self._valuation[i], [0, _i + 1, _i + n + 1])
                for _i, i in enumerate(indices)]
        args.append(
            ([_i + 1 for _i in range(n)] + [_i + n + 1 for _i in range(n)], ))

        args = list(chain(*args))
        ans = opt_einsum(*args).reshape(dim, dim)
        return ans.cpu().numpy()

    @property
    def ends(self) -> set[End]:
        return set(self.dims.keys())

    def expand(self, tensor: OptArray) -> OptArray:
        shape = list(tensor.shape)
        return tensor.unsqueeze(0).expand([self.n_terms] + shape)

    @staticmethod
    def reduce(tensors: OptArray) -> OptArray:
        return opt_sum(tensors, 0)

    @staticmethod
    def transforms(tensors: OptArray, op_dict: dict[int,
                                                    OptArray]) -> OptArray:
        order = tensors.ndim - 1
        n = len(op_dict)
        op_ax = order + n
        assert op_ax < MAX_EINSUM_AXES

        ax_list = list(
            sorted(op_dict.keys(), key=(lambda ax: tensors.shape[ax + 1])))
        mat_list = [op_dict[ax] for ax in ax_list]

        args = [(tensors, [op_ax] + list(range(order)))]
        args.extend(
            (mat_list[i], [op_ax, order + i, ax_list[i]]) for i in range(n))
        ans_axes = [op_ax] + [
            order + ax_list.index(ax) if ax in ax_list else ax
            for ax in range(order)
        ]
        args.append((ans_axes, ))

        args = list(chain(*args))
        ans = opt_einsum(*args)
        return ans

    @staticmethod
    def traces(tensors1: OptArray, tensors2: OptArray, ax: int) -> OptArray:
        order = tensors1.ndim - 1
        assert ax < order
        assert order + 2 < MAX_EINSUM_AXES

        op_ax = order
        i_ax = order + 1
        j_ax = order + 2

        axes1 = list(range(order))
        axes1[ax] = i_ax
        axes2 = list(range(order))
        axes2[ax] = j_ax
        return opt_einsum(tensors1, [op_ax] + axes1, tensors2, [op_ax] + axes2,
                          [op_ax, i_ax, j_ax])


class MasterEqn(object):
    r"""Solve the equation wrt Sum-of-Product operator:
        d/dt |state> = [op] |state>
    """
    reg_atol = 1.0e-4

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

    @staticmethod
    def trace(tensor1: OptArray, tensor2: OptArray, ax: int) -> OptArray:
        order = tensor1.ndim
        assert ax < order
        assert order + 1 < MAX_EINSUM_AXES

        i_ax = order
        j_ax = order + 1

        axes1 = list(range(order))
        axes1[ax] = i_ax
        axes2 = list(range(order))
        axes2[ax] = j_ax
        return opt_einsum(tensor1, axes1, tensor2, axes2, [i_ax, j_ax])

    def __init__(self, op: SumProdOp, state: CannonialModel) -> None:
        assert op.ends == state.ends
        self.op = op
        self.state = state

        self.mean_fields = dict()  # type: dict[tuple[Node, int], OptArray]

        # Temp for regularization
        self._reg_r = dict()  # type: dict[Node, OptArray]
        self._reg_s = dict()  # type: dict[Node, OptArray]
        self._reg_v = dict()  # type: dict[Node, OptArray]
        self._densities = dict()  # type: dict[Point, OptArray]
        # primitive
        dual = state.frame.dual
        for q in state.ends:
            p, i = dual(q, 0)
            self.mean_fields[(p, i)] = self.op[q]
        self._node_visitor = state.frame.node_visitor(start=state.root,
                                                      method='BFS')
        self._shape_list = []
        self._size_list = []
        self.update_size_list()
        return

    def update_size_list(self):
        self._shape_list = [self.state.shape(p) for p in self._node_visitor]
        # print(self._shape_list)
        self._size_list = [prod(s) for s in self._shape_list]
        return

    def vectorize(self, tensors: list[OptArray]) -> OptArray:
        return opt_cat([a.flatten() for a in tensors])

    def vector_split(self, vec: OptArray) -> list[OptArray]:
        tensors = opt_split(vec, self._size_list)
        return [a.reshape(s) for a, s in zip(tensors, self._shape_list)]

    def vector(self) -> OptArray:
        return self.vectorize([self.state[p] for p in self._node_visitor])

    def vector_update(self, vec: OptArray) -> None:
        update = self.state.opt_update
        for p, a in zip(self._node_visitor, self.vector_split(vec)):
            update(p, a)
        return

    def vector_eom(self,
                   method: str = 'svd') -> Callable[[OptArray], OptArray]:
        axes = self.state.axes
        orders = self.state.frame.order

        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        transform = self.transform
        trace = self.trace

        update = self.vector_update
        split = self.vector_split
        vectorize = self.vectorize
        visitor = self._node_visitor
        get_mf1 = self._get_mean_fields_type1
        mfs = self.mean_fields

        def _dd_svd(a: OptArray) -> OptArray:
            ans_list = []
            update(a)
            get_mf1()
            self._get_mean_fields_reg()
            reg_r = self._reg_r
            reg_s = self._reg_s
            reg_v = self._reg_v

            for p, a in zip(visitor, split(a)):
                order = orders(p)
                ax = axes[p]
                assert a.ndim == order

                op_list = {i: mfs[p, i] for i in range(order) if i != ax}
                if ax is not None:
                    s = reg_s[p]
                    s_r = self.reg_atol * torch.ones_like(s)
                    s = torch.maximum(s, s_r).to(opt_dtype)
                    op_list[ax] = reg_v[p].mH @ (1.0 / s).diag() @ reg_r[p]

                ans = reduce(transforms(expand(a), op_list))
                if ax is not None:
                    # Projection
                    projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                    ans -= projection

                ans_list.append(ans)
            return vectorize(ans_list)

        def _dd_pinv(a: OptArray) -> OptArray:
            ans_list = []
            update(a)
            get_mf1()
            self._get_mean_fields_type2()
            self._get_densities()
            dens = self._densities

            for p, a in zip(visitor, split(a)):
                order = orders(p)
                ax = axes[p]
                assert a.ndim == order

                op_list = {i: mfs[p, i] for i in range(order)}
                if ax is not None:
                    # Inversion
                    op_list[ax] = opt_pinv(dens[p]) @ op_list[ax]
                ans = reduce(transforms(expand(a), op_list))
                if ax is not None:
                    # Projection
                    projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                    ans -= projection

                ans_list.append(ans)
            return vectorize(ans_list)

        if method == 'svd':
            return _dd_svd
        elif method == 'pinv':
            return _dd_pinv
        else:
            raise NotImplementedError

    def node_eom(
            self,
            node: Node,
            method: Optional[str] = None) -> Callable[[OptArray], OptArray]:
        ax = self.state.axes[node]
        order = self.state.frame.order(node)
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        transform = self.transform
        trace = self.trace
        mfs = self.mean_fields

        def _dd(a: OptArray) -> OptArray:
            assert a.ndim == order
            assert ax is None
            op_list = {i: mfs[node, i] for i in range(order)}
            ans = reduce(transforms(expand(a), op_list))
            return ans

        def _dd_svd(a: OptArray) -> OptArray:
            assert a.ndim == order
            op_list = {i: mfs[node, i] for i in range(order) if i != ax}
            if ax is not None:
                # Inversion
                s = self._reg_s[node]
                reg = self.reg_atol * torch.ones_like(s)
                s = torch.maximum(s, reg)
                op_list[ax] = self._reg_v[node].mH @ (
                    1.0 / s).diag() @ self._reg_r[node]
            ans = reduce(transforms(expand(a), op_list))
            if ax is not None:
                # Projection
                projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                ans -= projection
            return ans

        def _dd_pinv(a: OptArray) -> OptArray:
            assert a.ndim == order
            op_list = {i: mfs[node, i] for i in range(order)}
            if ax is not None:
                # Inversion
                op_list[ax] = opt_pinv(self._densities[node]) @ op_list[ax]
            ans = reduce(transforms(expand(a), op_list))
            if ax is not None:
                # Projection
                projection = transform(a, {ax: trace(ans, a.conj(), ax)})
                ans -= projection
            return ans

        if method is None:
            return _dd
        elif method == 'svd':
            return _dd_svd
        elif method == 'pinv':
            return _dd_pinv
        else:
            raise NotImplementedError

    def node_eom_op(self, node: Node) -> OptArray:
        if node is not self.state.root:
            raise NotImplementedError

        a = self.state[node]
        dims = a.shape
        order = a.ndim

        ax_list = list(sorted(range(order), key=(lambda ax: dims[ax])))
        mat_list = [self.mean_field(node, ax) for ax in ax_list]

        op_ax = 2 * order

        from_axes = list(range(order))
        to_axes = list(range(order, 2 * order))

        args = [(mat_list[i], [op_ax, order + ax_list[i], ax_list[i]])
                for i in range(order)]
        args.append((to_axes + from_axes, ))
        diff = opt_einsum(*chain(*args))

        return diff

    def _mean_field_with_node(self, p: Node, i: int) -> OptArray:
        order = self.state.frame.order(p)

        a = self.op.expand(self.state[p])
        conj_a = a.conj()
        op_list = {
            _i: self.mean_fields[(p, _i)]
            for _i in range(order) if _i != i
        }
        return self.op.traces(conj_a, self.op.transforms(a, op_list), ax=i)

    def _get_densities(self) -> None:
        self._densities = self.state.calc_densities()
        return

    def _get_mean_fields_type1(self) -> None:
        """From leaves to the root."""
        axes = self.state.axes
        dual = self.state.frame.dual
        mf = self._mean_field_with_node

        for q in reversed(self._node_visitor):
            j = axes[q]
            if j is not None:
                p, i = dual(q, j)
                self.mean_fields[(p, i)] = mf(q, j)
        return

    def _get_mean_fields_type2(self) -> None:
        """From root to leaves."""
        axes = self.state.axes
        dual = self.state.frame.dual
        mf = self._mean_field_with_node

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                self.mean_fields[(p, i)] = mf(q, j)
        return

    def _get_mean_fields_reg(self) -> None:
        """From root to leaves."""
        axes = self.state.axes
        dual = self.state.frame.dual

        def regularize(p: Node, i: int) -> tuple[OptArray, ...]:
            order = self.state.frame.order(p)
            ax = axes[p]
            _a = self.state[p]
            if ax is not None:
                s = self._reg_s[p].to(opt_dtype)
                _a = self.transform(_a, {ax: s.diag() @ self._reg_v[p]})
            shape = list(_a.shape)
            dim = shape.pop(i)
            u, s, v = opt_svd(_a.moveaxis(i, -1).reshape((-1, dim)),
                              compressed=False)
            u = u.reshape(shape + [-1]).moveaxis(-1, i)

            op_list = {
                _i: self.mean_fields[(p, _i)]
                for _i in range(order) if _i != i and _i != ax
            }
            if ax is not None:
                op_list[ax] = self._reg_r[p]

            a = self.op.expand(self.state[p])
            conj_u = self.op.expand(u.conj())
            r = self.op.traces(conj_u, self.op.transforms(a, op_list), ax=i)

            return r, s, v

        for p in self._node_visitor:
            i = axes[p]
            if i is not None:
                q, j = dual(p, i)
                r, s, v = regularize(q, j)
                self._reg_r[p] = r
                self._reg_s[p] = s
                self._reg_v[p] = v
        return


class Propagator:
    max_steps = 1_000_000_000
    k_lanczos = 10
    mix_rank_threshold = 40
    reg_atol = 1.0e-4
    first_mix_method = 'ps2'
    second_mix_method = 'vmf'

    def __init__(
            self,
            op: SumProdOp,
            state: CannonialModel,
            dt: float,
            callback_steps: int = 1,  # how many steps performed during dt
            ode_method: Literal['bosh3', 'dopri5', 'dopri8'] = 'dopri5',
            reg_method: Literal['svd', 'pinv'] = 'svd',
            ps_method: Literal['cmf', 'ps1', 'ps2', 'vmf', 'mix'] = 'vmf',
            renormalize: bool = False) -> None:

        self.op = op
        MasterEqn.reg_atol = self.reg_atol
        self.eom = MasterEqn(op, state)
        self.state = self.eom.state

        self.dt = dt
        self.callback_steps = callback_steps
        self.ode_method = ode_method
        self.ps_method = ps_method
        self.reg_method = reg_method

        self._node_visitor = self.state.frame.node_visitor(self.state.root)
        self._node_link_visitor = self.state.frame.node_link_visitor(
            self.state.root)
        self._depths = self.state.depths
        self._move1 = self.state.split_unite_move
        self._move2 = self.state.unite_split_move

        self.ode_step_counter = None

        # cumm renormalization_factor
        if renormalize:
            self.renormalization_factor = 1.0
        else:
            self.renormalization_factor = None
        return

    def __iter__(self) -> Generator[float, None, None]:
        r = self.state.root
        yield (0.0)
        for i in range(1, self.max_steps):
            self.ode_step_counter = 0
            self.step()

            if self.renormalization_factor is not None:
                z = np.sqrt(self.state.get_normsq())
                self.state.opt_update(r, self.state[r] / z)
                self.renormalization_factor *= z

            yield (self.dt * i)
        return

    def step(self) -> None:
        # print({n:self.state.shape(n) for  n in self._node_visitor})
        ps_method = self.ps_method
        eom = self.eom
        if ps_method == 'mix':
            s = self.state
            link_it = s.frame.node_link_visitor(s.root)
            ranks = [s._dims[p, i] for p, i, _, _ in link_it]
            assert ranks
            max_rank = max(ranks)
            if max_rank < self.mix_rank_threshold:
                ps_method = self.first_mix_method
            else:
                ps_method = self.second_mix_method
                # recalcuate shapes and size for fixed rank method
                self.ps_method = ps_method
                self.eom.update_size_list()
                #print(f"# Fixed method size: {s.get_size()}", flush=True)
            # print(ps_method)

        if ps_method == 'vmf':
            y = eom.vector()
            ans = self._odeint(self.eom.vector_eom(method=self.reg_method), y,
                               1.0)
            self.eom.vector_update(ans)
        else:
            callback_steps = self.callback_steps
            it = range(callback_steps)
            if ps_method == 'cmf':
                for _ in it:
                    eom._get_mean_fields_type1()
                    eom._get_mean_fields_reg()
                    eom._get_densities()
                    for p in self._node_visitor:
                        self._node_step(p,
                                        1.0 / callback_steps,
                                        method=self.reg_method)
            elif ps_method == 'ps1':
                for _ in it:
                    eom._get_mean_fields_type1()
                    self.ps1_forward_step(0.5 / callback_steps)
                    self._node_step(self.state.root, 1.0 / callback_steps)
                    self.ps1_backward_step(0.5 / callback_steps)
            elif ps_method == 'ps2':
                for _ in it:
                    eom._get_mean_fields_type1()
                    self.ps2_forward_step(0.5 / callback_steps)
                    # node_step with ratio-1 is included in
                    # ps2_forward_step and ps2_backward_step
                    self.ps2_backward_step(0.5 / callback_steps)
            else:
                raise NotImplementedError(
                    f'No Projector splitting method `{self.ps_method}`.')
        return

    def _node_step(self, p: Node, ratio: float, method=None) -> None:
        ans = self._odeint(self.eom.node_eom(p, method=method), self.state[p],
                           ratio)
        self.state.opt_update(p, ans)
        return

    def ps1_forward_step(self, ratio: float) -> None:
        move = self._move1
        depths = self._depths
        node_step = self._node_step
        op1 = self._ps1_mid_op

        links = self._node_link_visitor
        for p, i, q, j in links:
            assert p is self.state.root
            if depths[p] < depths[q]:
                move(i, op=op1(p, i, q, j, None))
            else:
                node_step(p, ratio)
                move(i, op=op1(p, i, q, j, -ratio))
        return

    def ps1_backward_step(self, ratio: float) -> None:
        move = self._move1
        depths = self._depths
        node_step = self._node_step
        op1 = self._ps1_mid_op

        links = self._node_link_visitor
        for q, j, p, i in reversed(links):
            assert p is self.state.root
            if depths[p] < depths[q]:
                move(i, op=op1(p, i, q, j, -ratio))
                node_step(q, ratio)
            else:
                move(i, op=op1(p, i, q, j, None))
        return

    def _ps1_mid_op(self, p: Node, i: int, q: Node, j: int,
                    ratio: Optional[float]) -> Callable[[OptArray], OptArray]:
        """Assue the tensor for p in self.state has been updated."""

        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        get_mf = self.eom._mean_field_with_node
        mfs = self.eom.mean_fields
        odeint = self._odeint

        def _op(mid: OptArray) -> OptArray:
            l_mat = get_mf(p, i)
            r_mat = mfs[p, i]
            mfs[q, j] = l_mat
            del mfs[p, i]
            if ratio is None:
                return mid
            else:
                op_dict = {0: l_mat, 1: r_mat}
                _dd = lambda a: reduce(transforms(expand(a), op_dict))
                return odeint(_dd, mid, ratio)

        return _op

    def ps2_forward_step(self, ratio: float) -> None:
        depths = self._depths
        mfs = self.eom.mean_fields
        get_mf = self.eom._mean_field_with_node
        move2 = self._move2
        move1 = self._move1
        node_step = self._node_step
        op2 = self._ps2_mid_op
        op1 = self._ps1_mid_op

        links = self._node_link_visitor
        end = len(links) - 1
        for n, (p, i, q, j) in enumerate(links):
            assert p is self.state.root
            if depths[p] < depths[q]:
                move1(i, op=op1(p, i, q, j, None))
            else:
                move2(i, op=op2(p, i, q, j, ratio), compressed=True)
                mfs[q, j] = get_mf(p, i)
                del mfs[p, i]
                if n != end:
                    node_step(q, -ratio)
        return

    def ps2_backward_step(self, ratio: float) -> None:
        depths = self._depths
        mfs = self.eom.mean_fields
        get_mf = self.eom._mean_field_with_node
        move2 = self._move2
        move1 = self._move1
        node_step = self._node_step
        op2 = self._ps2_mid_op
        op1 = self._ps1_mid_op

        links = self._node_link_visitor
        start = 0
        for n, (q, j, p, i) in enumerate(reversed(links)):
            assert p is self.state.root
            if depths[p] < depths[q]:
                if n != start:
                    node_step(p, -ratio)
                move2(i, op=op2(p, i, q, j, ratio), compressed=True)
                mfs[q, j] = get_mf(p, i)
                del mfs[p, i]
            else:
                move1(i, op=op1(p, i, q, j, None))
        return

    def _ps2_mid_op(self, p: Node, i: int, q: Node, j: int,
                    ratio: float) -> Callable[[OptArray], OptArray]:
        expand = self.op.expand
        transforms = self.op.transforms
        reduce = self.op.reduce
        get_order = self.state.frame.order
        mfs = self.eom.mean_fields
        odeint = self._odeint

        def _op(mid: OptArray) -> OptArray:
            ord_p = get_order(p)
            ord_q = get_order(q)
            l_ops = [mfs[p, _i] for _i in range(ord_p) if _i != i]
            r_ops = [mfs[q, _j] for _j in range(ord_q) if _j != j]
            op_dict = dict(enumerate(l_ops + r_ops))

            _dd = lambda a: reduce(transforms(expand(a), op_dict))
            return odeint(_dd, mid, ratio)

        return _op

    def _odeint(self, func: Callable[[OptArray], OptArray], y0: OptArray,
                ratio: float) -> OptArray:

        # vectorized func
        @count_calls
        def _grad(t, y0):
            re, im = y0
            d = func((re + 1.0j * im))
            return opt_stack([d.real, d.imag])
            # return func(y0)

        y0 = opt_stack([y0.real, y0.imag])
        ans = odeint(_grad, y0, ratio * self.dt, method=self.ode_method)
        self.ode_step_counter += _grad.calls
        re, im = ans
        ans = (re + 1.0j * im)
        return ans
