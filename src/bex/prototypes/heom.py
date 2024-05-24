# coding: utf-8

from typing import Callable, Generator, Optional, Union

from bex.basis.dvr import SineDVR, SincDVR
from bex.bath.correlation import BoseEinstein, Correlation
from bex.heom.eom import HeomOp, Hierachy, NaiveHierachy
from bex.libs import backend
from bex.libs.backend import Array, OptArray, np
from bex.libs.logging import Logger
from bex.libs.quantity import Quantity as __
from bex.operator.spo import Propagator
from bex.state.frame import End

inversed_temperature_unit = '/K'
time_unit = 'fs'
energy_unit = '/cm'


def run_spin_boson(
    out: str,
    unit_energy: float,
    # System
    system_hamiltonian: Array,  # with unit
    system_operator: Array,  # without unit
    init_rdo: Array,
    # Bath
    corr: Correlation,
    rm_zeroes: bool,
    dvr_settings: dict[int, float],
    dim: int | list[int],
    # HEOM type
    heom_factor: None | str | float | tuple[float, float],
    # Error
    ode_method: str,
    ode_rtol: float,
    ode_atol: float,
    # Propagator
    dt: float,
    end: float,
    callback_steps: int,
    renormalize: bool,
    **kwargs,
) -> Generator[tuple[float, Hierachy], None, None]:

    backend.parameters.ode_rtol = ode_rtol
    backend.parameters.ode_atol = ode_atol

    energy_scale = __(unit_energy, energy_unit).au
    # System settings:
    h = backend.as_array(system_hamiltonian) * __(
        1, energy_unit).au / energy_scale
    op = backend.as_array(system_operator)
    init_rdo = backend.as_array(init_rdo)
    print(f"h: {h}", flush=True)
    print(f"op: {op}", flush=True)

    # HEOM settings:
    try:
        dims = list(dim)
    except:
        dims = [dim] * corr.k_max
    # print('Initial dims: ', dims, flush=True)
    dvr_bases = dict()
    for k, (_type, length) in dvr_settings.items():
        if _type == 'sinc':
            dvr_cls = SincDVR
        elif _type == 'sine':
            dvr_cls = SineDVR
        else:
            raise NotImplementedError(f"No basis named as {_type}.")
        dvr_bases[k] = dvr_cls(-length / 2.0, length / 2.0, dims[k])

    s = NaiveHierachy(init_rdo, dims, dvr_bases=dvr_bases)
    try:
        re, im = heom_factor
        HeomOp.metric = complex(re, im)
    except:
        if heom_factor is None:
            heom_factor = 'abs'
        HeomOp.metric = heom_factor
    HeomOp.rm_zeroes = rm_zeroes
    if kwargs.get('metric_order') is not None:
        HeomOp.metric_order = kwargs.get('metric_order')
    print(f" HeomOp.metric_order : {HeomOp.metric_order}", flush=True)
    heom_op = HeomOp(s, h, op, corr, dims, lindblad_rate=None)

    # Propagator settings:
    max_len = int(end / dt) + 1
    interval = __(dt, time_unit).au * energy_scale
    propagator = Propagator(heom_op.ops,
                            s,
                            interval,
                            callback_steps=callback_steps,
                            ode_method=ode_method,
                            renormalize=renormalize)

    logger = Logger(filename=out + '.dat.log', level='info').logger
    logger.info('# time \{rdo_ij\}')

    link_it = s.frame.node_link_visitor(s.root)
    indices = []
    saved = []
    for p, i, q, _ in link_it:
        if {p, q} not in saved:
            indices.append((p, i))
            saved.append({p, q})

    for _n, _t in enumerate(propagator):
        t = _t / energy_scale
        rdo = s.get_rdo()
        if renormalize:
            rdo *= propagator.renormalization_factor

        pops = np.diag(rdo)
        flat_rdo = rdo.reshape(-1)
        flat_conj_rdo = rdo.transpose().conj().reshape(-1)
        pr = np.dot(flat_rdo, flat_conj_rdo).real

        logger.info(f'{t}  ' + ' '.join([f'{_r:.8f}' for _r in flat_rdo]))

        time_fs = __(t).convert_to("fs").value

        yield (time_fs, s)

        cond = (any(pi.real > 1.01 or pi.real < -0.01 or abs(pi.imag) > 0.01
                    for pi in pops) or pr > 1.01 or pr < 0.01)
        if cond or _n >= max_len:
            break
    return
