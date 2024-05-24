# coding: utf-8

import os
import sys
import json as json
from tqdm import tqdm
from bex.libs import backend
from bex.libs.logging import Logger

from bex.prototypes.heom import run_spin_boson
from bex.prototypes.bath import gen_bcf
from bex.libs.quantity import Quantity as __

energy_unit = '/cm'

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    try:
        name = sys.argv[1]
    except IndexError:
        raise RuntimeError(
            'Need one command line argument as the input/output file basename.'
        )
    with open(name, 'r') as f:
        kwargs = json.load(f)  # type: dict
    out = os.path.splitext(name)[0]
    corr = gen_bcf(**kwargs)
    print(corr, flush=True)
    if kwargs.get('use_dvr', False):
        dvr_settings = {
            k: (kwargs['dvr_type'], kwargs['dvr_length'])
            for k in range(corr.k_max)
        }
    else:
        dvr_settings = {}

    steps = int(kwargs['end'] / kwargs['dt']) + 1
    unit_energy = kwargs['unit_energy']

    p0 = 0.5
    e = kwargs['elec_bias']
    v = kwargs['elec_coupling']
    h = backend.as_array([[-p0 * e, v], [v, (1.0 - p0) * e]])
    op = backend.as_array([[-p0, 0.0], [0.0, (1.0 - p0)]])

    propagator = run_spin_boson(out=out,
                                system_hamiltonian=h,
                                system_operator=op,
                                corr=corr,
                                dvr_settings=dvr_settings,
                                **kwargs)
    progress_bar = tqdm(propagator, total=steps)

    k_max = corr.k_max
    logger = Logger(filename=out + '.num.log', level='info').logger
    logger.info(r'# t[fs]  |EDO|^2 ' + f' <N> x{k_max} ' + f' <N^2> x{k_max} ')

    for (_t, _s) in (progress_bar):
        progress_bar.set_description(f'@{_t:.2f} fs')
        if kwargs['use_dvr'] == False:
            n2 = _s.get_normsq()
            msg = f'{_t}  {n2}'
            for k in range(k_max):
                op = _s.numberer_op(k)  # | position_op(k) | momentum_op(k)
                ev1 = _s.ev(op)
                msg += f"  {ev1}"
                ev2 = _s.ev(_s.numberer_op(k, power=2))
                msg += f"  {ev2}"
            logger.info(msg)
