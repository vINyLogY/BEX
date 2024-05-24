# coding: utf-8

from math import sqrt

from bex.bath.correlation import BoseEinstein, Correlation
from bex.bath.sd import Drude, SpectralDensity, UnderdampedBrownian
from bex.libs.quantity import Quantity as __

inversed_temperature_unit = '/K'
time_unit = 'fs'
energy_unit = '/cm'


def gen_bcf(
    unit_energy: float,
    use_cross: bool,
    # Drude bath
    include_drude: bool,
    re_d: float,
    width_d: float,
    # Brownian bath
    include_brownian: bool,
    freq_b: list[float],
    re_b: list[float],
    width_b: list[float],
    # Vibrational bath
    include_discrete: bool,
    freq_v: list[float],
    re_v: list[float],
    # LTC bath
    temperature: float,
    decomposition_method: str,
    n_ltc: int,
    **kwargs,
) -> Correlation:

    energy_scale = __(unit_energy, energy_unit).au
    # Bath settings:
    corr = Correlation()
    if temperature is not None:
        beta = __(1 / temperature, inversed_temperature_unit).au * energy_scale
    else:
        beta = None
    BoseEinstein.decomposition_method = decomposition_method
    distr = BoseEinstein(n=n_ltc, beta=beta)

    sds = []  # type:list[SpectralDensity]
    if include_drude:
        for l, g in zip(re_d, width_d):
            drude = Drude(
                __(l, energy_unit).au / energy_scale,
                __(g, energy_unit).au / energy_scale)
            sds.append(drude)

    if include_brownian:
        for w, l, g in zip(freq_b, re_b, width_b):
            b = UnderdampedBrownian(
                __(l, energy_unit).au / energy_scale,
                __(w, energy_unit).au / energy_scale,
                __(g, energy_unit).au / energy_scale,
            )
            sds.append(b)
    if use_cross:
        corr.add_trigonometric(sds, distr)
    else:
        corr.add_spectral_densities(sds, distr)

    if include_discrete:
        for w, l in zip(freq_v, re_v):
            g = sqrt(l * w)
            if use_cross:
                _add_v = corr.add_discrete_trigonometric
            else:
                _add_v = corr.add_discrete_vibration
            _add_v(
                __(w, energy_unit).au / energy_scale,
                __(g, energy_unit).au / energy_scale, distr.beta)

    return corr
