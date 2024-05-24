#!/usr/bin/env python
# coding: utf-8
"""
Spectral density factory
"""
from __future__ import annotations
from typing import Optional

from bex.bath.distribution import BoseEinstein

from bex.libs.backend import PI, np
from math import erf, exp, sqrt
from scipy.integrate import simpson, quad

from bex.libs.quantity import Quantity


class SpectralDensity:
    """
    Template for a spectral density.
    """
    FREQ_MIN = 1e-14
    FREQ_MAX = 1e4

    # print(FREQ_MAX, FREQ_MIN)

    def autocorrelation(self,
                        t: float,
                        beta: Optional[float] = None) -> complex:

        def _re(w):
            if beta is None:
                coth = 1.0
            else:
                coth = 1.0 / np.tanh(beta * w / 2.0)
                # coth = 2.0 / (beta * w)
            return self.function(w) * np.cos(w * t) * coth

        def _im(w):
            return -self.function(w) * np.sin(w * t)

        # w = np.logspace(np.log2(self.FREQ_MIN),
        #                 np.log2(self.FREQ_MAX),
        #                 num=100_000,
        #                 base=2)
        w = np.linspace(self.FREQ_MIN, self.FREQ_MAX, num=100_000)
        re = simpson(_re(w), w)
        im = simpson(_im(w), w)

        # re = quad(_re, self.FREQ_MIN, self.FREQ_MAX)[0]
        # im = quad(_im, self.FREQ_MIN, self.FREQ_MAX)[0]
        return re + 1.0j * im

    def function(self, w: complex) -> complex:
        pass

    def get_residues_poles(self) -> tuple[list[complex], list[complex]]:
        """
        Returns:
            residues : complex
            poles : complex
        """
        pass


class Drude(SpectralDensity):

    def __init__(self, reorganization_energy: float, relaxation: float) -> None:
        self.l = reorganization_energy
        self.g = relaxation
        return

    def function(self, w: complex) -> complex:
        l = self.l
        g = self.g
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def get_residues_poles(
            self) -> tuple[list[complex], list[complex], list[complex]]:
        return [-2.0j * self.l * self.g], [-1.0j * self.g]


class OhmicExp(SpectralDensity):

    def __init__(self, reorganization_energy: float, cutoff: float) -> None:
        self.eta = reorganization_energy / cutoff
        self.l = reorganization_energy
        self.g = cutoff
        return

    def function(self, w: complex) -> complex:
        eta = self.eta
        g = self.g
        return eta * w * np.exp(-w / g)

    def get_residues_poles(
            self) -> tuple[list[complex], list[complex], list[complex]]:
        raise NotImplementedError


class UnderdampedGaussian(SpectralDensity):

    def __init__(self, reorganization_energy: float, frequency: float,
                 relaxation: float) -> None:
        self.omega = frequency
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        w1 = self.omega
        return l / sqrt(2.0 * PI) / g * w * (np.exp(-0.5 * (
            (w - w1) / g)**2) + np.exp(-0.5 * ((w + w1) / g)**2))

    def get_residues_poles(
            self) -> tuple[list[complex], list[complex], list[complex]]:
        raise NotImplementedError


class UnderdampedBrownian(SpectralDensity):

    def __init__(self, reorganization_energy: float, frequency: float,
                 relaxation: float) -> None:
        self.omega = frequency
        self.gamma = relaxation
        self.lambda_ = reorganization_energy
        return

    def function(self, w: complex) -> complex:
        l = self.lambda_
        g = self.gamma
        w1 = self.omega
        return (4.0 / PI) * l * g * (w1**2 + g**2) * w / (
            (w + w1)**2 + g**2) / ((w - w1)**2 + g**2)

    def get_residues_poles(
            self) -> tuple[list[complex], list[complex], list[complex]]:
        l = self.lambda_
        g = self.gamma
        w = self.omega

        a = l * (w**2 + g**2) / w

        residues = [a, -a]
        # [-g - 1j * w, -g + 1j * w]
        poles = [-1.0j * (g + 1.0j * w), -1.0j * (g - 1.0j * w)]
        return residues, poles


if __name__ == '__main__':
    from bex.libs.quantity import Quantity as __
    from matplotlib import pyplot as plt
    unit = __(1000, '/cm').au
    # print('Pseudo g:', sd.lambda_**2 / sd.omega)
    beta = __(1 / 300, '/K').au * unit
    sd = UnderdampedBrownian(0.2, 0.5, 0.01)
    be = BoseEinstein(beta=beta)

    fig, ax = plt.subplots(1, 2, tight_layout=True)
    time_max = 100
    time_max_fs = __(time_max / unit).convert_to('fs').value
    time_space = np.linspace(0, time_max)
    time_space_fs = np.linspace(0, time_max_fs)
    ct_ref = np.array([sd.autocorrelation(t, beta=be.beta) for t in time_space])
    ax[0].plot(time_space_fs, ct_ref.real, 'r:', label='Ref.', lw=3)
    ax[1].plot(time_space_fs, ct_ref.imag, 'r:', label='Ref.', lw=3)
    ax[1].legend()
    plt.show()