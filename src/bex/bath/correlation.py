#!/usr/bin/env python
# coding: utf-8
"""
Correlation function object
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from bex.bath.distribution import BoseEinstein
from bex.bath.sd import SpectralDensity
from bex.libs.backend import PI, Array, as_array, np


class Correlation(object):

    def __init__(self) -> None:
        self.coefficients = list()  # type: list[complex]
        self.conj_coefficents = list()  # type: list[complex]
        self.zeropoints = list()  # type: list[complex]
        self.derivatives = list()  # type: list[dict[int, complex]]

        return

    def dump(self, output_file: str) -> None:
        with open(output_file, 'w') as f:
            c = [(_c.real, _c.imag) for _c in self.coefficients]
            cc = [(_cc.real, _cc.imag) for _cc in self.conj_coefficents]
            z = [(_z.real, _z.imag) for _z in self.zeropoints]
            d = [{
                j: (_d.real, _d.imag)
                for j, _d in di.items()
            } for di in self.derivatives]
            kwargs = {
                'coefficients': c,
                'conj_coefficents': cc,
                'zeropoints': z,
                'derivatives': d,
            }
            json.dump(kwargs, f, indent=4, sort_keys=True)
        return

    def load(self, input_file: str) -> None:
        with open(input_file, 'r') as f:
            kwargs = json.load(f)
            c = [complex(x, y) for x, y in kwargs['coefficients']]
            cc = [complex(x, y) for x, y in kwargs['conj_coefficents']]
            z = [complex(x, y) for x, y in kwargs['zeropoints']]
            d = [{
                int(j): complex(x, y)
                for j, (x, y) in di.items()
            } for di in kwargs['derivatives']]
            assert len(c) == len(cc) == len(z) == len(d)
            self.coefficients = c
            self.conj_coefficents = cc
            self.zeropoints = z
            self.derivatives = d
        return

    @property
    def k_max(self):
        assert len(self.coefficients) == len(self.coefficients) == len(
            self.derivatives) == len(self.zeropoints)
        return len(self.coefficients)

    def add_discrete_vibration(self, frequency: float, coupling: float,
                               beta: Optional[float]) -> None:
        w0 = frequency
        g = coupling

        coth = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0
        self.coefficients.extend(
            [g**2 / 2.0 * (coth + 1.0), g**2 / 2.0 * (coth - 1.0)])
        self.conj_coefficents.extend(
            [g**2 / 2.0 * (coth - 1.0), g**2 / 2.0 * (coth + 1.0)])
        self.zeropoints.extend([1.0, 1.0])
        k = len(self.derivatives)
        self.derivatives.extend([{k: -1.0j * w0}, {k + 1: 1.0j * w0}])

        return

    def add_discrete_trigonometric(self, frequency: float, coupling: float,
                                   beta: Optional[float]) -> None:
        w0 = frequency
        g = coupling

        coth = 1.0 / np.tanh(beta * w0 / 2.0) if beta is not None else 1.0
        c1 = g**2 / 2.0 * (coth + 1.0)
        c2 = g**2 / 2.0 * (coth - 1.0)
        self.coefficients.extend([(c2 + c1), (c2 - c1) * 1.0j])
        self.conj_coefficents.extend([(c2 + c1).conj(),
                                      ((c2 - c1) * 1.0j).conj()])
        self.zeropoints.extend([1.0, 0.0])  # cos * exp, sin * exp
        k = len(self.derivatives)
        self.derivatives.extend([
            {
                k + 1: -w0
            },
            {
                k: w0
            },
        ])
        return

    def _add_ltc(self, sds: list[SpectralDensity], distribution: BoseEinstein):
        """Add LTC terms for spectral densities with poles.
        """
        residue_pairs = distribution.residues
        if sds and residue_pairs:
            for res, pole in residue_pairs:
                cs = [-2.0j * PI * res * sd.function(pole) for sd in sds]
                c = np.sum(cs)
                self.coefficients.append(c)
                self.conj_coefficents.append(np.conj(c))
                self.zeropoints.append(1.0)
                k = len(self.derivatives)
                self.derivatives.append({k: -1.0j * pole})
        return

    def add_spectral_densities(self,
                               sds: list[SpectralDensity],
                               distribution: BoseEinstein,
                               zeropoint=None):
        f = distribution.function
        if zeropoint is None:
            zeropoint = 1.0
        for sd in sds:
            rs, ps = sd.get_residues_poles()
            if len(rs) == 1:
                c = rs[0] * f(ps[0])
                self.coefficients.append(c / zeropoint)
                self.conj_coefficents.append(c.conj() / zeropoint)
                self.zeropoints.append(zeropoint)
                k = len(self.derivatives)
                self.derivatives.append({k: -1.0j * ps[0]})
            elif len(rs) == 2:
                c1 = rs[0] * f(ps[0])
                c2 = rs[1] * f(ps[1])
                self.coefficients.extend([c1 / zeropoint, c2 / zeropoint])
                self.conj_coefficents.extend(
                    [c2.conj() / zeropoint,
                     c1.conj() / zeropoint])
                self.zeropoints.extend([zeropoint, zeropoint])
                k = len(self.derivatives)
                self.derivatives.extend([{
                    k: -1.0j * ps[0]
                }, {
                    k + 1: -1.0j * ps[1]
                }])

            else:
                raise NotImplementedError

        self._add_ltc(sds, distribution)
        return

    def add_trigonometric(self, sds: list[SpectralDensity],
                          distribution: BoseEinstein):
        f = distribution.function
        for sd in sds:
            rs, ps = sd.get_residues_poles()
            if len(rs) == 2:
                # ps = [-1.0j * (g + 1.0j * w), -1.0j * (g - 1.0j * w)]
                g = (ps[0] + ps[1]) * 0.5j
                w = (ps[0] - ps[1]) * 0.5
                c1 = rs[0] * f(ps[0])  # for term exp[(- iw - g) t]
                c2 = rs[1] * f(ps[1])  # for term exp[(+ iw - g) t]
                self.coefficients.extend([(c2 + c1), (c2 - c1) * 1.0j])
                self.conj_coefficents.extend([(c2 + c1).conj(),
                                              ((c2 - c1) * 1.0j).conj()])
                self.zeropoints.extend([1.0, 0.0])  # cos * exp, sin * exp
                k = len(self.derivatives)
                self.derivatives.extend([
                    {
                        k: -g,
                        k + 1: -w
                    },
                    {
                        k: w,
                        k + 1: -g
                    },
                ])
            elif len(rs) == 1:
                c = rs[0] * f(ps[0])
                self.coefficients.append(c)
                self.conj_coefficents.append(c.conj())
                self.zeropoints.append(1.0)
                k = len(self.derivatives)
                self.derivatives.append({k: -1.0j * ps[0]})
            else:
                raise NotImplementedError

        self._add_ltc(sds, distribution)
        return

    def __str__(self) -> None:
        if self.k_max > 0:
            string = f"Correlation {self.k_max} * ( c | c* | z | g ):"
            for c, cc, z, g in zip(self.coefficients, self.conj_coefficents,
                                   self.zeropoints, self.derivatives):
                string += f"\n{c.real:+.4e}{c.imag:+.4e}j | {cc.real:+.4e}{cc.imag:+.4e}j | {z} | {g}"
        else:
            string = 'Empty Correlation object'
        return string
