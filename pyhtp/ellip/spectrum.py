# -*- coding: utf-8 -*-
"""
Define a class to store ellipsometry spectrum at a point.
"""
from __future__ import annotations
from typing import Literal, NamedTuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from ..typing import SpectrumInfo, BandGap


class EllipPointSet(NamedTuple):
    """Contains a crystalline and a amorphous set on the same point."""
    crystalline: EllipSpectrum
    amorphous: EllipSpectrum

    def __len__(self):
        return len(self.crystalline)

    def plot_tauc(
            self,
            exponent: float = 0.5,
            ax: Axes | None = None,
            **kwargs):
        """Plot tauc plot of both structures."""
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        index = self.crystalline.info.index
        self.crystalline.plot_tauc(exponent=exponent, ax=ax,
                                   label=f'crystalline-{index}', **kwargs)
        self.amorphous.plot_tauc(exponent=exponent, ax=ax,
                                 label=f'amorphous-{index}', **kwargs)

    @property
    def wavelength(self):
        """The wavelength array of the spectrum."""
        return self.crystalline.wavelength


class EllipSpectrum:
    """A class to store ellipsometry spectrum at a point."""

    def __init__(
            self,
            refrac: ArrayLike,
            extinc: ArrayLike,
            info: SpectrumInfo,
            wavelength: ArrayLike | None = None):
        """Initialize the class.

        Args:
            refrac (ArrayLike): Refractive index n.
            extinc (ArrayLike): Extinction coefficient k.
            wavelength (ArrayLike | None): Wavelength array (nm). If none, determine by the info.wavelength_range Defaults to None.
            info (SpectrumInfo): Info for the spectrum.
        """
        self.refrac = np.array(refrac)
        self.extinc = np.array(extinc)
        if len(self.refrac) != len(self.extinc):
            raise ValueError('The length of refractive index and extinction coefficient should be the same.')
        self.info = info
        if wavelength is None:
            self.wavelength = np.linspace(
                info.wavelength_range[0],
                info.wavelength_range[1],
                len(self))
        else:
            self.wavelength = np.array(wavelength)
        assert self.wavelength is not None
        self.complex = self.refrac + 1j * self.extinc
        self.absorp = 4 * np.pi * self.extinc / (self.wavelength * 1e-9)  # m^-1

    def __len__(self):
        return len(self.refrac)

    def plot(
            self,
            param: Literal['n', 'k', 'alpha', 'n & k', 'all'] = 'all',
            if_label: bool = True,
            if_legend: bool = True,
            ax: Axes | None = None,
            **kwargs) -> None:
        """Plot the spectrum.

        Args:
            param (Literal['n', 'k', 'alpha', 'n & k', 'all'], optional): Constants to plot. Defaults to 'all'.
            ax (Axes | None, optional): The axes of the plot. Defaults to None.
        """
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        if param == 'n':
            ax.plot(self.wavelength, self.refrac, label='n', **kwargs)
        elif param == 'k':
            ax.plot(self.wavelength, self.extinc, label='k', **kwargs)
        elif param == 'alpha':
            ax.plot(self.wavelength, self.absorp, label=r'$\alpha$', **kwargs)
        elif param == 'n & k':
            ax.plot(self.wavelength, self.refrac, label='n', **kwargs)
            ax.plot(self.wavelength, self.extinc, label='k', **kwargs)
        elif param == 'all':
            ax.plot(self.wavelength, self.refrac, label='n', **kwargs)
            ax.plot(self.wavelength, self.extinc, label='k', **kwargs)
            ax.plot(self.wavelength, self.absorp, label=r'$\alpha$', **kwargs)
        if if_label:
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Value')
        if if_legend:
            ax.legend(loc='best')

    def plot_tauc(
            self,
            exponent: float = 0.5,
            if_label: bool = True,
            ax: Axes | None = None,
            **kwargs) -> None:
        """Plot the Tauc plot.

        Args:
            exponent (float, optional): The exponent of the plot. 2 for direct allowed transitions; 2/3 for direct forbidden transitions; 0.5 for indirect allowed transitions; 1/3 for indirect forbidden transitions. Defaults to 0.5.
            if_label (bool, optional): If add axis label. Defaults to True.
            ax (Axes | None, optional): The axes of the plot. Defaults to None.
            **kwargs: Other arguments for ax.plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        freq = scipy.constants.c / (self.wavelength * 1e-9)
        x = scipy.constants.Planck * freq
        y = self.absorp * x ** exponent
        # Use ev as unit, original ones are use joule as unit.
        x /= scipy.constants.e
        y /= scipy.constants.e
        ax.plot(x, y, **kwargs)
        if if_label:
            ax.set_xlabel(r'$h\nu$ (eV)')
            ax.set_ylabel(r'$\alpha h\nu$ (eV)')

    @property
    def bandgap(self) -> BandGap:
        """Do linear regression on Tauc plot to get bandgap.

        Returns:
            Bandgap: The different types of bandgap.
        """
        bandgap = {
            'direct_allowed': .0,
            'direct_forbidden': .0,
            'indirect_allowed': .0,
            'indirect_forbidden': .0
        }
        for (exponent, key) in zip([2, 2 / 3, 0.5, 1 / 3], bandgap):
            freq = scipy.constants.c / (self.wavelength * 1e-9)
            x = scipy.constants.Planck * freq  # J
            y = self.absorp * x ** exponent * 1e2  # J/cm^2
            x /= scipy.constants.e  # eV
            y /= scipy.constants.e  # eV/cm^2
            # Fit the right 1/4 part of the curve.
            x_fit = x[len(x) * 3 // 4:]
            y_fit = y[len(y) * 3 // 4:]
            # Fit the curve.
            fit = np.polyfit(x_fit, y_fit, 1)
            # The intersection with x axis is the bandgap.
            bandgap[key] = -fit[1] / fit[0]
        return BandGap(**bandgap)

    def _fit_linear(
            self,
            exponent: float = 0.5,
            r2_tol: float = 0.99,
            angle_tol: float = 0.1,
            length_tol: float = 0.1) -> tuple[float, float]:
        """Fit the linear segment of the Tauc plot.

        Args:
            exponent (float, optional): The exponent of the plot. 2 for direct allowed transitions; 2/3 for direct forbidden transitions; 0.5 for indirect allowed transitions; 1/3 for indirect forbidden transitions. Defaults to 0.5.
            r2_tol (float, optional): The tolerance of R^2. Defaults to 0.99.
            angle_tol (float, optional): The tolerance of the angle between two linear segments (degree). Defaults to 0.1.

        Returns:
            tuple[float, float]: The bandgap and error.
        """
        freq = scipy.constants.c / (self.wavelength * 1e-9)
        x = scipy.constants.Planck * freq
        y = self.absorp * x ** exponent
        # Use ev as unit, original ones are use joule as unit.
        x /= scipy.constants.e
        y /= scipy.constants.e
        # Normalize y to the same scale as x.
        y = y / np.max(y) * (np.max(x) - np.min(x))

        # Recursively bisect the curve to find the linear segment.
        section_index: list[int] = []
        section_index.append(0)
        section_index.append(len(x) - 1)
        while any(_sections_r2(section_index, x, y) < r2_tol):
            current_r2 = _sections_r2(section_index, x, y)
            # Find the section with r2 < tol_r2.
            index = np.where(current_r2 < r2_tol)[0]
            # Bisect the section, add index to the section_index.
            for i in index:
                # If the section is too small, skip.
                if section_index[i + 1] - section_index[i] == 1:
                    continue
                section_index.append((section_index[i] + section_index[i + 1]) // 2)
            section_index.sort()

        # Merge the segments with similar inclination.
        # Calculate the inclination of each segment.
        inclination = np.deg2rad(
            np.arctan((y[section_index][1:] - y[section_index][:-1])
                      / (x[section_index][1:] - x[section_index][:-1])))
        delta_inclination = inclination[1:] - inclination[:-1]
        # Merge the segments with similar inclination.
        while (len(delta_inclination) > 0
               and np.abs(np.min(delta_inclination)) < angle_tol):
            min_index = np.argmin(delta_inclination)
            # Merge the two segments by remove
            # the middle point in the section_index.
            section_index.pop(min_index + 1)
            inclination = np.deg2rad(
                np.arctan((y[section_index][1:] - y[section_index][:-1])
                          / (x[section_index][1:] - x[section_index][:-1])))
            delta_inclination = inclination[1:] - inclination[:-1]

        # Identify Tauc segment and base line segment.
        # Detect Tauc segment based on 4 rules
        # 1. The number of points in the segment should be larger than 2.
        # 2. It's not the first or the last segment.
        # 3. It's inclination is larger than 0.
        # 4. It's inclination is larger then those of adjacent segments.
        tauc_index = []
        for i in range(len(section_index) - 2):
            if (section_index[i + 1] - section_index[i] > 2
                    and i not in [0, len(section_index) - 2]
                    and inclination[i] > max(inclination[i - 1], inclination[i + 1], 0)):
                tauc_index.append(i)
        # if len(tauc_index) == 0:
        #     raise ValueError('No Tauc segment found.')
        # Find the base line segment corresponding to the Tauc segment.
        # There are also 4 rules to identify the base line segment.
        # 1. The number of points in the segment should be larger than 2.
        # 2. the ratio of its length and that of the corresponding Tauc segment is larger than the minimum ratio
        # 3. Its inclination angle is lower than a half of its corresponding Tauc segment.
        # 4. It is located at the left side of its corresponding TS, but at the right of the last TS.
        pair_index: list[tuple[int, int]] = []  # (Tauc segment index, base line segment index)
        for tauc in tauc_index:
            base_index = []
            for j in range(len(section_index) - 2):
                if j == tauc:
                    continue
                base_length = np.linalg.norm([x[section_index[j + 1]] - x[section_index[j]],
                                              y[section_index[j + 1]] - y[section_index[j]]])
                tauc_length = np.linalg.norm([x[section_index[tauc + 1]] - x[section_index[tauc]],
                                              y[section_index[tauc + 1]] - y[section_index[tauc]]])
                if (section_index[j + 1] - section_index[j] > 2
                        and base_length / tauc_length > length_tol
                        and inclination[j] < inclination[tauc] / 2
                        and j < tauc):
                    base_index.append(j)
            if len(base_index) == 0:
                continue
            if len(base_index) > 1:
                # Choose the segment with the smallest inclination.
                base_index = [base_index[np.argmin(inclination[base_index])]]
            pair_index.append((tauc, base_index[0]))
        # if len(pair_index) == 0:
        #     raise ValueError('No base line segment and tauc segment pair found.')
        return section_index


def _sections_r2(
        section_index: list[int],
        x: NDArray,
        y: NDArray) -> NDArray:
    """Calculate the R^2 of each section."""
    result = np.zeros(len(section_index) - 1)
    for i, _ in enumerate(section_index):
        if i == len(section_index) - 1:
            break
        sec_x = x[section_index[i]:section_index[i + 1]]
        sec_y = y[section_index[i]:section_index[i + 1]]
        # Calculate the R^2 of the linear regression.
        corr = np.corrcoef(sec_x, sec_y)[0, 1]
        result[i] = corr ** 2
    return result