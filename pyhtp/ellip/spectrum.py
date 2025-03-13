# -*- coding: utf-8 -*-
"""
This module contains the class for a single ellipsometry spectrum.
- Author: Junyuan Lu
- E-mail: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from ..typing import SpectrumInfo, BandGap, BandGapLegacy


class EllipSpectrum:
    """A class to process a single ellipsometry spectrum.

    Attributes:
        n (NDArray): Refractive index n.
        k (NDArray): Extinction coefficient k.
        info (SpectrumInfo): Info for the spectrum.
        wavelength (NDArray): Wavelength array (nm).
        complex_n (NDArray): Complex refractive index n + ik.
        absorp (NDArray): Absorption coefficient alpha (m^-1). alpha = 4 * pi * k / lambda.
    """

    def __init__(
            self,
            n: NDArray[np.float64] | list[float],
            k: NDArray[np.float64] | list[float],
            info: SpectrumInfo,
            wavelength: NDArray[np.float64] | list[float] | None = None):
        """Create an instance of EllipSpectrum.

        The class does not support import from a file. The refractive index and extinction coefficient
        should be provided.

        Args:
            n (NDArray[np.float64] | list[float]): Refractive index n.
            k (NDArray[np.float64] | list[float]): Extinction coefficient k.
            wavelength (NDArray[np.float64] | list[float] | None): Wavelength array (nm). If none,
                determine by the info.wavelength_range. Defaults to None.
            info (SpectrumInfo): Info for the spectrum. Please refer to the docstring of SpectrumInfo.
                Contains the name, wavelength range, element, temperature, and index of the spectrum.
        """
        # Check if the length of refrac, extinc, and wavelength are the same.
        if (len(n) != len(k)) or (wavelength is not None and len(n) != len(wavelength)):
            raise ValueError('The length of refrac, extinc, and wavelength should be the same.')
        # Convert to numpy array.
        self.n = np.array(n)
        self.k = np.array(k)
        self.info = info
        if wavelength is None:
            self.wavelength = np.linspace(
                info.wavelength_range[0],
                info.wavelength_range[1],
                len(self))
        else:
            self.wavelength = np.array(wavelength)
        assert isinstance(self.wavelength, np.ndarray)
        self.complex_n = self.n + 1j * self.k
        self.absorp = 4 * np.pi * self.k / (self.wavelength * 1e-9)  # m^-1

    def __len__(self):
        return len(self.n)

    def plot(
            self,
            data_type: Literal['n', 'k', 'alpha', 'n & k', 'all'] = 'all',
            if_label: bool = True,
            ax: Axes | None = None,
            **kwargs) -> None:
        """Plot the spectrum of specific data.

        Args:
            data_type (Literal['n', 'k', 'alpha', 'n & k', 'all'], optional): The type of data to plot.
                'n' for refractive index n; 'k' for extinction coefficient k;
                'alpha' for absorption coefficient alpha; 'n & k' for n and k;
                'all' for all of them. Defaults to 'all'.
            if_label (bool, optional): If add axis label. Defaults to True.
            ax (Axes | None, optional): The axes of the plot. Defaults to None.
            **kwargs: Other arguments for ``matplotlib.pyplot.plot`` and ``matplotlib.pyplot.legend``.
                For only 'n', 'k', or 'alpha', the label of the plot can be set by 'label'.
        """
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        if data_type == 'n':
            ax.plot(self.wavelength, self.n,
                    label=kwargs.pop('label', 'n'), **kwargs)
        elif data_type == 'k':
            ax.plot(self.wavelength, self.k,
                    label=kwargs.pop('label', 'k'), **kwargs)
        elif data_type == 'alpha':
            ax.plot(self.wavelength, self.absorp,
                    label=kwargs.pop('label', r'$\alpha$'), **kwargs)
        elif data_type == 'n & k':
            ax.plot(self.wavelength, self.n, label='n', **kwargs)
            ax.plot(self.wavelength, self.k, label='k', **kwargs)
        elif data_type == 'all':
            ax.plot(self.wavelength, self.n, label='n', **kwargs)
            ax.plot(self.wavelength, self.k, label='k', **kwargs)
            ax.plot(self.wavelength, self.absorp, label=r'$\alpha$', **kwargs)
        if if_label:
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Value')

    def plot_tauc(
            self,
            exponent: float = 0.5,
            if_label: bool = True,
            ax: Axes | None = None,
            **kwargs) -> None:
        """Plot the Tauc plot.

        Args:
            exponent (float, optional): The exponent of the plot. 2 for direct allowed transitions;
                2/3 for direct forbidden transitions; 0.5 for indirect allowed transitions;
                1/3 for indirect forbidden transitions. Defaults to 0.5.
            if_label (bool, optional): If add axis label. Defaults to True.
            ax (Axes | None, optional): The axes of the plot. Defaults to None.
            **kwargs: Other arguments for ``matplotlib.pyplot.plot``. The label of the plot can be
                set by 'label', which defaults to 'Exponent = {exponent}'.
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
        ax.plot(x, y, label=kwargs.pop('label', f'Exponent = {exponent}'), **kwargs)
        if if_label:
            ax.set_xlabel(r'$h\nu$ (eV)')
            ax.set_ylabel(r'$\alpha h\nu$ (eV)')

    def bandgap_legacy(self, fit_fraction: float = 0.25) -> BandGapLegacy:
        """Do linear regression on Tauc plot to get bandgap.

        This is the legacy version of bandgap calculation. It is not recommended to use this method.
        The function just perform linear fit on the right part of the curve, and the intersection
        with x axis is the bandgap.

        Args:
            fit_fraction (float, optional): The right fraction of the curve to fit. Defaults to 0.25.

        Returns:
            BandGapLegacy: The different types of bandgap. Please refer to the docstring of BandGapLegacy.
                In this version, the error is not provided.
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
            # Fit the right fraction of the curve.
            start_index = int(len(x) * (1 - fit_fraction))
            x_fit = x[start_index:]
            y_fit = y[start_index:]
            # Fit the curve.
            fit = np.polyfit(x_fit, y_fit, 1)
            # The intersection with x axis is the bandgap.
            bandgap[key] = -fit[1] / fit[0]
        return BandGapLegacy(**bandgap)

    def bandgap(
            self,
            r2_tol: float = 0.99,
            angle_tol: float = 0.1,
            length_tol: float = 0.1) -> BandGap:
        """Do linear regression on Tauc plot to get bandgap.

        This is the recommended version of bandgap calculation. The function will recursively bisect
        the curve to find the linear segment. Then merge the segments with similar inclination.
        Identify Tauc segment and base line segment. Linear fit the Tauc segment and base line segment.
        Please refer to 10.1016/j.heliyon.2019.e01505 for more information.

        Returns:
            BandGap: The different types of bandgap.
        """
        bandgap = {
            'direct_allowed': (0., 0.),
            'direct_forbidden': (0., 0.),
            'indirect_allowed': (0., 0.),
            'indirect_forbidden': (0., 0.)
        }
        for (exponent, key) in zip([2, 2 / 3, 0.5, 1 / 3], bandgap):
            freq = scipy.constants.c / (self.wavelength * 1e-9)
            x = scipy.constants.Planck * freq  # J
            y = self.absorp * x ** exponent * 1e2  # J/cm^2
            x /= scipy.constants.e  # eV
            y /= scipy.constants.e  # eV/cm^2
            value, error = self._get_tauc_bg(exponent, r2_tol, angle_tol, length_tol)
            # The intersection with x axis is the bandgap.
            bandgap[key] = (value, error)
        return BandGap(**bandgap)

    @staticmethod
    def fit_linear_segment(
            x: NDArray,
            y: NDArray,
            r2_tol: float,
            angle_tol: float) -> tuple[list[int], NDArray]:
        """Fit the linear segment of a curve.

        Args:
            x (NDArray): _description_
            y (NDArray): _description_
            exponent (float): _description_
            r2_tol (float): _description_
            angle_tol (float): _description_

        Returns:
            tuple[float, float]: _description_
        """
        x = x.copy()
        y = y.copy()
        # Normalize y to the same scale as x.
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * (np.max(x) - np.min(x))

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
        inclination = np.rad2deg(
            np.arctan((y[section_index][1:] - y[section_index][:-1])
                      / (x[section_index][1:] - x[section_index][:-1])))
        delta_inclination = inclination[1:] - inclination[:-1]
        # Merge the segments with similar inclination.
        while (len(delta_inclination) > 0
               and np.min(np.abs(delta_inclination)) < angle_tol):
            min_index = np.argmin(delta_inclination)
            # Merge the two segments by remove
            # the middle point in the section_index.
            section_index.pop(min_index + 1)
            inclination = np.rad2deg(
                np.arctan((y[section_index][1:] - y[section_index][:-1])
                          / (x[section_index][1:] - x[section_index][:-1])))
            delta_inclination = inclination[1:] - inclination[:-1]
        return section_index, inclination

    def _get_tauc_bg(
            self,
            exponent: float,
            r2_tol: float,
            angle_tol: float,
            length_tol: float) -> tuple[float, float]:
        """Fit the linear segment of the Tauc plot."""
        freq = scipy.constants.c / (self.wavelength * 1e-9)
        x = scipy.constants.Planck * freq
        y = self.absorp * x ** exponent
        # Use ev as unit, original ones are use joule as unit.
        x /= scipy.constants.e
        y /= scipy.constants.e

        # Normalize y to the same scale as x.
        y_original = y
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * (np.max(x) - np.min(x))

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
        if len(tauc_index) == 0:
            raise ValueError('No Tauc segment found.')

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
        if len(pair_index) == 0:
            raise ValueError('No base line segment and tauc segment pair found.')

        # Recover the original scale of y before linear fit.
        y = y_original

        # Linear fit the Tauc segment and base line segment.
        for tauc, base in pair_index:

            fit_tauc, cov_tauc = np.polyfit(
                x[section_index[tauc]:section_index[tauc + 1]],
                y[section_index[tauc]:section_index[tauc + 1]], 1, cov=True)
            fit_base, cov_base = np.polyfit(
                x[section_index[base]:section_index[base + 1]],
                y[section_index[base]:section_index[base + 1]], 1, cov=True)
            # Calculate the intersection of the two lines.
            x_intersect = (fit_base[1] - fit_tauc[1]) / (fit_tauc[0] - fit_base[0])
            y_intersect = fit_tauc[0] * x_intersect + fit_tauc[1]
            # Get the standard error of slope and intercept of tauc fit and base fit.
            sigma_slope_tauc = np.sqrt(cov_tauc[0, 0])
            sigma_intercept_tauc = np.sqrt(cov_tauc[1, 1])
            sigma_slope_base = np.sqrt(cov_base[0, 0])
            sigma_intercept_base = np.sqrt(cov_base[1, 1])
            # Calculate the difference between min and max intersection.
            x_intersect_max = (
                ((fit_base[1] + sigma_intercept_base) - (fit_tauc[1] - sigma_intercept_tauc))
                / ((fit_tauc[0] - sigma_slope_tauc) - (fit_base[0] + sigma_slope_base)))
            x_intersect_min = (
                ((fit_base[1] - sigma_intercept_base) - (fit_tauc[1] + sigma_intercept_tauc))
                / ((fit_tauc[0] + sigma_slope_tauc) - (fit_base[0] - sigma_slope_base)))
            sigma_x_intersect = x_intersect_max - x_intersect_min

            # Judge if the intersection can be selected
            if (not (x_intersect_min < x_intersect < x_intersect_max)
                    or x_intersect > sigma_x_intersect):
                continue
            if y_intersect > (y[section_index[tauc]]
                              + 0.2 * (y[section_index[tauc + 1]]
                                       - y[section_index[tauc]])):
                continue

            return x_intersect, sigma_x_intersect

        raise ValueError('No valid intersection found.')


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
