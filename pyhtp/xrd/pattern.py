# -*- coding: utf-8 -*-
"""
A class that contains the data of a single diffraction pattern.
"""
from __future__ import annotations
from typing import Optional, Union
import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy.spatial import distance
from pybaselines import Baseline
from ..typing import PatternInfo, AngleRange, IcsdData
from .icsd import ICSD


class XrdPattern:
    """A class that contains the data of a single diffraction pattern.
    """
    def __init__(self,
                 two_theta: NDArray,
                 intensity: NDArray,
                 info: PatternInfo = PatternInfo()):
        """Create a instance of XrdPattern.

        Args:
            two_theta (NDArray): The two_theta angle of diffraction pattern.
            intensity (NDArray): The intensity of diffraction pattern.
            description (Optional[str], optional): Description of the pattern. Defaults to None.
        """
        if len(two_theta) != len(intensity):
            raise ValueError('The length of two_theta and intensity should be the same!')
        self.two_theta = two_theta
        self.intensity = intensity
        self.info = info
        # Cut the two_theta and intensity based on the scan angle
        left_index = np.argwhere(self.two_theta >= self.info.angle_range[0])[0, 0]
        right_index = np.argwhere(self.two_theta <= self.info.angle_range[1])[-1, 0]
        self.two_theta = self.two_theta[left_index:right_index]
        self.intensity = self.intensity[left_index:right_index]

    def copy(self) -> XrdPattern:
        """Create a deep copy instance of XrdPattern.

        Returns:
            XrdPattern: A deep copy of the class instance.
        """
        return XrdPattern(two_theta=self.two_theta, intensity=self.intensity, info=self.info)

    def get_baseline(self, lam: int = 200) -> NDArray:
        """Get the baseline of xrd data.

        Args:
            lam (int, optional): Parameters that control the fitting of baseline, the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html for more infomation.
                Defaults to 200.

        Returns:
            NDArray: The baseline intensity of the xrd data.
        """
        baseline_fitter = Baseline(x_data=self.two_theta)
        baseline, _ = baseline_fitter.pspline_psalsa(self.intensity, lam=lam)
        return baseline

    def subtract_baseline(self, lam: int = 200) -> XrdPattern:
        """Subtract the baseline of xrd data.

        Args:
            lam (int, optional): Parameters that control the fitting of baseline, the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html for more infomation.
                Defaults to 200.

        Returns:
            XrdPattern: A new instance of XrdPattern with baseline subtracted.
        """
        baseline = self.get_baseline(lam=lam)
        baseline = baseline[:-1]
        corrected_pattern = self.copy()
        corrected_pattern.intensity = corrected_pattern.intensity - baseline
        return corrected_pattern

    def smooth(self, window: int = 101, factor: float = 0.5) -> XrdPattern:
        """Smooth xrd data by Savitzky-Golay filter and UnivariateSpline.

        Args:
            window (int, optional): The parameter of Savitzky-Golay filter.
                Please check https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html.
                Defaults to 101.
            factor (float, optional): The parameter of UnivariateSpline.
                Please check https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html.
                Defaults to 0.5.

        Returns:
            XrdPattern: A new instance of XrdPattern with smoothed intensity.
        """
        smoothed_pattern = self.copy()
        smoothed_pattern.intensity = scipy.signal.savgol_filter(smoothed_pattern.intensity, window, 3)
        interpolate_smooth = scipy.interpolate.UnivariateSpline(smoothed_pattern.two_theta, smoothed_pattern.intensity)
        interpolate_smooth.set_smoothing_factor(factor)
        smoothed_pattern.intensity = interpolate_smooth(smoothed_pattern.two_theta)
        return smoothed_pattern

    def get_peak(self,
                 mask: Optional[list[AngleRange]] = None,
                 height: float = -1,
                 mask_height: float = -1) -> tuple[NDArray, dict]:
        """Find peaks by scipy.signal.find_peaks. Please substract and smooth the data before peak detection.

        Args:
            mask (Optional[list[AngleRange]], optional): The mask area to not be detected. Defaults to None.
            threshold (float, optional): Height threshold. Defaults to -1.
            mask_threshold (float, optional): Height threshold at masked area. Defaults to -1.

        Returns:
            tuple[NDArray, dict]: (the index of peaks in the array, properties of the peaks)
        """
        def _create_mask(mask: list[AngleRange]) -> NDArray:
            """Create a mask for peak detection.

            Args:
                mask (list[AngleRange]): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...].

            Returns:
                NDArray: The mask for peak detection. An array of bool, where True means the data is masked.
            """
            mask_condition = np.full(self.intensity.shape[0], False)
            for info in mask:
                left_index = np.argwhere(self.two_theta >= info.left)[0, 0]
                right_index = np.argwhere(self.two_theta <= info.right)[-1, 0]
                mask_condition[left_index:right_index] = True
            return mask_condition
        # Set the mask bool array
        if mask:
            mask_condition = _create_mask(mask)
        else:
            mask_condition = np.full(self.intensity.shape[0], False)
        if height == -1:  # Use the 1% of the maximum intensity as the threshold
            height = 0.01 * np.max(self.intensity)
        if mask_height == -1:  # Use the 10% of the maximum intensity as the threshold
            mask_height = 0.1 * np.max(self.intensity)
        # Set the masked area to nan
        unmasked_intensity = self.intensity.copy()
        unmasked_intensity[np.where(mask_condition)] = np.nan
        masked_intensity = self.intensity.copy()
        masked_intensity[np.where(~mask_condition)] = np.nan
        # Find peaks
        angle_step = self.two_theta[1] - self.two_theta[0]
        peak_index, properties = scipy.signal.find_peaks(
            unmasked_intensity, height=height, distance=int(0.5 / angle_step))
        additional_index, a_properties = scipy.signal.find_peaks(
            masked_intensity, height=mask_height, distance=int(0.5 / angle_step))
        if additional_index.size > 0:
            peak_index = np.concatenate((peak_index, additional_index))
            for key, value in properties.items():
                properties[key] = np.concatenate((value, a_properties[key]))
        properties['peak_angles'] = self.two_theta[peak_index]
        return peak_index, properties

    def match(self,
              icsd: ICSD,
              number: int = 5,
              **kwargs) -> list[IcsdData]:
        """Match the peaks of the pattern with the ICSD database.

        Args:
            icsd (ICSD): The ICSD database.
            number (int, optional): The number of matched patterns. Defaults to 5.
            **kwargs: The arguments for get_peak.

        Returns:
            list[IcsdData]: The matched information from high to low.
        """
        # Get the peaks
        _, properties = self.get_peak(**kwargs)
        peak_angles = properties['peak_angles']

        def calculate_mse(angles1, angles2) -> float:
            """Calculate the mean squared error between two sets of angles using nearest neighbor matching."""
            angles1 = np.array(angles1)
            angles2 = np.array(angles2)
            distances = distance.cdist(angles1.reshape(-1, 1), angles2.reshape(-1, 1), 'sqeuclidean')
            min_distances = np.min(distances, axis=1)
            mse = np.mean(min_distances)
            return mse

        # Match the peaks. For each icsd file, calculate the mse of the peak angles
        mse_list = []
        index_to_search = icsd.index(element=self.info.element)
        for i in index_to_search:
            data_angle = icsd.data[i].two_theta
            mse = calculate_mse(peak_angles, data_angle)
            mse_list.append(mse)

        # Sort the mse list and return the top matches
        mse_list = np.array(mse_list)
        index = np.argsort(mse_list)
        matched_data = []
        for i in range(number):
            matched_data.append(icsd.data[index[i]])
        return matched_data

    def plot(
            self,
            ax: Optional[Axes] = None,
            offset: float = 0,
            if_label: bool = True,
            **kwargs) -> None:
        """Plot the diffraction pattern.

        Args:
            ax (Optional[Axes], optional): The matplotlib axes. Defaults to None.
            offset (float): The offset of the intensity.
            if_label (bool, optional): If show the label. Defaults to True.
            **kwargs: The keyword arguments for the plot.

        Raises:
            ValueError: The ax is not correctly set.
        """
        if ax is None:
            _, ax = plt.subplots()
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        ax.plot(np.array(self.two_theta),
                np.array(self.intensity) + offset,
                label=f'{self.info.name}-{self.info.index}', **kwargs)
        if if_label:
            ax.set_xlabel(r'$2\theta$')
            ax.set_ylabel('Intensity')

    def plot_baseline(self,
                      lam: int = 200,
                      ax: Optional[Axes] = None,
                      **kwargs) -> None:
        """Plot the baseline of the diffraction pattern.

        Args:
            lam (int, optional): Parameters that control the fitting of baseline, the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html for more infomation.
                Defaults to 200.
            ax (Optional[Axes], optional): The matplotlib axes. Defaults to None.
            **kwargs: The keyword arguments for the plot.

        Raises:
            ValueError: The ax is not correctly set.
        """
        if ax is None:
            _, ax = plt.subplots()
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        baseline = self.get_baseline(lam)
        ax.plot(self.two_theta, baseline, alpha=0.7, **kwargs)
        ax.set_xlabel(r'$2\theta$')
        ax.set_ylabel('Intensity')

    def plot_with_icsd(self,
                       icsd: ICSD,
                       number: int = 5,
                       cmap: Union[str, Colormap] = 'tab10',
                       **kwargs) -> None:
        """Plot the diffraction pattern with the matched ICSD data.

        Args:
            ax (Optional[Axes], optional): The matplotlib axes. Defaults to None.
            number (int, optional): The number of matched patterns. Defaults to 5.
            **kwargs: The keyword arguments for the plot.

        Returns:
            Axes: The matplotlib axes.
        """
        # Get the matched data
        matched_data = self.match(icsd, number=number)
        # Plot the pattern in vertical subplots
        fig, axs = plt.subplots(number + 1, 1, figsize=(6, 2 * number))
        fig.subplots_adjust(hspace=0)
        print(self.intensity)
        print(self.two_theta)
        self.plot(ax=axs[0], **kwargs)
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        # Plot the matched data
        for i, data in enumerate(matched_data):
            icsd.plot(
                file_name=data.name, ax=axs[i + 1],
                if_show=False, color=cmap(i),
                angle_range=AngleRange(left=self.two_theta[0],
                                       right=self.two_theta[-1]))
        plt.show()
