# -*- coding: utf-8 -*-
"""
This module contains the class for a single XRD pattern.
- Author: Junyuan Lu
- E-mail: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import make_smoothing_spline
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from pybaselines import Baseline
from ..typing import PatternInfo, AngleRange, CIFData, PeakParam
from .cifdatabase import CIFDatabase


class XRDPattern:
    """A class to process a single diffraction pattern.

    For a high-throughput XRD analysis, patterns are the smallest unit.

    Attributes:
        two_theta (NDArray[np.float64]): The 2 theta angle of diffraction pattern.
        intensity (NDArray[np.float64]): The intensity of diffraction pattern.
        info (PatternInfo): The information of the pattern. Please refer to the
            PatternInfo class for more information.
    """
    def __init__(
            self,
            info: PatternInfo,
            file_path: str | None = None,
            two_theta: NDArray[np.float64 | np.int_] | list[float | int] | None = None,
            intensity: NDArray[np.float64 | np.int_] | list[float | int] | None = None):
        """Create a instance of XRDPattern.

        There are two ways to create a XRDPattern instance
        - Provide the file_path of the .xy file.
        - Provide the two_theta and intensity of the pattern.

        Args:
            info (PatternInfo): The information of the pattern.
            file_path (str, optional): The path of the .xy file. Defaults to None.
            two_theta (NDArray[np.float64 | np.int_] | list[float | int] | None, optional):
                The 2 theta angle of the pattern. Defaults to None.
            intensity (NDArray[np.float64 | np.int_] | list[float | int] | None, optional):
                The intensity of the pattern. Defaults to None.
        """
        if file_path is None and any([two_theta is None, intensity is None]):
            raise ValueError('Provide either file_path or two_theta and intensity!')
        if file_path:
            two_theta, intensity = self._read_xy(file_path)
        else:
            two_theta = np.array(two_theta).astype(np.float64)
            intensity = np.array(intensity).astype(np.float64)
        if len(two_theta) != len(intensity):
            raise ValueError('The length of two_theta and intensity should be the same!')
        if not np.all(np.diff(two_theta) > 0):
            raise ValueError('The two_theta should be in ascending order!')
        # Cut the two_theta and intensity based on the scan angle
        left_index = np.argwhere(two_theta >= info.two_theta_range.left)[0, 0]
        right_index = np.argwhere(two_theta < info.two_theta_range.right)[-1, 0]
        self.two_theta = two_theta[left_index: right_index].astype(np.float64)
        self.intensity = intensity[left_index: right_index].astype(np.float64)
        self.info = info

    @staticmethod
    def _read_xy(
            file_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Import two_theta and intensity from .xy file."""
        # The first line is the header, so skip it
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines.pop(0)
        # Split each line of data into a list: [two-theta, intensity]
        data = np.array([line.split() for line in lines], dtype=np.float64)
        return data[:, 0], data[:, 1]

    def copy(self) -> XRDPattern:
        """Create a deep copy instance of XrdPattern.

        Returns:
            XrdPattern: A deep copy of the class instance.
        """
        return XRDPattern(two_theta=self.two_theta, intensity=self.intensity, info=self.info)

    def get_baseline(
            self,
            baseline_lam: int = 200) -> NDArray[np.float64] | None:
        """Get the baseline of xrd data.

        The parameter lam should be positive integer. The smaller the lam, the more the baseline
        will be close to the original data. If lam < 0, return None.

        Args:
            baseline_lam (int, optional): Parameters that control the fitting of baseline,
                the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html
                for more infomation.
                Defaults to 200.

        Returns:
            NDArray[np.float64] | None: The baseline of the xrd data, which has the same shape
                as the intensity. If lam < 0, return None.
        """
        if baseline_lam <= 0:
            return None
        baseline_fitter = Baseline(x_data=self.two_theta)
        baseline, _ = baseline_fitter.pspline_psalsa(self.intensity, lam=baseline_lam)
        return baseline.astype(np.float64)

    def subtract_baseline(
            self,
            baseline_lam: int = 200) -> XRDPattern:
        """Subtract the baseline of xrd data.

        The parameter lam should be positive integer. The smaller the lam, the more the baseline
        will be close to the original data. If lam < 0, return the original XRDPattern without
        any change.

        Args:
            baseline_lam (int, optional): Parameters that control the fitting of baseline,
                the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html
                for more infomation.
                Defaults to 200.

        Returns:
            XrdPattern: A new instance of XrdPattern with baseline subtracted.
        """
        if baseline_lam <= 0:
            return self.copy()
        baseline = self.get_baseline(baseline_lam=baseline_lam)
        assert isinstance(baseline, np.ndarray)
        baseline = baseline[:-1]
        new_pattern = self.copy()
        new_pattern.intensity = new_pattern.intensity - baseline
        return new_pattern

    def smooth(
            self,
            window: int = 51,
            spline_lam: float | None = None) -> XRDPattern:
        """Smooth xrd data by Savitzky-Golay filter and make_splrep.

        Any illigal parameter will not be used. For example, if window < 3 and factor < 0,
        the original data will be returned. If window < 3, only make_splrep will be used.
        Also, any even window will be converted to odd by adding 1.

        Args:
            window (int, optional): The parameter of Savitzky-Golay filter, which should be
                an odd number larger than 3. The larger the window, the smoother the data.
                If window < 3, do not use Savitzky-Golay filter.
                Please check
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html.
                Defaults to 51.
            spline_lam (float | None, optional): The parameter lam of make_smooth_spline. The larger
                the lam, the smoother the data. If lam < 0, do not use make_smooth_spline.
                If None, use GCV to determine the lam. Please check
                https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.make_smoothing_spline.html.
                Defaults to None.

        Returns:
            XrdPattern: A new instance of XrdPattern with smoothed intensity.
        """
        if window < 3 and spline_lam and spline_lam < 0:
            return self.copy()
        new_pattern = self.copy()
        if window % 2 == 0:
            window += 1
        if window > 3:
            new_pattern.intensity = savgol_filter(
                x=new_pattern.intensity,
                window_length=window,
                polyorder=3)
        if spline_lam is None or spline_lam >= 0:
            interpolate = make_smoothing_spline(
                x=new_pattern.two_theta,
                y=new_pattern.intensity,
                lam=spline_lam)
            new_pattern.intensity = interpolate(new_pattern.two_theta)
        return new_pattern

    def postprocess(
            self,
            baseline_lam: int = 200,
            window: int = 51,
            spline_lam: float | None = None) -> XRDPattern:
        """Default postprocess for xrd pattern.

        The ``self.postprocess()`` is equivalent to ``self.subtract_baseline().smooth()``.
        Please refer to the subtract_baseline and smooth method for more information.

        Args:
            baseline_lam (int, optional): lam parameter for subtract_baseline. Defaults to 200.
            window (int, optional): window_length for smooth. Defaults to 51.
            spline_lam (float | None, optional): spline_lam for smooth. Defaults to None.

        Returns:
            XRDPattern: A new instance of XrdPattern with baseline subtracted and smoothed.
        """
        return self.subtract_baseline(
            baseline_lam=baseline_lam).smooth(
                window=window, spline_lam=spline_lam)

    def get_peak(
            self,
            mask: AngleRange | list[AngleRange] | None = None,
            param: PeakParam | None = None,
            mask_param: PeakParam | None = None,
            max_intensity: float | None = None
    ) -> tuple[NDArray[np.int_], dict[str, NDArray[np.float64]]]:
        """Find peaks by scipy.signal.find_peaks.

        Please substract and smooth the data before peak detection to get better results.
        Please check
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        for more information.

        Args:
            mask (AngleRange | list[AngleRange] | None, optional): Mask information in the form of
                [(mask1_left, mask1_right), (mask2_left, mask2_right) ...]. Defaults to None.
            param (PeakParam, optional): The parameters for finding peaks. Defaults to None.
            mask_param (PeakParam, optional): The parameters for finding peaks in the masked area.
                Defaults to None.
            max_intensity (float, optional): The maximum intensity of the peaks, the parameter
                is for the whole database. If None, use the maximum intensity of the pattern.
                Defaults to None.

        Returns:
            tuple: The index and properties of the peaks.
                - NDArray[int]: The index of the peaks.
                - dict[str, NDArray[float]]: The properties of the peaks. Keys include
                    ``peak_heights``, ``prominences``, ``left_bases``, ``right_bases``,
                    and ``peak_angles``.
        """
        if param is None:
            param = PeakParam()
        if mask_param is None:
            mask_param = PeakParam()
        if max_intensity is None:
            max_intensity = self.intensity.max()
        if isinstance(mask, AngleRange):
            mask = [mask]

        # Set the function to create mask bool array
        def _create_mask(mask: list[AngleRange]) -> NDArray:
            """Create array of bool for peak detection."""
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
        # Set the masked area to nan
        unmasked_intensity = self.intensity.copy()
        unmasked_intensity[np.where(mask_condition)] = np.nan
        masked_intensity = self.intensity.copy()
        masked_intensity[np.where(~mask_condition)] = np.nan
        # Find peaks
        assert isinstance(param, PeakParam)
        assert isinstance(mask_param, PeakParam)
        assert isinstance(max_intensity, float)
        angle_step = np.mean(np.diff(self.two_theta))
        peak_index, properties = find_peaks(
            unmasked_intensity,
            height=param.height * max_intensity,
            distance=int(param.distance / angle_step),
            prominence=param.prominence * max_intensity)
        additional_index, a_properties = find_peaks(
            masked_intensity,
            height=mask_param.height * max_intensity,
            distance=int(mask_param.distance / angle_step),
            prominence=mask_param.prominence * max_intensity)
        if additional_index.size > 0:
            peak_index = np.concatenate((peak_index, additional_index))
            for key, value in properties.items():
                properties[key] = np.concatenate((value, a_properties[key]))
        properties['peak_angles'] = self.two_theta[peak_index]
        return peak_index, properties

    def match(
            self,
            cif_database: CIFDatabase,
            number: int = 5,
            two_theta_tol: float = 0.1,
            **kwargs) -> list[CIFData]:
        """Match the peaks of the pattern with the reference phases in CIFdatabase.

        Since the peak number of the pattern and those of the reference phases are usually
        different, the Jaccard similarity is used to calculate the similarity of the two sets.
        Peaks with in the two theta tolerance are considered equal.

        Args:
            cif_database (CIFDatabase): The database of CIF data.
            number (int, optional): The number of output matched patterns. If number is greater
                than the total reference phases according to conditions in kwargs, the number
                will be automatically set to that number. Defaults to 5.
            two_theta_tol (float, optional): The tolerance of two theta for peak matching.
                Defaults to 0.1.
            **kwargs: The keyword arguments for get_peak. Please refer to the get_peak method
                for more information.
                - mask (AngleRange | list[AngleRange] | None)
                - param (PeakParam | None)
                - mask_param (PeakParam | None)
                - max_intensity (float | None)

        Returns:
            list[CIFData]: The matched information from high score to low score.
        """
        # Get the peaks
        _, properties = self.get_peak(**kwargs)
        peak_angles = properties['peak_angles']

        def jaccard_similarity(list1, list2, threshold=0.1):
            """Calculate the Jaccard similarity of two lists."""
            for i, val1 in enumerate(list1):
                for val2 in list2:
                    if abs(val1 - val2) < threshold:
                        list1[i] = val2
                        break
            intersection = len(set(list1) & set(list2))
            union = (len(list1) + len(list2)) - intersection
            return intersection / union

        # Match the peaks.
        score = []
        index_to_search = cif_database.index(element=self.info.element)
        for i in index_to_search:
            data_angle = cif_database.data[i].two_theta
            score.append(jaccard_similarity(peak_angles, data_angle, two_theta_tol))

        # Check if number is greater than the total cif datas in CIFDatabase
        number = min(number, len(index_to_search))

        # Sort the mse list and return the top matches
        score = np.array(score)
        index = np.argsort(score)
        matched_data = []
        for i in range(number):
            matched_data.append(cif_database.data[index[i]])
        return matched_data

    def plot(
            self,
            ax: Axes | None = None,
            offset: float = 0,
            if_label: bool = True,
            **kwargs) -> None:
        """Plot the diffraction pattern.

        Args:
            ax (Optional[Axes], optional): The matplotlib axes. Defaults to None.
            offset (float): The offset of the intensity. Defaults to 0.
            if_label (bool, optional): If show the x and y label. Defaults to True.
            **kwargs: The keyword arguments for the plot and get_peak.  If any of the
                    mask, param, mask_param, and max_intensity is provided, the peaks
                    will be plotted with 'x' marker.
                - mask (AngleRange | list[AngleRange] | None)
                - param (PeakParam | None)
                - mask_param (PeakParam | None)
                - max_intensity (float | None)
                - scatter_color (str): The color of the scatter. Defaults to 'tab:blue'.
                - line_color (str): The color of the curve. Defaults to 'tab:orange'.
                - color (str): If set, the color of scatter and line will be set to this value.
                - label (str): The label of the curve that will be displayed in legend.
                    Defaults to f'{self.info.name}-{self.info.index}'.
                - Other keyword arguments for the plot method.
                    Please refer to matplotlib.pyplot.plot.
        """
        if 'color' in kwargs:
            kwargs['scatter_color'] = kwargs['color']
            kwargs['line_color'] = kwargs['color']
            kwargs.pop('color')
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        if any(key in kwargs for key in ['mask', 'param', 'mask_param', 'max_intensity']):
            peak_index, _ = self.get_peak(
                mask=kwargs.pop('mask', None),
                param=kwargs.pop('param', None),
                mask_param=kwargs.pop('mask_param', None),
                max_intensity=kwargs.pop('max_intensity', None))
            kw = kwargs.copy()
            kw.pop('label', None)  # Scatter should not have legend
            kw.pop('alpha', None)  # Scatters' alpha should be 1
            kw.pop('line_color', None)
            ax.plot(
                self.two_theta[peak_index], self.intensity[peak_index] + offset,
                'x', color=kw.pop('scatter_color', 'tab:orange'), **kw)
        kwargs.pop('scatter_color', None)
        ax.plot(
            np.array(self.two_theta), np.array(self.intensity) + offset,
            label=kwargs.pop('label', f'{self.info.name}-{self.info.index}'),
            color=kwargs.pop('line_color', 'tab:blue'), **kwargs)
        if if_label:
            ax.set_xlabel(r'$2\theta$')
            ax.set_ylabel('Intensity')

    def plot_with_ref(
            self,
            cif_database: CIFDatabase,
            cmap: str | Colormap = '',
            title: str | None = None,
            ylim: tuple[float | int, float | int] | None = None,
            **kwargs) -> None:
        """Plot the diffraction pattern with the matched reference phase XRD data.

        Since this figure is a subplot, it will create a new figure and show immediately,
        which is different from the plot method. Also, the title of the plot can only be
        set by passing the title parameter.

        Args:
            cif_database (CIFDatabase): The database of reference phases.
            cmap (str | Colormap, optional): The colormap for the matched reference
                phases. Defaults to ''. If number <= 10, default to 'tab10'. If number <= 20,
                default to 'tab20'. If number > 20, default to 'viridis'.
            title (str, optional): The title of the plot. Defaults to None.
            ylim (tuple[float | int, float | int] | None, optional): The y-axis limit of the
                pattern. Defaults to None. If None, the y-axis limit will be set automatically.
            **kwargs: The keyword arguments for match and the plot.
                - two_theta_tol (float): The tolerance of two theta for peak matching.
                    Defaults to 0.1.
                - number (int): The number of output matched patterns. Defaults to 5.
                - Other keyword arguments for the plot method.
                    Please refer to ``XRDPattern.plot()``.
        """
        # Get the matched data
        number = kwargs.pop('number', 5)
        matched_data = self.match(
            cif_database, number=number,
            two_theta_tol=kwargs.pop('two_theta_tol', 0.1),
            mask=kwargs.get('mask', None),
            param=kwargs.get('param', None),
            mask_param=kwargs.get('mask_param', None),
            max_intensity=kwargs.get('max_intensity', None))
        # Plot the pattern in vertical subplots
        fig, axs = plt.subplots(number + 1, 1, figsize=(6, 2 * number), sharex=True)
        fig.subplots_adjust(hspace=0)
        self.plot(ax=axs[0], **kwargs)  # type: ignore
        if ylim:
            axs[0].set_ylim(*ylim)  # type: ignore
        # Set the colormap
        if cmap == '':
            if number <= 10:
                cmap = 'tab10'
            elif number <= 20:
                cmap = 'tab20'
            else:
                cmap = 'viridis'
        assert cmap
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        # Plot the matched data
        for i, data in enumerate(matched_data):
            cif_database.plot(
                cif_name=data.name, ax=axs[i + 1],  # type: ignore
                if_show=False, color=cmap(i),
                two_theta_range=AngleRange(
                    left=self.two_theta[0],
                    right=self.two_theta[-1]))
        # Set the title
        if title:
            axs[0].set_title(title)  # type: ignore
        plt.show()

    def save_txt(
            self,
            save_dir: str = '',
            save_path: str = '') -> None:
        """Save the diffraction pattern to a txt file.

        The txt file will be saved in the format of two columns, the first column is the
        two_theta and the second column is the intensity, separated by comma.

        Args:
            save_dir (str, optional): Directory for saving. Defaults to ''.
            save_path (str, optional): Path for saving. Defaults to ''.

        Raises:
            ValueError: If both save_dir and save_path are provided or not provided.
        """
        if all([not save_dir, not save_path]) or all([save_dir, save_path]):
            raise ValueError('Please provide either save_dir or save_path!')
        file_name = ''
        if save_path:
            file_name = save_path
        if save_dir:
            file_name = os.path.join(save_dir, f'{self.info.name}-{self.info.index}.txt')
        assert file_name != ''
        # Use comma to separate the two columns
        np.savetxt(file_name, np.array([self.two_theta, self.intensity]).T, delimiter=',')
