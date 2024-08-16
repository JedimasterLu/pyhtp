# -*- coding: utf-8 -*-
"""
Filename: processing.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import pickle
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pybaselines import Baseline
from pymatgen.analysis.diffraction.xrd import XRDCalculator


class XrdProcess:
    ''' A class that can process xrd data by baseline substraction, smoothing, and peak analysis.

    Attributes:
        intensity: np.ndarray
            The intensity of the xrd data.
        two_theta: np.ndarray
            The two_theta of the xrd data.
        pattern_path: str
            The path of the pattern database, which is a pickle file.
        structure_path: str
            The path of the structure database, which is a pickle file.

    Methods:
        copy() -> XrdData:
            Copy a instance.
        set_data(intensity: np.ndarray=None, two_theta: np.ndarray=None):
            Set intensity and two_theta.
        set_database_path(pattern_path: str=None, structure_path: str=None):
            Set the database path.
        substract_baseline(lam: int=200) -> XrdData:
            Substract baseline from xrd data.
        get_baseline(lam: int=200) -> np.ndarray:
            Get baseline from xrd data.
        smooth(window: int=101, factor: float=0.5) -> XrdData:
            Smooth xrd data by Savitzky-Golay filter and UnivariateSpline.
        peaks(mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12) -> tuple[np.ndarray, np.ndarray, dict]:
            Find peaks by scipy.signal.find_peaks.
        similar_peak_number(measured_peaks: np.ndarray, reference_peaks: np.ndarray, tolerance: float) -> int:
            Calculate the number of similar peaks between measured and reference peaks.
        avg_min_lse(measured_peaks: np.ndarray, reference_peaks: np.ndarray) -> float:
            Calculate the average minimum least square error between measured and reference peaks.
        match(tolerance: float=0.3, mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12) -> str | list[str]:
            Match the xrd data with reference data.
        identify(mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12,
            tolerance: float=0.3, display_number: int=5, figure_title: str='', smooth_window: int=101,
            smooth_factor: float=0.5, save_path: str='', if_show: bool=True, lam: int=200):
            Identify the structure of the xrd data.
    '''

    def __init__(self, file_path: list[str]=None, two_theta: np.ndarray=None, intensity: np.ndarray=None, pattern_path: str=None, structure_path: str=None):
        '''__init__ create a instance of XrdData.

        Args:
            file_path (list[str], optional): The file path of .xy files, which should only contain 2 str elements. Defaults to None.
            two_theta (np.ndarray, optional): The two_theta angle of xrd spectrum. Defaults to None.
            intensity (np.ndarray, optional): The intensity of xrd spectrum. Defaults to None.
            pattern_path (str, optional): The file path of pattern.pkl. Defaults to None.
            structure_path (str, optional): The file path of structure.pkl. Defaults to None.

        Raises:
            ValueError: File path and two_theta/intensity cannot be set at the same time!
            ValueError: Only two .xy files are allowed.
            ValueError: Intensity and two_theta should have the same length!
        '''
        self.intensity = None
        self.two_theta = None
        if file_path and (two_theta is not None or intensity is not None):
            raise ValueError('File path and two_theta/intensity cannot be set at the same time!')
        if file_path and len(file_path) != 2:
            raise ValueError('Only two .xy files are allowed.')
        if intensity is not None and two_theta is not None and len(intensity) != len(two_theta):
            raise ValueError('Intensity and two_theta should have the same length!')
        if file_path:
            double_xrd_data = []
            for path in file_path:
                current_data = self.read_data(path)
                double_xrd_data.append(current_data)
            self.two_theta, self.intensity = self.combine_data(double_xrd_data[0], double_xrd_data[1])
        if two_theta is not None:
            self.two_theta = two_theta
        if intensity is not None:
            self.intensity = intensity
        self.pattern_path = pattern_path
        self.structure_path = structure_path

    def copy(self) -> XrdProcess:
        '''copy generates a new instance of XrdData.

        Returns:
            XrdData: A deep copy of the class instance.
        '''
        return XrdProcess(two_theta=self.two_theta, intensity=self.intensity, pattern_path=self.pattern_path, structure_path=self.structure_path)

    def set_data(self, intensity: np.ndarray=None, two_theta: np.ndarray=None, file_path: list[str]=None):
        '''set_data set intensity and two_theta by directly given or set file path.

        Args:
            intensity (np.ndarray, optional): The intensity of xrd spectrum.. Defaults to None.
            two_theta (np.ndarray, optional): The two_theta angle of xrd spectrum. Defaults to None.
            file_path (list[str], optional): The file path of .xy files, which should only contain 2 str elements. Defaults to None.
                If file_path is set, intensity and two_theta will be set by reading the file.

        Raises:
            ValueError: Intensity and two_theta should have the same length!
            ValueError: File path and two_theta/intensity cannot be set at the same time!
            ValueError: Only two .xy files are allowed.
        '''
        if file_path and (two_theta is not None or intensity is not None):
            raise ValueError('File path and two_theta/intensity cannot be set at the same time!')
        if file_path and len(file_path) != 2:
            raise ValueError('Only two .xy files are allowed.')
        if intensity is not None and two_theta is not None and len(intensity) != len(two_theta):
            raise ValueError('Intensity and two_theta should have the same length!')
        if intensity is not None and two_theta is not None:
            self.intensity = intensity
        if two_theta is not None:
            self.two_theta = two_theta
        if file_path:
            double_xrd_data = []
            for path in file_path:
                current_data = self.read_data(path)
                double_xrd_data.append(current_data)
            self.two_theta, self.intensity = self.combine_data(double_xrd_data[0], double_xrd_data[1])

    def set_database_path(self, pattern_path: str=None, structure_path: str=None):
        '''set_database_path set the path of the pattern and structure database.

        Args:
            pattern_path (str, optional): The path of pattern.pkl. Defaults to None.
            structure_path (str, optional): The path of structure.pkl. Defaults to None.
        '''
        self.pattern_path = pattern_path
        self.structure_path = structure_path

    def read_data(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        '''read_data read xrd data from a single .xy files.

        Args:
            file_path (str): The file path of .xy files to be read.

        Returns:
            tuple[np.ndarray, np.ndarray]: The (two_theta, intensity) of the xrd data.
        '''
        # Read the file into a List[str] and pop the first line
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            data.pop(0)
        # Split each line of data into a list: [2-theta, intensity]
        for index, line in enumerate(data):
            data[index] = line.split()
            for i, value in enumerate(data[index]):
                data[index][i] = float(value)
        data = np.array(data)
        return data[:, 0], data[:, 1]

    def combine_data(self, left_data: tuple[np.ndarray, np.ndarray], right_data: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        '''combine_data combine two xrd data (left and right) into one.

        Args:
            left_data (tuple[np.ndarray, np.ndarray]): The (two_theta, intensity) of the left xrd data.
            right_data (tuple[np.ndarray, np.ndarray]): The (two_theta, intensity) of the right xrd data.

        Returns:
            tuple[np.ndarray, np.ndarray]: The (two_theta, intensity) of the combined xrd data.
        '''
        # Find the element which two_theta is between 10 and 34 and remove all elements out of this range.
        left_two_theta = left_data[0]
        left_intensity = left_data[1]
        left_data = pd.DataFrame({'two_theta': left_two_theta, 'intensity': left_intensity})
        left_data = left_data[(left_data['two_theta'] > 10) & (left_data['two_theta'] < 34)]
        # Find the element which two_theta is between 34 and 58 and remove all elements out of this range.
        right_two_theta = right_data[0]
        right_intensity = right_data[1]
        right_data = pd.DataFrame({'two_theta': right_two_theta, 'intensity': right_intensity})
        right_data = right_data[(right_data['two_theta'] > 34) & (right_data['two_theta'] < 58)]
        # Merge the two dataframes
        combined_data = pd.concat([left_data, right_data])
        return combined_data['two_theta'].values, combined_data['intensity'].values

    def substract_baseline(self, lam: int=200) -> XrdProcess:
        '''substract_baseline substract the baseline of xrd data.

        Args:
            lam (int, optional): Parameters that control the fitting of baseline, the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html for more infomation.
                Defaults to 200.

        Raises:
            ValueError: Intensity and two_theta cannot be None!

        Returns:
            XrdData: A new instance of XrdData with baseline substracted.
        '''
        substracted_xrd = self.copy()
        if substracted_xrd.intensity is None or substracted_xrd.two_theta is None:
            raise ValueError('Intensity and two_theta cannot be None!')
        # Get the baseline
        baseline_fitter = Baseline(x_data=substracted_xrd.two_theta)
        baseline, _ = baseline_fitter.pspline_psalsa(substracted_xrd.intensity, lam=lam)
        # Substract the baseline from the data
        substracted_xrd.intensity = substracted_xrd.intensity - baseline
        return substracted_xrd

    def get_baseline(self, lam: int=200) -> np.ndarray:
        '''get_baseline return the baseline of xrd data.

        Args:
            lam (int, optional): Parameters that control the fitting of baseline, the larger the finer.
                Please check https://pybaselines.readthedocs.io/en/latest/parameter_selection.html for more infomation.
                Defaults to 200.
        Raises:
            ValueError: Intensity and two_theta cannot be None!

        Returns:
            np.ndarray: The baseline intensity of the xrd data.
        '''
        if self.intensity is None or self.two_theta is None:
            raise ValueError('Intensity and two_theta cannot be None!')
        baseline_fitter = Baseline(x_data=self.two_theta)
        baseline, _ = baseline_fitter.pspline_psalsa(self.intensity, lam=lam)
        return baseline

    def smooth(self, window: int=101, factor: float=0.5) -> XrdProcess:
        '''smooth smooth xrd data by Savitzky-Golay filter and UnivariateSpline.

        Args:
            window (int, optional): The parameter of Savitzky-Golay filter.
                Please check https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html.
                Defaults to 101.
            factor (float, optional): The parameter of UnivariateSpline.
                Please check https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html.
                Defaults to 0.5.

        Raises:
            ValueError: Intensity and two_theta cannot be None!

        Returns:
            XrdData: A new instance of XrdData with smoothed intensity.
        '''
        smoothed_xrd = self.copy()
        if smoothed_xrd.intensity is None or smoothed_xrd.two_theta is None:
            raise ValueError('Intensity and two_theta cannot be None!')
        smoothed_xrd.intensity = scipy.signal.savgol_filter(smoothed_xrd.intensity, window, 3)
        interpolate_smooth = scipy.interpolate.UnivariateSpline(smoothed_xrd.two_theta, smoothed_xrd.intensity)
        interpolate_smooth.set_smoothing_factor(factor)
        smoothed_xrd.intensity = interpolate_smooth(smoothed_xrd.two_theta)
        return smoothed_xrd

    def _create_mask(self, left_angle: float, right_angle: float, current_mask: np.ndarray=None) -> np.ndarray:
        '''_create_mask create a mask for peak detection. If current_mask is not None, combine the two masks.

        Args:
            left_angle (float): The left boundary angle of the mask.
            right_angle (float): The right boundary angle of the mask.
            current_mask (np.ndarray, optional): A previous mask. Defaults to None. If None, create a new mask.

        Returns:
            np.ndarray: The mask for peak detection. An array of bool, where True means the data is masked.
        '''
        left_index = np.argwhere(self.two_theta >= left_angle)[0, 0]
        right_index = np.argwhere(self.two_theta <= right_angle)[-1, 0]
        mask_condition = np.full(self.intensity.shape[0], False)
        mask_condition[left_index:right_index] = True
        if current_mask is not None:
            mask_condition = mask_condition | current_mask
        return mask_condition

    def peaks(self, mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12) -> tuple[np.ndarray, np.ndarray, dict]:
        '''peaks find peaks by scipy.signal.find_peaks. Please substract and smooth the data before peak detection.

        Args:
            mask (list[list[float, float]], optional): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...]. Defaults to None.
            height (float, optional): The height above which peaks could be detected. Defaults to 0.06.
            mask_height (float, optional): The height above which peaks could be detected in masked area. Defaults to 0.12.

        Raises:
            ValueError: Intensity and two_theta cannot be None!
            ValueError: Mask information should be in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...].

        Returns:
            tuple[np.ndarray, np.ndarray, dict]: (peaks angle value, the index of peaks in the array, properties of the peaks)
        '''
        if self.intensity is None or self.two_theta is None:
            raise ValueError('Intensity and two_theta cannot be None!')
        mask_condition = np.full(self.intensity.shape[0], False)
        if mask is not None:
            # pylint: disable=C0123
            if (type(mask[0]) is not list) or (len(mask[0]) != 2):
                raise ValueError('Mask information should be in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...].')
            # pylint: enable=C0123
            for angle in mask:
                mask_condition = self._create_mask(angle[0], angle[1], mask_condition)
        # Set the masked area to nan
        unmasked_intensity = self.intensity.copy()
        unmasked_intensity[np.where(mask_condition)] = np.nan
        masked_intensity = self.intensity.copy()
        masked_intensity[np.where(~mask_condition)] = np.nan
        # Find peaks
        peaks_index, properties = scipy.signal.find_peaks(unmasked_intensity, height=height)
        additional_peaks, a_properties = scipy.signal.find_peaks(masked_intensity, height=mask_height)
        if additional_peaks.size > 0:
            peaks_index = np.concatenate((peaks_index, additional_peaks))
            properties['peak_heights'] = np.concatenate((properties['peak_heights'], a_properties['peak_heights']))
        peaks_value = self.two_theta[peaks_index]
        return peaks_value, peaks_index, properties

    def _similar_peak_number(self, measured_peaks: np.ndarray, reference_peaks: np.ndarray, tolerance: float=0.2) -> int:
        '''_similar_peak_number calculate the number of similar peaks between measured and reference peaks.

        Args:
            measured_peaks (np.ndarray): A array of the angles of measured peaks.
            reference_peaks (np.ndarray): A array of the angles of reference peaks from pattern database.
            tolerance (float, optional): The tolerance of the peak angle, in degree.
                If the difference between two peaks is less than tolerance, they are considered to be similar.
                Defaults to 0.2.

        Returns:
            int: the number of similar peaks between measured and reference peaks.
        '''
        similar_number = 0
        if len(measured_peaks) == 0 or len(reference_peaks) == 0:
            return 0
        for index, ref_peak in enumerate(reference_peaks):
            for msr_peak in measured_peaks:
                # Major peaks can earn more points
                if abs(ref_peak - msr_peak) < tolerance and index <= 1:
                    similar_number += 5
                    break
                if abs(ref_peak - msr_peak) < tolerance and index <= 3:
                    similar_number += 2
                    break
                if abs(ref_peak - msr_peak) < tolerance:
                    similar_number += 1
                    break
        return similar_number

    def _avg_min_lse(self, measured_peaks: np.ndarray, reference_peaks: np.ndarray) -> float:
        '''_avg_min_lse calculate the average minimum least square error between measured and reference peaks.

        Args:
            measured_peaks (np.ndarray): A array of the angles of measured peaks.
            reference_peaks (np.ndarray): A array of the angles of reference peaks from pattern database.

        Returns:
            float: The average minimum least square error between measured and reference peaks.
        '''
        min_lse = 0
        if len(measured_peaks) == 0 or len(reference_peaks) == 0:
            return 0
        for ref_peak in reference_peaks:
            current_minimum_lse = 1e10
            for msr_peak in measured_peaks:
                current_lse = (ref_peak - msr_peak)**2
                if current_lse < current_minimum_lse:
                    current_minimum_lse = current_lse
            min_lse += current_minimum_lse
        min_lse = min_lse / len(reference_peaks)
        return min_lse

    def match(self, tolerance: float=0.3, mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12) -> list[str]:
        '''match return the rating of reference data comparing with the xurrent xrd data.

        Args:
            tolerance (float, optional): The tolerance of the peak angle, in degree.
                If the difference between two peaks is less than tolerance, they are considered to be similar. Defaults to 0.3.
            mask (list[list[float, float]], optional): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...]. Defaults to None.
            height (float, optional): The height above which peaks could be detected. Defaults to 0.06.
            mask_height (float, optional): The height above which peaks could be detected in masked area. Defaults to 0.12.

        Raises:
            ValueError: Pattern path is not set!

        Returns:
            list[str]: A list of the names of the reference data, sorted by rating from high to low.
        '''
        # Load the reference data
        if not self.pattern_path:
            raise ValueError('Pattern path is not set!')
        with open(self.pattern_path, 'rb') as db:
            pattern_database = pickle.load(db)
        # Create a rating for each structure
        peaks_value, _, _ = self.peaks(mask=mask, height=height, mask_height=mask_height)
        rating = {}
        for ref_pattern in pattern_database:
            # Sort the peaks so that the two_theta of larger intensity are in the front
            sort_index = np.argsort(ref_pattern['intensity'])[::-1]
            if self._avg_min_lse(peaks_value, ref_pattern['two_theta'][sort_index]) != 0:
                rating[ref_pattern['name']] = self._similar_peak_number(peaks_value, ref_pattern['two_theta'][sort_index], tolerance) + 1 / self._avg_min_lse(peaks_value, ref_pattern['two_theta'][sort_index]) * 10
            else:
                rating[ref_pattern['name']] = self._similar_peak_number(peaks_value, ref_pattern['two_theta'][sort_index], tolerance)
        # Sort the rating from high to low
        rating = dict(sorted(rating.items(), key=lambda item: item[1], reverse=True))
        return list(rating.keys())

    def identify(self,
                 mask: list[list[float, float]]=None,
                 height: float=0.06,
                 mask_height: float=0.12,
                 tolerance: float=0.3,
                 display_number: int=5,
                 figure_title: str='',
                 window: int=101,
                 factor: float=0.5,
                 save_path: str='',
                 if_show: bool=True,
                 lam: int=200,
                 if_process: bool=True,
                 elements: list[str]=None) -> None | mpl.axes.Axes:
        '''identify the structure of the xrd data by generate a plot to show the comparison between the xrd data and reference data.

        Args:
            mask (list[list[float, float]], optional): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...]. Defaults to None.
            height (float, optional): The height above which peaks could be detected. Defaults to 0.06.
            mask_height (float, optional): The height above which peaks could be detected in masked area. Defaults to 0.12.
            tolerance (float, optional): The tolerance of the peak angle, in degree.
                If the difference between two peaks is less than tolerance, they are considered to be similar. Defaults to 0.3.
            display_number (int, optional): Number of reference spectrums to be displayed. Defaults to 5.
            figure_title (str, optional): The title of the figure. Defaults to ''.
            window (int, optional): The parameter of smooth. Defaults to 101.
            factor (float, optional): The parameter of smooth. Defaults to 0.5.
            save_path (str, optional): If the figure needs to be saved, please set the path. Defaults to ''.
            if_show (bool, optional): If the figure needs to be shown immediately. Defaults to True.
            lam (int, optional): The parameter of substract_baseline. Defaults to 200.
            if_process (bool, optional): If the data needs to be processed before identification. Defaults to True.

        Returns:
            None | plt.axes._axes.Axes: If if_show is True, return None. Otherwise, return the axes of the figure.
        '''
        # Process data
        if if_process:
            data = self.substract_baseline(lam=lam).smooth(window=window, factor=factor)
        else:
            data = self.copy()
        _, peaks_index, _ = data.peaks(mask=mask, height=height, mask_height=mask_height)
        # Compare peaks and get rating
        match_result = data.match(tolerance=tolerance, mask=mask, height=height, mask_height=mask_height)
        with open(self.pattern_path, 'rb') as db:
            pattern_database = pickle.load(db)
        with open(self.structure_path, 'rb') as db:
            structure_database = pickle.load(db)
        # Display the top display_number reference data
        if display_number > len(pattern_database):
            display_number = len(pattern_database)
        # Plot
        plt.rc('font', family='Calibri', size=10)
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('tab10')(np.linspace(0, 1, 10)))
        # Plot the data and the peak
        fig, ax = plt.subplots(display_number + 1, 1, figsize=(12, 9))
        ax[0].plot(data.two_theta, data.intensity, label="Smoothed and substracted")
        ax[0].plot(data.two_theta[peaks_index], data.intensity[peaks_index], "x", label="Peaks")
        ax[0].set_xlim(8, 60)
        if figure_title:
            ax[0].set_title(figure_title, fontsize=10)
        # Plot reference data
        adpc = XRDCalculator()
        possible_index = []
        # If elements are given, only display the reference data that contains the elements
        if elements:
            index_to_remove = []
            for index, possible_ref in enumerate(match_result):
                # If possible_ref doesn't contain all the elements, remove it from the list
                for element in elements:
                    if not element.startswith('-'):
                        if element not in possible_ref:
                            index_to_remove.append(index)
                            break
                    else:  # If the element is in the negative form, the reference data should not contain the element
                        if element.lstrip('-') in possible_ref:
                            index_to_remove.append(index)
                            break
            for index in reversed(index_to_remove):
                match_result.pop(index)
        if len(match_result) < display_number:
            display_number = len(match_result)
        for possible_ref in match_result[0:display_number]:
            for index, pattern in enumerate(pattern_database):
                if pattern['name'] == possible_ref:
                    possible_index.append(index)
                    break
        structure_to_plot = [structure_database[index] for index in possible_index]
        pattern_to_plot = [pattern_database[index] for index in possible_index]
        for index, (structure, pattern) in enumerate(zip(structure_to_plot, pattern_to_plot)):
            adpc.get_plot(structure, two_theta_range=(10, 60), fontsize=8, ax=ax[index + 1], with_labels=False)
            ax[index + 1].set_ylim(0, 120)
            ax[index + 1].set_xlim(8, 60)
            ax[index + 1].set_title(f"{pattern['formula']} {pattern['space_group']} ({pattern['space_group_number']}), {pattern['icsd_code']}", fontsize=10)
        plt.xlabel(r"$2\theta$ ($^\circ$)", fontsize=10)
        plt.subplots_adjust(hspace=0.4, bottom=0.057)
        if save_path:
            fig.savefig(save_path, dpi=600)
        if if_show:
            plt.show()
            return None
        plt.close(fig)
        return ax
