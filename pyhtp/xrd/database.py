# -*- coding: utf-8 -*-
"""
This module contains the class for XRD database.
- Author: Junyuan Lu
- E-mail: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
from typing import Literal
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..typing import SampleInfo, AngleRange, PatternInfo, PeakParam
from .pattern import XRDPattern
from .cifdatabase import CIFDatabase

PeakInfoType = tuple[NDArray[np.int_], dict[str, NDArray[np.float64]]]


class XRDDatabase:
    """A class that contains diffraction patterns of a high-throughput sample.

    For a high-throughput characterized sample, multiple diffraction patterns of different
    positions on the sample are collected. Therefore, the class is for the storage and
    processing of the diffraction data of sample level.

    Attributes:
        data (list[XRDPattern]): All the XRD data of different positioned points are stored
            in a list. The element in the list is the XRDPattern class instance. Please refer
            to the XRDPattern class for more information.
        info (SampleInfo): The information of the sample. Please refer to the SampleInfo NamedTuple
            for more information.
    """
    def __init__(
            self,
            info: SampleInfo,
            file_dir: str | list[str] | None = None,
            data: list[XRDPattern] | None = None):
        """Initialize the XrdDatabase.

        There are two ways to import the data.
        - Import the data from the file directory.
        - Import the data from a existing XRDPattern list. This method is used in the copy function.
            Since the element in the list is not verified, please use this method carefully.

        Args:
            info (SampleInfo): The information of the sample. Please refer to the SampleInfo NamedTuple.
                For XRD analysis, the two_theta_range must be set.
            file_dir (str | list[str] | None, optional): The directory of the database. Input a list if
                there are multiple scan angle ranges. Defaults to None.
            data (list[XrdPattern] | None, optional): A list of XrdPattern. Defaults to None.

        Raises:
            ValueError: The two_theta_range should be set for XRD analysis.
            ValueError: The file_dir and info.two_theta_range should be list together or str, AngleRange.
                If list, the length should be the same.
            FileNotFoundError: Some directories in the file_dir is not existing.
            ValueError: Either file_dir or data should be set.
            ValueError: The number of files in file_dir are not identical.
            ValueError: The AngleRange in info.two_theta_range are not continuous.
        """
        if info.two_theta_range is None:
            raise ValueError('The SampleInfo.two_theta_range should be set for XRD analysis!')
        if file_dir and data is None:
            if not ((isinstance(file_dir, list)
                     and isinstance(info.two_theta_range, list)
                     and len(file_dir) == len(info.two_theta_range))
                    or (isinstance(file_dir, str)
                        and isinstance(info.two_theta_range, AngleRange))):
                raise ValueError(
                    'The file_dir and info.two_theta_range should be list together \
                        or str, AngleRange! If list, the length should be the same!')
            if isinstance(file_dir, str) and not os.path.isdir(file_dir):
                raise FileNotFoundError(f'{file_dir} in the file_dir is not found!')
            if isinstance(file_dir, list):
                file_numbers = []
                for path in file_dir:
                    if not os.path.isdir(path):
                        raise FileNotFoundError(f'{path} in the file_dir is not found!')
                    file_numbers.append(
                        len([file_name for file_name in os.listdir(path)
                            if file_name.endswith('.xy')]))
                if len(set(file_numbers)) != 1:
                    raise ValueError(
                        f'The number of files in file_dir {file_numbers} are not identical!')

            self._file_dir = file_dir
            self.data = self._read_data(file_dir, info)
        elif data and file_dir is None:
            self.data = data
        else:
            raise ValueError('Either file_dir or data should be set!')
        # Set the info
        if isinstance(info.two_theta_range, list):
            info = info._replace(
                angle_range=AngleRange(info.two_theta_range[0].left,
                                       info.two_theta_range[-1].right))
        info = info._replace(point_number=len(self.data))
        self.info = info
        # Unify the length of the patterns
        self._unify_length()

    def _unify_length(self) -> None:
        """Unify the length of all the patterns.

        The function will unify the length of all the patterns by cropping the patterns.
        """
        min_length = min(len(pattern.intensity) for pattern in self.data)
        for pattern in self.data:
            pattern.intensity = pattern.intensity[:min_length]
            pattern.two_theta = pattern.two_theta[:min_length]

    @property
    def intensity(self) -> NDArray[np.float64]:
        """Return the intensity of all patterns.

        Returns:
            NDArray: The intensity of all patterns. Shape: (n_patterns, n_points).
        """
        return np.array([pattern.intensity for pattern in self.data])

    @property
    def two_theta(self) -> NDArray[np.float64]:
        """Return the two_theta of all patterns.

        Returns:
            NDArray: The two_theta of all patterns. Shape: (n_patterns, n_points).
        """
        return np.array([pattern.two_theta for pattern in self.data])

    @staticmethod
    def _read_data(
            file_dir: str | list[str],
            info: SampleInfo) -> list[XRDPattern]:
        """Read the data from the xy file directory.

        Args:
            file_dir (list[str]): The directory of the database. Input a list if
                there are multiple scan angles.
            info (SampleInfo): The information of the sample. Please refer to the
                SampleInfo NamedTuple.

        Returns:
            list[XrdPattern]: The list of XrdPattern.
        """
        def _read_xy(
                info: SampleInfo,
                file_path: str,
                angle_range: AngleRange,
                index: int) -> XRDPattern:
            '''Read xrd data from a single .xy files, and return a XrdPattern object.
            '''
            # Read the file into a List[str] and pop the first line
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines.pop(0)
            # Split each line of data into a list: [two-theta, intensity]
            data = np.array([line.split() for line in lines], dtype=float)
            # Convert the list into a numpy array
            assert isinstance(info.temperature, (int, float))  # temperature can only be float for xrd
            pattern = XRDPattern(
                two_theta=data[:, 0], intensity=data[:, 1],
                info=PatternInfo(
                    name=info.name, element=info.element,
                    temperature=info.temperature,
                    two_theta_range=angle_range, index=index))
            return pattern

        def _combine_xy(
                info: SampleInfo,
                left: XRDPattern,
                right: XRDPattern) -> XRDPattern:
            """Combine two XrdPattern into one.
            """
            # Detect if the range of two angle ranges are continuous
            if left.info.two_theta_range[1] != right.info.two_theta_range[0]:
                raise ValueError(
                    f'The two angle ranges {left.info.two_theta_range} and \
                        {right.info.two_theta_range} are not continuous!')
            # Calculate the average intensity on the overlapping area
            left_avg_mask = ((left.two_theta > left.info.two_theta_range[1] - 0.1)
                             & (left.two_theta < left.info.two_theta_range[1]))
            right_avg_mask = ((right.two_theta > right.info.two_theta_range[0])
                              & (right.two_theta < right.info.two_theta_range[0] + 0.1))
            left_avg = left.intensity[left_avg_mask].mean()
            right_avg = right.intensity[right_avg_mask].mean()
            # Normalize the intensity of the right pattern
            right.intensity = right.intensity + left_avg - right_avg
            # Merge the two arrays
            two_theta = np.concatenate((left.two_theta, right.two_theta))
            intensity = np.concatenate((left.intensity, right.intensity))
            assert isinstance(info.temperature, (int, float))  # temperature can only be float for xrd
            new_info = PatternInfo(
                name=info.name,
                index=left.info.index,
                two_theta_range=AngleRange(left.info.two_theta_range[0], right.info.two_theta_range[1]),
                element=info.element,
                temperature=info.temperature)
            return XRDPattern(two_theta=two_theta, intensity=intensity, info=new_info)

        assert isinstance(info.two_theta_range, (AngleRange, list))
        if isinstance(file_dir, str):
            file_dir = [file_dir]

        # Read the data
        data: list[XRDPattern] = []
        for i, path in enumerate(file_dir):
            # Exclude the files that are not .xy files
            file_names = [file_name for file_name in os.listdir(path) if file_name.endswith('.xy')]
            # Read the data
            for j, file_name in enumerate(file_names):
                two_theta_range = (
                    info.two_theta_range[i] if isinstance(info.two_theta_range, list)
                    else info.two_theta_range)
                pattern = _read_xy(info, os.path.join(path, file_name), two_theta_range, j)
                # For the first pattern, append it to the list; for the rest, combine them
                if i == 0:
                    data.append(pattern)
                else:
                    data[j] = _combine_xy(info, data[j], pattern)
        return data

    def copy(self) -> XRDDatabase:
        """Copy the XRDDatabase object.

        Returns:
            XXRDDatabase: The deep copied XRDDatabase object.
        """
        return XRDDatabase(data=self.data.copy(), info=self.info)

    def subtract_baseline(
            self,
            baseline_lam: int = 200) -> XRDDatabase:
        """Subtract the baseline of the diffraction patterns.

        This function calls the subtract_baseline function of all the XRDPatterns.

        Args:
            baseline_lam (int, optional): Subtract parameter. Please refer to
                https://pybaselines.readthedocs.io/en/latest/parameter_selection.html
                for more infomation. If lam is illegal (lam <= 0), the function will only
                return a copy of the original database. Defaults to 200.

        Returns:
            XRDDatabase: The subtracted XRDDatabase object.
        """
        if baseline_lam <= 0:
            return self.copy()
        new_db = self.copy()
        for index, pattern in enumerate(new_db.data):
            new_db.data[index] = pattern.subtract_baseline(baseline_lam)
        return new_db

    def smooth(
            self,
            window: int = 51,
            spline_lam: float | None = None) -> XRDDatabase:
        """Smooth the diffraction patterns.

        Please check XRDPattern.smooth for more information.

        Args:
            window (int, optional): The window size of the smoothing. Defaults to 101.
            spline_lam (float | None, optional): The lambda of the spline. If <= 0, the function
                will only return a copy of the original database. Defaults to None.

        Returns:
            XRDDatabase: The smoothed XRDDatabase object.
        """
        if window <= 3 and spline_lam and spline_lam <= 0:
            return self.copy()
        new_db = self.copy()
        for index, pattern in enumerate(new_db.data):
            new_db.data[index] = pattern.smooth(window, spline_lam)
        return new_db

    def postprocess(
            self,
            baseline_lam: int = 200,
            window: int = 51,
            spline_lam: float | None = None,
            concurrent: bool = True,
            **kwargs) -> XRDDatabase:
        """Post-process the diffraction patterns.

        Please refer to XRDPattern.postprocess for more information. If the data is large
        enough (len(self.data) > 100), the function will use the concurrent method to process
        the data. A ProcessPoolExecutor with 4 workers and a chunksize of 10 will be used.
        The worker number and chunksize is tested on my computer. You can change the number by
        passing kwargs to the function.

        Args:
            baseline_lam (int, optional): Subtract parameter. Defaults to 200.
            window (int, optional): The window size of the smoothing. Defaults to 101.
            spline_lam (float | None, optional): The lambda of the spline. Defaults to None.
            concurrent (bool, optional): If True, the function will use the concurrent method
                to process the data if self.data is large. Defaults to True.
            **kwargs: Other arguments for ProcessPool.
                - max_workers: The maximum number of workers. Defaults to 4.
                - chunksize: The chunksize of the data. Defaults to 10.

        Returns:
            XRDDatabase: The post-processed XRDDatabase object.
        """
        if len(self.data) < 100 or not concurrent:
            return self.subtract_baseline(
                baseline_lam=baseline_lam).smooth(
                    window=window, spline_lam=spline_lam)
        else:
            new_db = self.copy()
            with ProcessPoolExecutor(max_workers=kwargs.get('max_workers', 4)) as executor:
                new_db.data = list(
                    executor.map(
                        self._concurrent_process,
                        self.data,
                        [baseline_lam] * len(self.data),
                        [window] * len(self.data),
                        [spline_lam] * len(self.data),
                        chunksize=kwargs.get('chunksize', 10)))
            return new_db

    @staticmethod
    def _concurrent_process(
            pattern: XRDPattern,
            baseline_lam: int,
            window: int,
            spline_lam: float | None) -> XRDPattern:
        """Process the pattern with the given parameters."""
        return pattern.postprocess(
            baseline_lam=baseline_lam, window=window, spline_lam=spline_lam)

    def plot(
            self,
            index: list[int] | None = None,
            style: Literal['combine', 'stack'] = 'combine',
            cmap: str | Colormap = 'viridis',
            ax: Axes | None = None,
            amorphous_index: int = -1,
            **kwargs) -> None:
        """Plot the diffraction patterns.

        Args:
            index (list[int]] | None, optional): The index of patterns to plot.
                If None, all the patterns will be plot. Defaults to None.
            style (Literal['combine', 'stack'], optional): Plot mode. For combine mode,
                all the patterns are overlapped without any offset. For stack mode, the patterns
                are stacked with an offset from bottom to the top. In stack mode, the peak will
                be marked. Defaults to 'combine'.
            ax (Axes | None, optional): The axes to plot. Defaults to None.
            amorphous_index (int, optional): The index of a reference amorphous pattern.
                If set, intensities will be subtracted by the reference. Defaults to -1.
            **kwargs: Other arguments for ``XRDPattern.get_peak()`` and plot. Please refer to the
                function for more information.
                - mask: The mask for the peak detection. Defaults to None.
                - param: The parameter for the peak detection. Defaults to None.
                - mask_param: The parameter for the mask. Defaults to None.
                - alpha: The transparency of the curves. Defaults to 0.8.
        """
        # Process the pattern to be plot
        if index is None:
            index = list(range(len(self.data)))
        assert isinstance(index, list)
        # Select the patterns to plot
        patterns = [self.data[i] for i in index]
        # Process the patterns with kwargs given
        for i, pattern in enumerate(patterns):
            if amorphous_index >= 0:
                patterns[i].intensity = patterns[i].intensity - self.data[amorphous_index].intensity
        # Plot the patterns
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)
        # Get the colormap
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        if style == 'combine':
            for i, pattern in enumerate(patterns):
                pattern.plot(
                    ax=ax, color=cmap(i / len(patterns)),
                    if_label=False, alpha=kwargs.get('alpha', 0.8))
        if style == 'stack':
            # Get the margin from the maximum intensity variation of the patterns
            margin = max(self.intensity.max(axis=1))
            for i, pattern in enumerate(patterns):
                pattern.plot(
                    ax=ax, color=cmap(i / len(patterns)), offset=i * margin,
                    alpha=kwargs.get('alpha', 0.8), if_label=False,
                    mask=kwargs.get('mask', None),
                    param=kwargs.get('param', None),
                    mask_param=kwargs.get('mask_param', None),
                    max_intensity=self.intensity.max())

    def plot_with_ref(
            self,
            cif_database: CIFDatabase,
            icsd_code: int | list[int],
            ref_cmap: str | Colormap = 'tab10',
            **kwargs) -> None:
        """Plot the diffraction patterns with reference phases XRD data.

        Args:
            cif_database (CIFDatabase): The cif database.
            icsd_code (int | list[int]): The icsd code of the reference phases.
            ref_cmap (str | Colormap, optional): The colormap for the reference phases.
                Defaults to 'tab10'.
            **kwargs: Other arguments for ``XRDDatabase.plot()``. DO NOT pass ``ax``
                since the function will create a new figure.
        """
        if isinstance(icsd_code, int):
            icsd_code = [icsd_code]
        cif_data = [cif_database.data[cif_database.index(icsd_code=c)[0]]
                    for c in icsd_code]
        # Plot the sample patterns
        _, axs = plt.subplots(
            1 + len(cif_data), 1,
            figsize=(6, 2 * (1 + len(cif_data))))
        self.plot(ax=axs[0], **kwargs)
        # Get the xlim
        xlim = axs[0].get_xlim()
        # Plot the cif patterns
        if isinstance(ref_cmap, str):
            ref_cmap = plt.cm.get_cmap(ref_cmap)
        for i, c in enumerate(icsd_code):
            axs[i + 1] = cif_database.plot(
                ax=axs[i + 1], color=ref_cmap(i), icsd_code=c,
                two_theta_range=AngleRange(*xlim))
        plt.show()

    def plot_surf(
            self,
            index: list[int] | None = None,
            cmap: str | Colormap = 'viridis',
            ax: Axes3D | None = None,
            baseref_index: int = -1) -> None:
        """Plot the diffraction patterns in 3D surface.

        Args:
            index (Optional[list[int]], optional): Index to plot. Defaults to None.
            ax (Optional[Axes3D], optional): matplotlib axis. Defaults to None.
            amorphous_index (int, optional): If set, subtract baseline based on a reference point. Defaults to -1.

        Raises:
            ValueError: The ax is not correctly set!
        """
        # Process the pattern to be plot
        if index is None:
            index = list(range(len(self.data)))
        # Select the patterns to plot
        patterns = [self.data[i] for i in index]
        # Process the patterns with kwargs given
        for pattern in patterns:
            if baseref_index >= 0:
                pattern.intensity = pattern.intensity - self.data[baseref_index].intensity
        # Plot the patterns
        fig = None
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})  # type: ignore
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        assert ax
        # Convert all the intensity data into a surface
        x = np.array(patterns[0].two_theta)
        y = np.arange(len(patterns))
        x_value, y_value = np.meshgrid(x, y)
        z_value = self.intensity[index]
        # Plot the surface
        ax.plot_surface(x_value, y_value, z_value, cmap=cmap)
        ax.set_box_aspect([4, 3, 1])

    def get_peak(
            self,
            mask: AngleRange | list[AngleRange] | None = None,
            param: PeakParam | None = None,
            mask_param: PeakParam | None = None,
            index: list[int] | NDArray[np.int_] | None = None
    ) -> list[PeakInfoType]:
        """Return the peak informations of all the patterns.

        This function call the ``XRDPattern.get_peak()`` functions. Please refer
        to the function and the nametuple ``PeakParam`` for more information.

        Args:
            mask (AngleRange | list[AngleRange], optional): The mask for the peak detection.
                Defaults to None.
            param (PeakParam, optional): The parameter for the peak detection. Defaults to None.
            mask_param (PeakParam, optional): The parameter for the mask. Defaults to None.
            index (list[int] | NDArray[np.int_] | None, optional): The index of the patterns to
                get the peak. If None, all the patterns will be processed. Defaults to None.

        Returns:
            list[PeakInfoType]: The peak informations of all the patterns.
                - The first element is the peak index array. The index is that of the two_theta
                    and intensity array.
                - The second element is a dict that contains the peaks' properties. The keys
                    include ``peak_heights``, ``prominences``, ``left_bases``, ``right_bases``,
                    and ``peak_angles``.

        """
        if index is None:
            index = list(range(len(self.data)))
        patterns_to_process = [self.data[i] for i in index]
        peak_info = []
        for pattern in patterns_to_process:
            peak_info.append(
                pattern.get_peak(
                    mask, param, mask_param,
                    max_intensity=self.intensity.max()))
        return peak_info

    def get_peak_two_theta(
            self,
            param: PeakParam | None = None,
            mask: AngleRange | list[AngleRange] | None = None,
            mask_param: PeakParam | None = None,
            index: list[int] | NDArray[np.int_] | None = None
    ) -> list[NDArray[np.float64]]:
        """Return the peak two_theta of all the patterns.

        Args:
            param (PeakParam, optional): The parameter for the peak detection. Defaults to None.
            mask (AngleRange | list[AngleRange], optional): The mask for the peak detection.
                Defaults to None.
            mask_param (PeakParam, optional): The parameter for the mask. Defaults to None.
            index (list[int] | NDArray[np.int_] | None, optional): The index of the patterns to
                get the peak. If None, all the patterns will be processed. Defaults to None.

        Returns:
            list[NDArray]: The peak two_theta of all the patterns.
        """
        peak_info = self.get_peak(mask, param, mask_param, index)
        return [info[1]['peak_angles'] for info in peak_info]

    def classify_legacy(
            self,
            n_clusters: tuple[int, int] | int = (2, 10),
            msg_display: int = 0,
            **kwargs) -> NDArray:
        """Cluster the diffraction patterns based on peak_two_theta.

        Since the length of peak_two_theta for different patterns are not the same,
        a encoder is used to convert them into the same shape before classification.
        Please check the ``XRDDatabase.peak_encoder()`` for more information.

        Args:
            n_clusters (tuple[int, int] | int, optional): n_cluster range. If int,
                only one cluster number value will be used. Defaults to (2, 10).
            msg_display (int, optional): The verbose level. Defaults to 0.
                Please refer to sklearn.cluster.KMeans for more information.
            **kwargs: Other arguments for ``XRDDatabase.get_peak()`` and
                ``XRDDatabase.peak_encoder()``. Please refer to the functions for more information.
                - mask (AngleRange | list[AngleRange], optional): The mask for the peak detection.
                - param (PeakParam, optional): The parameter for the peak detection.
                - mask_param (PeakParam, optional): The parameter for the mask.
                - two_theta_tol (int | float, optional): The tolerance for the two_theta.
                - peak_number (int | None, optional): The number of peaks for the encoding.

        Returns:
            NDArray: The clustering label of all the patterns.
        """
        from sklearn.cluster import KMeans  # type: ignore  # pylint: disable=import-outside-toplevel

        if isinstance(n_clusters, int):
            n_clusters = (n_clusters, n_clusters + 1)

        # Copy the database
        db = self.copy()

        # Get the peak angles of the patterns
        # Since the module is temporarily for thin film, the peak intensity is not considered
        peak_two_theta = db.get_peak_two_theta(
            param=kwargs.pop('param', None),
            mask=kwargs.pop('mask', None),
            mask_param=kwargs.pop('mask_param', None))
        # The length of elements in peak_data are not the same
        # Encode the peaks to a matrix with shape (n_patterns, features)
        encoded_two_theta = db.peak_encoder_legacy(
            two_theta=peak_two_theta,
            two_theta_tol=kwargs.pop('two_theta_tol', 0.2),
            peak_number=kwargs.pop('peak_number', None))

        # KMeans clustering
        results = []
        scores = []
        for k in range(*n_clusters):
            result = KMeans(
                n_clusters=k, random_state=0, tol=1e-5,
                verbose=msg_display).fit(encoded_two_theta)
            label = result.labels_
            score = silhouette_score(encoded_two_theta, label)
            scores.append(score)
            results.append(result)
            if msg_display > 0:
                print(f'Classification with {k} clusters, \
                      silhouette score: {score:.2f}')
        best_index = np.argmax(scores)
        label = results[best_index].labels_
        if msg_display > 0:
            print(f'Finish with {best_index + n_clusters[0]} clusters, \
                  silhouette score: {max(scores):.2f}')
        return label

    def classify(
            self,
            method: Literal['hdbscan', 'kmeans'] = 'hdbscan',
            n_clusters: tuple[int, int] | int = (2, 10),
            phase_of_peaks: list[str] | NDArray[np.str_] | None = None,
            **kwargs) -> NDArray[np.str_]:
        """Cluster the diffraction patterns based on peak_two_theta.

        Since the length of peak_two_theta for different patterns are not the same,
        a encoder is used to convert them into the same shape before classification.
        Please check the ``XRDDatabase.peak_encoder()`` for more information.

        Args:
            method (Literal[&#39;hdbscan&#39;, &#39;kmeans&#39;], optional):
                The algorithm used to cluster. Defaults to 'hdbscan'.
            n_clusters (int | None, optional): The cluster number for kmeans.
                Defaults to (2, 10). If tuple, all the cluster numbers in the range
                will be tested and the best one with highest silhouette_score will be selected.
            phase_of_peaks (list[str] | NDArray[np.str_] | None, optional): The phase of each
                characteristic peak. Automatic phase detection from cif_database is not
                implemented yet. Please note that the order of the phase should be
                from low angle peak to high angle peak. Defaults to None.
                - If cif_database is given, the result label will contain phase information.
                - If not, the label will be integers.
            **kwargs: Other arguments for ``XRDDatabase.get_peak()`` and
                ``XRDDatabase.peak_encoder()``. Please refer to the functions for more information.
                - mask (AngleRange | list[AngleRange], optional): The mask for the peak detection.
                - param (PeakParam, optional): The parameter for the peak detection.
                - mask_param (PeakParam, optional): The parameter for the mask.
                - two_theta_tol (int | float, optional): The tolerance for the two_theta.
                - peak_number (int | None, optional): The number of peaks for the encoding. If None,
                    hdbscan will be used to cluster the peaks without given cluster number.
                - min_cluster_size (int, optional): The minimum cluster size for hdbscan.
                    Defaults to 10.

        Returns:
            NDArray: The clustering label of all the patterns.
        """
        if isinstance(n_clusters, int):
            n_clusters = (n_clusters, n_clusters + 1)

        # Get the peak angles of the patterns
        # Since the module is temporarily for thin film, the peak intensity is not considered
        peak_two_theta = self.get_peak_two_theta(
            param=kwargs.get('param', None),
            mask=kwargs.get('mask', None),
            mask_param=kwargs.get('mask_param', None))
        # The length of elements in peak_data are not the same
        # Encode the peaks to a matrix with shape (n_patterns, features)
        encoded_two_theta = self.peak_encoder(
            two_theta=peak_two_theta,
            two_theta_tol=kwargs.get('two_theta_tol', 0.2),
            peak_number=kwargs.pop('peak_number', None))
        # HDBSCAN clustering
        if method == 'hdbscan':
            from sklearn.cluster import HDBSCAN  # type: ignore  # pylint: disable=import-outside-toplevel
            model = HDBSCAN(
                min_cluster_size=kwargs.pop('min_cluster_size', 10),
                store_centers='both',
                n_jobs=-1).fit(encoded_two_theta)
        if method == 'kmeans':
            from sklearn.cluster import MiniBatchKMeans  # pylint: disable=import-outside-toplevel
            scores = []
            results = []
            for k in range(*n_clusters):
                model = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=0).fit(encoded_two_theta)
                score = silhouette_score(encoded_two_theta, model.labels_)
                scores.append(score)
                results.append(model)
            best_index = np.argmax(scores)
            model = results[best_index]
        print(f'Finish with {len(set(model.labels_))} clusters, \
              silhouette score: {silhouette_score(encoded_two_theta, model.labels_):.2f}')
        # Return directly if phase_of_peaks is not given
        if phase_of_peaks is None:
            return model.labels_.astype(str)

        phase_cluster = self.get_label_modifier(
            model=model,
            two_theta=peak_two_theta,
            two_theta_tol=kwargs.get('two_theta_tol', 0.2),
            phase_of_peaks=phase_of_peaks)
        # Get the new label
        phase_label = model.labels_.copy().astype(str)
        for key, value in phase_cluster.items():
            for i in value:
                phase_label[phase_label == i] = key
        return phase_label

    def get_label_modifier(
            self,
            model,
            two_theta: list[NDArray[np.float64]],
            two_theta_tol: int | float,
            phase_of_peaks: list[str] | NDArray[np.str_],
            save_dir: str = '') -> dict[str, list[str]]:
        """Get the label to phase name modifier for the clustering result.

        Args:
            model (MiniBatchKmeans | HDBSCAN): The clustering model of sklearn.clustering.
            two_theta (list[NDArray[np.float64]]): The peak angles of all the patterns.
            two_theta_tol (int | float): The tolerance for the two_theta. This parameter
                is used to determine if two peaks are the same in generating existing matrix.
            phase_of_peaks (list[str] | NDArray[np.str_]): The phase of each characteristic peak.
                The order of the phase should be from low angle peak to high angle peak.
            save_dir (str, optional): The directory to save the label modifier. Defaults to ''.
                If not given, the modifier will not be saved.

        Returns:
            dict[str, list[str]]: The label to phase name modifier.
                - Key: The phase name.
                - Value: The index of the cluster.
        """
        # Use HDBSCAN to cluster, since the peak number of each cluster is unknown
        from sklearn.cluster import HDBSCAN  # type: ignore  # pylint: disable=import-outside-toplevel

        # Else, calculate the phase of each cluster
        existing_peaks = self.existing_two_theta(
            two_theta=two_theta, two_theta_tol=two_theta_tol)
        peak_exist_matrix = np.zeros((len(set(model.labels_)), len(existing_peaks)))

        for label in set(model.labels_):
            index = np.where(model.labels_ == label)[0]
            current_two_theta = [two_theta[i] for i in index]
            current_two_theta = np.concatenate(current_two_theta)
            # Set the min_cluster_size to 80% of the data
            # Which means that only if a peak appears at most of the patterns
            # it will be considered as a characteristic peak of this cluster
            min_cluster_size = int(0.8 * len(index))
            current_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                store_centers='both',
                n_jobs=-1).fit(current_two_theta)
            cluster_centers = current_model.centroids_.flatten()
            for angle in cluster_centers:
                # If the angle is within the tol of a existing peak angle
                # The matrix[angle, j] will be set to 1
                nearest_index = np.argmin(np.abs(existing_peaks - angle))
                if np.abs(existing_peaks[nearest_index] - angle) < two_theta_tol:
                    peak_exist_matrix[label, nearest_index] = 1

        # Check if two rows have the same diffraction peaks distribution
        # If exist, the clustering is probably not good
        for i in range(peak_exist_matrix.shape[0]):
            for j in range(i + 1, peak_exist_matrix.shape[0]):
                if np.all(peak_exist_matrix[i] == peak_exist_matrix[j]):
                    print(f'Cluster {i} and Cluster {j} have the same phase.')

        # Get the phase of each cluster
        # The key is the phase name, the value is the index of the cluster
        # The original label generated from clustering are integers
        # The new label will be the phase name
        phase_cluster = {}
        # 1 in peak_exist_matrix means the phase is existing
        for index in range(peak_exist_matrix.shape[0]):
            # If all 0, set phase name to be amorphous
            if np.all(peak_exist_matrix[index] == 0):
                phase_name = 'Amorphous'
            else:
                current_phase = []
                for i, value in enumerate(peak_exist_matrix[index]):
                    if value == 1 and phase_of_peaks[i] not in current_phase:
                        current_phase.append(phase_of_peaks[i])
                # Sort current_phase
                current_phase.sort()
                phase_name = ' + '.join(current_phase)
            if phase_name not in phase_cluster:
                phase_cluster[phase_name] = [str(index)]
            else:
                phase_cluster[phase_name].append(str(index))

        if save_dir:
            import json  # pylint: disable=import-outside-toplevel
            with open(
                    os.path.join(save_dir, 'label_modifier.json'), 'w',
                    encoding='utf-8') as f:
                json.dump(phase_cluster, f)

        return phase_cluster

    def peak_encoder(
            self,
            two_theta: list[NDArray[np.float64]],
            two_theta_tol: int | float = 0.2,
            peak_number: int | None = None) -> NDArray[np.float64]:
        """Encode the peaks based on the combine peaks of the sample.

        The resulting peak vector of all the patterns have the same length as the combine peaks.

        Args:
            two_theta (list[NDArray]): The peak angles of all the patterns. [pattern_1, pattern_2, ...]
            two_theta_tol (float, optional): 2 theta tolerance. Defaults to 0.w. If the distance
                between two peaks are bigger than two_theta_tol, the similarity is directly set to 0.
            peak_number (int, optional): The number of clusters for kmeans. Defaults to None.
                If None, the function will use the peak quantity of the pattern with most peaks.

        Returns:
            NDArray[float]: The encoded peaks. Shape: (n_patterns, features).
        """
        existing_peaks = self.existing_two_theta(
            two_theta=two_theta,
            peak_number=peak_number)
        # Encode the peaks
        encoded_data = np.zeros((len(two_theta), len(existing_peaks)))
        for i, angles in enumerate(two_theta):
            if len(angles) == 0:
                continue
            for j, peak in enumerate(existing_peaks):
                # Calculate the nearest peak in angles to the specific ch_peak
                nearest_index = np.argmin(np.abs(angles - peak))
                # If the difference is too large, set 0
                if np.abs(angles[nearest_index] - peak) > two_theta_tol:
                    encoded_data[i, j] = 0
                    continue
                encoded_data[i, j] = 1 - np.abs(angles[nearest_index] - peak) / peak
        return encoded_data

    def peak_encoder_legacy(
            self,
            two_theta: list[NDArray[np.float64]],
            two_theta_tol: int | float = 0.2,
            peak_number: int | None = None) -> NDArray[np.float64]:
        """Encode the peaks based on the combine peaks of the sample.

        The resulting peak vector of all the patterns have the same length as the combine peaks.

        Args:
            two_theta (list[NDArray]): The peak angles of all the patterns. [pattern_1, pattern_2, ...]
            two_theta_tol (float, optional): 2 theta tolerance. Defaults to 0.w. If the distance
                between two peaks are bigger than two_theta_tol, the similarity is directly set to 0.
            peak_number (int, optional): The number of clusters for kmeans. Defaults to None.
                If None, the function will use the peak quantity of the pattern with most peaks.

        Returns:
            NDArray[float]: The encoded peaks. Shape: (n_patterns, features).
        """
        existing_peaks = self.existing_two_theta(
            two_theta=two_theta,
            peak_number=peak_number)
        # Encode the peaks
        encoded_data = np.zeros((len(two_theta), len(existing_peaks) + 1))
        for i, angles in enumerate(two_theta):
            if len(angles) == 0:
                continue
            for j, peak in enumerate(existing_peaks):
                # Calculate the nearest peak in angles to the specific ch_peak
                nearest_index = np.argmin(np.abs(angles - peak))
                # If the difference is too large, set 0
                if np.abs(angles[nearest_index] - peak) > two_theta_tol:
                    encoded_data[i, j] = 0
                    continue
                encoded_data[i, j] = 1 - np.abs(angles[nearest_index] - peak) / peak
        return encoded_data

    @staticmethod
    def existing_two_theta(
            two_theta: list[NDArray[np.float64]],
            peak_number: int | None = None,
            two_theta_tol: int | float = 0.2) -> NDArray[np.float64]:
        """Return the combined diffraction peaks of all the patterns.

        The peaks are clustered by kmeans.
        For example, if there are two patterns, the peaks angles are [22, 50] and [21, 40, 50], respectively.
        Then the combined peaks will be [21.5, 40, 50] or [21, 22, 40, 50], based on the n_cluster of kmeans.
        The function is for the diffraction peak encoding process in classify function.

        Args:
            two_theta (list[NDArray]): The peak angles of all the patterns. [pattern_1, pattern_2, ...]
            peak_number (int, optional): The number of clusters for kmeans. If peak_number is given,
                kmeans will be used to cluster the peaks. If None, hdbscan will be used to
                automatically cluster the peaks without given cluster number. Defaults to None.
            two_theta_tol (float, optional): The 2 theta tolerance for hdbscan. This parameter
                is passed to the cluster_selection_epsilon of hdbscan. Defaults to 0.2.

        Returns:
            NDArray[float]: The combined diffraction peaks.
        """
        if peak_number is None:
            # Get the max peak number as k_value for kmeans
            peak_number = max(len(i) for i in two_theta)
        assert isinstance(peak_number, int)
        # Get the characteristic peaks
        # First, concatenate all the peaks
        all_peaks = np.concatenate(two_theta).reshape(-1, 1)
        # Second, cluster the peaks
        if peak_number:
            from sklearn.cluster import MiniBatchKMeans  # pylint: disable=import-outside-toplevel
            # kmeans = KMeans(
            #     n_clusters=peak_number,
            #     random_state=0, tol=1e-5).fit(all_peaks)
            model = MiniBatchKMeans(
                n_clusters=peak_number,
                random_state=0).fit(all_peaks)
            cluster_centers = model.cluster_centers_.flatten()
        else:
            from sklearn.cluster import HDBSCAN  # pylint: disable=import-outside-toplevel  # type: ignore
            model = HDBSCAN(
                min_cluster_size=5,  # So that small phase areas can be detected
                store_centers='both',
                cluster_selection_epsilon=two_theta_tol,
                n_jobs=-1).fit(all_peaks)
            cluster_centers = model.centroids_.flatten()
        # Third, get the characteristic peaks as the center of each cluster
        return np.sort(cluster_centers)
