# -*- coding: utf-8 -*-
"""
A class that contains diffraction patterns of a high-throughput sample.
"""
from __future__ import annotations
import os
from typing import Union, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from ..typing import SampleInfo, AngleRange, PatternInfo
from .pattern import XrdPattern
from .icsd import ICSD


class XrdDatabase:
    """A class that contains diffraction patterns of a high-throughput sample.
    """
    def __init__(self,
                 info: SampleInfo,
                 file_dir: Optional[Union[str, list[str]]] = None,
                 data: Optional[list[XrdPattern]] = None):
        """Initialize the XrdDatabase.

        Args:
            file_dir (Union[str, list[str]]): The directory of the database. Input a list if there are multiple scan angles.
            info (SampleInfo): The information of the sample.

        Raises:
            ValueError: The length of file_path and scan_angle should be the same!
            ValueError: The scan_angle should be set for XRD sample!
        """
        if file_dir and data is None:
            # Check the input
            if isinstance(file_dir, str) and isinstance(info.angle_range, list):
                raise ValueError('The length of file_path and scan_angle should be the same!')
            if isinstance(file_dir, list) and not isinstance(info.angle_range, list):
                raise ValueError('The length of file_path and scan_angle should be the same!')
            if isinstance(file_dir, list) and isinstance(info.angle_range, list):
                if len(file_dir) != len(info.angle_range):
                    raise ValueError('The length of file_path and scan_angle should be the same!')
            if info.angle_range is None:
                raise ValueError('The scan_angle should be set for XRD sample!')
            self._file_dir = file_dir
            # Assign the attributes
            if isinstance(file_dir, str):
                file_dir = [file_dir]
            self.data = self._read_data(file_dir, info)
            # Change the angle range of self.info to a tuple if it is a list
            if isinstance(info.angle_range, list):
                self.info = info._replace(angle_range=AngleRange(info.angle_range[0].left, info.angle_range[-1].right))
            else:
                self.info = info
        elif data and not file_dir:
            self.data = data
            # Change the angle range of self.info to a tuple if it is a list
            if isinstance(info.angle_range, list):
                self.info = info._replace(angle_range=AngleRange(info.angle_range[0].left, info.angle_range[-1].right))
            else:
                self.info = info
        else:
            raise ValueError('Either file_dir or data should be set!')

    def _read_data(self, file_dir: list[str], info: SampleInfo) -> list[XrdPattern]:
        """Read the data from the file directory.

        Args:
            file_dir (list[str]): The directory of the database. Input a list if there are multiple scan angles.

        Raises:
            FileNotFoundError: The directory is not found!
            ValueError: The number of files in the directories are not the same!
            ValueError: The scan_angle should be set for XRD sample!

        Returns:
            list[XrdPattern]: The list of XrdPattern.
        """
        # Define the function to read the data from a single .xy file
        def _read_xy(info: SampleInfo,
                     file_path: str,
                     angle_range: AngleRange,
                     index: int) -> XrdPattern:
            '''Read xrd data from a single .xy files.

            Args:
                info (SampleInfo): The information of the sample.
                file_path (str): The file path of .xy files to be read.
                angle_range (AngleRange): The angle range of the pattern.
                index (int): The index of the pattern.

            Returns:
                XrdPattern: The XrdPattern object.
            '''
            # Read the file into a List[str] and pop the first line
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines.pop(0)
            # Split each line of data into a list: [2-theta, intensity]
            two_theta = []
            intensity = []
            for line in lines:
                line = line.split()
                line = [float(i) for i in line]
                two_theta.append(line[0])
                intensity.append(line[1])
            # Convert the list into a numpy array
            pattern = XrdPattern(two_theta=np.array(two_theta), intensity=np.array(intensity),
                                 info=PatternInfo(name=info.name, element=info.element,
                                                  temperature=info.temperature,
                                                  angle_range=angle_range, index=index))
            return pattern

        # Define the function to combine two XrdPattern into one
        def _combine_xy(info: SampleInfo,
                        left: XrdPattern,
                        right: XrdPattern) -> XrdPattern:
            """Combine two XrdPattern into one.

            Args:
                info (SampleInfo): The information of the sample.
                left (XrdPattern): The left pattern to be combined.
                right (XrdPattern): The right pattern to be combined.

            Raises:
                ValueError: The two patterns are not continuous!

            Returns:
                XrdPattern: The combined pattern.
            """
            # Detect if the range of two patterns are continuous
            if left.info.angle_range[1] != right.info.angle_range[0]:
                raise ValueError('The two patterns are not continuous!')
            # Calculate the average intensity on the overlapping area
            left_avg_mask = ((left.two_theta > left.info.angle_range[1] - 0.5)
                             & (left.two_theta < left.info.angle_range[1]))
            right_avg_mask = ((right.two_theta > right.info.angle_range[0])
                              & (right.two_theta < right.info.angle_range[0] + 0.5))
            left_avg = left.intensity[left_avg_mask].mean()
            right_avg = right.intensity[right_avg_mask].mean()
            # Normalize the intensity of the right pattern
            right.intensity = right.intensity + left_avg - right_avg
            # Merge the two arrays
            two_theta = np.concatenate((left.two_theta, right.two_theta))
            intensity = np.concatenate((left.intensity, right.intensity))
            new_info = PatternInfo(
                name=info.name,
                index=left.info.index,
                angle_range=AngleRange(left.info.angle_range[0], right.info.angle_range[1]),
                element=info.element,
                temperature=info.temperature)
            return XrdPattern(two_theta=two_theta, intensity=intensity, info=new_info)

        if info.angle_range is None:
            raise ValueError('The scan_angle should be set for XRD sample!')
        # Read the data
        data: list[XrdPattern] = []
        for i, path in enumerate(file_dir):
            if not os.path.isdir(path):
                raise FileNotFoundError(f'The directory {path} is not found!')
            if i > 1 and len(os.listdir(path)) != len(data):
                raise ValueError('The number of files in the directories are not the same!')
            # Read the data
            for j, file_name in enumerate(os.listdir(path)):
                if not file_name.endswith('.xy'):
                    continue
                angle_range = info.angle_range[i] if isinstance(info.angle_range, list) \
                    else info.angle_range
                pattern = _read_xy(info, os.path.join(path, file_name), angle_range, j)
                # For the first pattern, append it to the list; for the rest, combine them
                if i == 0:
                    data.append(pattern)
                else:
                    data[j] = _combine_xy(info, data[j], pattern)
        return data

    def copy(self) -> XrdDatabase:
        """Copy the XrdDatabase object.

        Returns:
            XrdDatabase: The copied XrdDatabase object.
        """
        return XrdDatabase(data=self.data, info=self.info)

    def subtract_baseline(self, lam: int = 200) -> XrdDatabase:
        """Subtract the baseline of the diffraction patterns.

        Args:
            lam (int, optional): Subtract parameter. Defaults to 200.

        Returns:
            XrdDatabase: The XrdDatabase object.
        """
        for index, pattern in enumerate(self.data):
            self.data[index] = pattern.subtract_baseline(lam=lam)
        return self

    def smooth(self, window: int = 101, factor: float = 0.5) -> XrdDatabase:
        """Smooth the diffraction patterns.

        Args:
            window (int, optional): The window size of the smoothing. Defaults to 101.
            factor (float, optional): The factor of the smoothing. Defaults to 0.5.

        Returns:
            XrdDatabase: The XrdDatabase object.
        """
        new_db = self.copy()
        for index, pattern in enumerate(new_db.data):
            new_db.data[index] = pattern.smooth(window=window, factor=factor)
        return new_db

    def plot(
            self,
            index: Optional[list[int]] = None,
            style: Literal['combine', 'stack'] = 'combine',
            dpi: int = 600,
            save_path: Optional[str] = None,
            ax: Optional[Axes] = None,
            amorphous_index: int = -1,
            **kwargs):
        """Plot the diffraction patterns.

        Args:
            index (Optional[list[int]], optional): The index of patterns to plot. Defaults to None.
            style (Literal['combine', 'stack'], optional): Mode. Defaults to 'combine'.
            dpi (int, optional): Output dpi. Defaults to 300.
            save_path (Optional[str], optional): If set, fig will be saved. Defaults to None.
            ax (Optional[Axes], optional): The axes to plot. Defaults to None.
            amorphous_index (int, optional): The index of a reference amorphous pattern. If set, intensities will be subtracted by the reference. Defaults to -1.
            **kwargs: Other arguments for post-processing and plotting.
        """
        # Process the pattern to be plot
        if index is None:
            index = list(range(len(self.data)))
        # Select the patterns to plot
        patterns = [self.data[i] for i in index]
        peaks: list[tuple] = []
        # Process the patterns with kwargs given
        for i, pattern in enumerate(patterns):
            if amorphous_index >= 0:
                patterns[i].intensity = patterns[i].intensity - self.data[amorphous_index].intensity
            if 'lam' in kwargs:
                patterns[i] = patterns[i].subtract_baseline(lam=kwargs['lam'])
            if 'window' in kwargs or 'factor' in kwargs:
                patterns[i] = patterns[i].smooth(
                    window=kwargs.get('window', 101),
                    factor=kwargs.get('factor', 0.5))
            peaks.append(
                pattern.get_peak(mask=kwargs.get('mask', None),
                                 height=kwargs.get('height', -1),
                                 mask_height=kwargs.get('mask_height', -1)))
        # Plot the patterns
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'))
        if style == 'combine':
            for i, pattern in enumerate(patterns):
                pattern.plot(ax=ax, color=cmap(i / len(patterns)),
                             if_label=False, alpha=0.8)
        if style == 'stack':
            # Get the margin from the maximum intensity variation of the patterns
            margin = max(pattern.intensity.max() - pattern.intensity.min() for pattern in patterns)
            for i, (pattern, peak) in enumerate(zip(patterns, peaks)):
                pattern.plot(ax=ax, color=cmap(i / len(patterns)),
                             offset=i * margin, alpha=0.8, if_label=False)
                ax.scatter(pattern.two_theta[peak[0]],
                           pattern.intensity[peak[0]] + i * margin,
                           color=cmap(i / len(patterns)), s=10, marker='x')
                ax.text(0.5, i * margin + margin * 0.3, f'No. {pattern.info.index}',
                        fontsize=10, color=cmap(i / len(patterns)))
        # Save the figure
        if save_path and fig:
            fig.savefig(save_path, dpi=dpi)

    def plot_with_icsd(self,
                       icsd: ICSD,
                       code: Union[int, list[int]],
                       **kwargs) -> None:
        """Plot the diffraction patterns with the ICSD data.

        Args:
            icsd (ICSD): The ICSD data.
            code (Union[int, list[int]]): The ICSD code to be plot.
            **kwargs: Other arguments for plotting.

        Returns:
            Axes: The axes of the plot.
        """
        if isinstance(code, int):
            code = [code]
        icsd_data = [icsd.data[icsd.index(icsd_code=c)[0]]
                     for c in code]
        # Plot the sample patterns
        fig, axs = plt.subplots(1 + len(icsd_data), 1, figsize=(6, 2 * (1 + len(icsd_data))))
        self.plot(ax=axs[0], **kwargs)
        # Get the xlim
        xlim = axs[0].get_xlim()
        # Plot the icsd patterns
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'tab10'))
        for i, c in enumerate(code):
            axs[i + 1] = icsd.plot(ax=axs[i + 1], color=cmap(i), icsd_code=c,
                                   angle_range=AngleRange(*xlim))
        fig.show()

    def plot_surf(
            self,
            index: Optional[list[int]] = None,
            dpi: int = 600,
            save_path: Optional[str] = None,
            ax: Optional[Axes3D] = None,
            amorphous_index: int = -1,
            **kwargs):
        """Plot the diffraction patterns in 3D surface.

        Args:
            index (Optional[list[int]], optional): Index to plot. Defaults to None.
            dpi (int, optional): Save fig dpi. Defaults to 600.
            save_path (Optional[str], optional): Save path. Defaults to None.
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
        peaks: list[tuple] = []
        # Process the patterns with kwargs given
        for i, pattern in enumerate(patterns):
            if amorphous_index >= 0:
                patterns[i].intensity = patterns[i].intensity - self.data[amorphous_index].intensity
            if 'lam' in kwargs:
                patterns[i] = patterns[i].subtract_baseline(lam=kwargs['lam'])
            if 'window' in kwargs or 'factor' in kwargs:
                patterns[i] = patterns[i].smooth(
                    window=kwargs.get('window', 101),
                    factor=kwargs.get('factor', 0.5))
            peaks.append(
                pattern.get_peak(mask=kwargs.get('mask', None),
                                 height=kwargs.get('height', -1),
                                 mask_height=kwargs.get('mask_height', -1)))
        # Plot the patterns
        fig = None
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'))
        # Convert all the intensity data into a surface
        x = np.array(patterns[0].two_theta)
        y = np.arange(len(patterns))
        X, Y = np.meshgrid(x, y)  # pylint: disable=invalid-name
        min_length = min(len(pattern.intensity) for pattern in patterns)
        Z = np.array([pattern.intensity[:min_length] for pattern in patterns])
        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap=cmap)
        # Save the figure
        if save_path and fig:
            fig.savefig(save_path, dpi=dpi)

    def classify(
            self,
            k_range: tuple[int, int] = (3, 10),
            full_run: bool = False,
            **kwargs) -> NDArray:
        """_summary_

        Args:
            k_range (tuple[int, int], optional): n_cluster range. Defaults to (3, 10).
            full_run (bool, optional): If set, run the full range of k. Defaults to False.
            **kwargs: Other arguments for subtract and smooth and verbose.

        Raises:
            ValueError: The clustering is not successful!

        Returns:
            NDArray: The clustering label.
        """
        # Import the KMeans and silhouette_score
        os.environ["OMP_NUM_THREADS"] = '2'
        from sklearn.cluster import KMeans  # pylint: disable=import-outside-toplevel
        from sklearn.metrics import silhouette_score  # pylint: disable=import-outside-toplevel
        # Copy the database
        db = self.copy()
        # Process the patterns with kwargs given
        if 'lam' in kwargs:
            db = db.subtract_baseline(lam=kwargs['lam'])
        if 'window' in kwargs or 'factor' in kwargs:
            db = db.smooth(window=kwargs.get('window', 101),
                           factor=kwargs.get('factor', 0.5))
        # Get the peak of the patterns
        peak_data = []
        for pattern in db.data:
            _, properties = pattern.get_peak(
                mask=kwargs.get('mask', None),
                height=kwargs.get('height', -1),
                mask_height=kwargs.get('mask_height', -1))
            peak_data.append(properties['peak_angles'])
        # The length of elements in peak_data are not the same
        # fill the missing elements with 0
        max_len = max(len(i) for i in peak_data)
        for i, data in enumerate(peak_data):
            peak_num = len(data)
            peak_data[i] = np.pad(data, (0, max_len - len(data)))
            # Add the number of peaks as the first column
            peak_data[i] = np.insert(peak_data[i], 0, peak_num)
        peak_data = np.array(peak_data)
        print(f'Peak numbers: {np.unique(peak_data[:, 0])}')
        # KMeans clustering
        print("K-means clustering ...")
        result = None
        last_result = None
        score = -1
        last_score = -1
        all_result = []
        all_score = []
        for k in range(*k_range):
            result = KMeans(
                n_clusters=k,
                random_state=0,
                verbose=kwargs.get('verbose', 1)).fit_predict(peak_data)
            score = silhouette_score(peak_data, result)
            all_score.append(score)
            all_result.append(result)
            if score < last_score and not full_run:
                result = last_result
                break
            last_result = result
            last_score = score
        if full_run:
            result = all_result[all_score.index(max(all_score))]
        if result is None:
            raise ValueError('The clustering is not successful!')
        print(f'Classification with {k - 1} clusters, silhouette score: {score:.2f}')
        return result
