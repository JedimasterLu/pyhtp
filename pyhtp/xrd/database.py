# -*- coding: utf-8 -*-
"""
A class that contains diffraction patterns of a high-throughput sample.
"""
import os
from typing import Union, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .infotuple import SampleInfo, PatternInfo, AngleRange
from .pattern import XrdPattern
from .icsd import ICSD


class XrdDatabase:
    """A class that contains diffraction patterns of a high-throughput sample.
    """
    def __init__(self,
                 file_dir: Union[str, list[str]],
                 info: SampleInfo):
        """Create a instance of XrdDatabase.

        Args:
            file_dir (Union[str, list[str]]): The directory of the database. Input a list if there are multiple scan angles.
        """
        # Check the input
        if isinstance(file_dir, str) and isinstance(info.angle_range, list):
            raise ValueError('The length of file_path and scan_angle should be the same!')
        if isinstance(file_dir, list) and not isinstance(info.angle_range, list):
            raise ValueError('The length of file_path and scan_angle should be the same!')
        if isinstance(file_dir, list) and isinstance(info.angle_range, list):
            if len(file_dir) != len(info.angle_range):
                raise ValueError('The length of file_path and scan_angle should be the same!')
        # Assign the attributes
        if isinstance(file_dir, str):
            file_dir = [file_dir]
        self.data = self._read_data(file_dir, info)
        # Change the angle range of self.info to a tuple if it is a list
        if isinstance(info.angle_range, list):
            self.info = SampleInfo(
                name=info.name, element=info.element,
                temperature=info.temperature,
                angle_range=AngleRange(info.angle_range[0].left, info.angle_range[-1].right),
                description=info.description)
        else:
            self.info = info

    def _read_data(self, file_dir: list[str], info: SampleInfo) -> list[XrdPattern]:
        """Read the data from the file directory.

        Args:
            file_dir (list[str]): The directory of the database. Input a list if there are multiple scan angles.

        Raises:
            FileNotFoundError: The directory is not found!
            ValueError: The number of files in the directories are not the same!

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
                tuple[np.ndarray, np.ndarray]: The (two_theta, intensity) of the xrd data.
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

    def plot(self,
             index: Optional[list[int]]=None,
             style: Literal['combine', 'stack']='combine',
             dpi: int=600,
             save_path: Optional[str]=None,
             ax: Optional[Axes]=None,
             amorphous_index: int=-1,
             **kwargs) -> Axes:
        """Plot the diffraction patterns.

        Args:
            index (Optional[list[int]], optional): The index of patterns to plot. Defaults to None.
            style (Literal['combine', 'stack'], optional): Mode. Defaults to 'combine'.
            dpi (int, optional): Output dpi. Defaults to 300.
            save_path (Optional[str], optional): If set, fig will be saved. Defaults to None.
            ax (Optional[Axes], optional): The axes to plot. Defaults to None.
            amorphous_index (int, optional): The index of a reference amorphous pattern. If set, intensities will be subtracted by the reference. Defaults to -1.
            **kwargs: Other arguments for post-processing and plotting.

        Returns:
            Axes: The axes of the plot.
        """
        # Process the pattern to be plot
        if index is None:
            index = list(range(len(self.data)))
        # Select the patterns to plot
        patterns = [self.data[i] for i in index]
        peak: list[tuple] = []
        # Process the patterns with kwargs given
        for pattern in patterns:
            if amorphous_index >= 0:
                pattern.intensity = pattern.intensity - self.data[amorphous_index].intensity
            if 'lam' in kwargs:
                pattern = pattern.subtract_baseline(lam=kwargs['lam'])
            if 'window' in kwargs or 'factor' in kwargs:
                pattern = pattern.smooth(window=kwargs.get('window', 101),
                                         factor=kwargs.get('factor', 0.5))
            peak.append(
                pattern.get_peak(mask=kwargs.get('mask', None),
                                 height=kwargs.get('height', -1),
                                 mask_height=kwargs.get('mask_height', -1)))
        # Plot the patterns
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        if style == 'combine':
            for i, pattern in enumerate(patterns):
                cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'))
                pattern.plot(ax=ax, color=cmap(i / len(patterns)))
        if style == 'stack':
            # Get the margin from the maximum intensity variation of the patterns
            margin = max(pattern.intensity.max() - pattern.intensity.min() for pattern in patterns)
            for i, pattern in enumerate(patterns):
                cmap = plt.cm.get_cmap(kwargs.get('cmap', 'viridis'))
                pattern.plot(ax=ax, color=cmap(i / len(patterns)), offset=i * margin, alpha=0.8)
                ax.scatter(pattern.two_theta[peak[i]],
                           pattern.intensity[peak[i]] + i * margin,
                           color=cmap(i / len(patterns)), s=10, marker='x')
                ax.text(0.5, i * margin + margin * 0.3, f'No. {pattern.info.index}',
                        fontsize=10, color=cmap(i / len(patterns)))
        # Save the figure
        if save_path and fig:
            fig.savefig(save_path, dpi=dpi)
        return ax

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
        fig, axs = plt.subplots(1 + len(icsd_data), 1, figsize=(4, 1 * (1 + len(icsd_data))))
        self.plot(ax=axs[0], **kwargs)
        # Get the xlim
        xlim = axs[0].get_xlim()
        # Plot the icsd patterns
        cmap = plt.cm.get_cmap(kwargs.get('cmap', 'tab10'))
        for i, c in enumerate(code):
            icsd.plot(ax=axs[i + 1], color=cmap(i), icsd_code=c, angle_range=AngleRange(*xlim))
        fig.show()
