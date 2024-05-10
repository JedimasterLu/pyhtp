# -*- coding: utf-8 -*-
"""
Filename: plotter.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
import ternary
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri
from matplotlib.animation import FuncAnimation
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from pyxrd.process import XrdProcess
from pyxrd.database import XrdDatabase


class XrdPlotter:
    ''' XrdPlotter is a class that plots the xrd data into animations or figures.
    '''
    def __init__(self, file_dir: list[str], save_dir: str='', title: str='', colormap: mpl.colors.ListedColormap=None):
        '''__init__ is a function that initializes the XrdPlotter class.

        Args:
            file_dir (list[str]): The file path to .xy files. Should be a list that contains only 2 str elements.
            save_dir (str, optional): The path to save plots. Defaults to ''.
            title (str, optional): The title of plots. Defaults to ''.
            colormap (matplotlib.colors.ListedColormap, optional): Set the color of plots. Defaults to batlowS from cmcrameri.

        Raises:
            ValueError: Two file directories are required for left and right xrd data!
            ValueError: Number of left and right xrd data do not match!
        '''
        if len(file_dir) != 2:
            raise ValueError('Two file directories are required for left and right xrd data!')
        for folder in file_dir:
            if not folder.endswith('/'):
                raise ValueError('File directory should end with /')
        if save_dir and not save_dir.endswith('/'):
            raise ValueError('Save directory should end with /')
        # Load the data
        self.file_dir = file_dir
        self.left_xy = os.listdir(file_dir[0])
        self.right_xy = os.listdir(file_dir[1])
        for filename in self.left_xy:
            if filename.split('.')[1] != 'xy':
                self.left_xy.remove(filename)
        for filename in self.right_xy:
            if filename.split('.')[1] != 'xy':
                self.right_xy.remove(filename)
        if len(self.left_xy) != len(self.right_xy):
            raise ValueError('Number of left and right xrd data do not match!')
        if save_dir:
            self.save_dir = save_dir
        else:
            self.save_dir = os.getcwd() + '/'
        self.title = title
        # Generate colormap
        if colormap is None:
            # pylint: disable=no-member
            self.colormap = cmcrameri.cm.batlowS
            # pylint: enable=no-member
        else:
            self.colormap = colormap

    def set_save_dir(self, save_dir: str) -> None:
        '''set_save_dir sets the save directory.

        Args:
            save_dir (str): The path to save plots.
        '''
        if not save_dir.endswith('/'):
            raise ValueError('Save directory should end with /')
        self.save_dir = save_dir

    def plot_animation(self, dpi: int=300, mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12, window: int=101, factor: float=0.5, lam: int=200) -> None:
        '''plot_animation plots multiple xrd spectrums into an animation.

        Args:
            dpi (int, optional): The dpi of the animation. Defaults to 300.
            mask (list[list[float, float]], optional): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...]. Defaults to None.
            height (float, optional): The height above which peaks could be detected. Defaults to 0.06.
            mask_height (float, optional): The height above which peaks could be detected in masked area. Defaults to 0.12.
            window (int, optional): The parameter of smooth. Defaults to 101.
            factor (float, optional): The parameter of smooth. Defaults to 0.5.
            lam (int, optional): The parameter of substract_baseline. Defaults to 200.
        '''
        # Set font
        plt.rc('font', family='Calibri')
        # Get all the smoothed data, raw data and peaks for the animation
        total_data_number = len(self.left_xy)
        all_peaks = []
        all_smoothed_data = []
        all_raw_data = []
        for index in range(total_data_number):
            left_path = self.file_dir[0] + self.left_xy[index]
            right_path = self.file_dir[1] + self.right_xy[index]
            original_data = XrdProcess(file_path=[left_path, right_path])
            substracted_data = original_data.substract_baseline(lam=lam)
            smoothed_data = substracted_data.smooth(window=window, factor=factor)
            _, peaks_index, _ = smoothed_data.peaks(mask=mask, height=height, mask_height=mask_height)
            all_peaks.append(peaks_index)
            all_smoothed_data.append(smoothed_data)
            all_raw_data.append(original_data)
        # Initialization
        fig, ax = plt.subplots()
        x_processed = all_smoothed_data[0].two_theta
        x_peaks = all_smoothed_data[0].two_theta[all_peaks[0]]
        x_baseline = all_raw_data[0].two_theta.copy()
        x_raw = all_raw_data[0].two_theta.copy()
        y_processed = all_smoothed_data[0].intensity
        y_peaks = all_smoothed_data[0].intensity[all_peaks[0]]
        y_baseline = all_raw_data[0].get_baseline()
        y_raw = all_raw_data[0].intensity
        line_processed = ax.plot(x_processed, y_processed, label='Processed', color=self.colormap(0))[0]
        line_peaks = ax.plot(x_peaks, y_peaks, 'x', label='Peaks', color=self.colormap(1))[0]
        line_raw = ax.plot(x_raw, y_raw, 'o', markersize=0.3, label='Raw data', color=self.colormap(2))[0]
        line_baseline = ax.plot(x_baseline, y_baseline, '--', label='Baseline', color=self.colormap(3))[0]
        plt.xlabel(r'2$\theta$')
        plt.ylabel('Intensity')
        plt.title(self.title + '-0')
        plt.legend(loc='upper right')

        def init():
            return line_processed, line_peaks, line_raw, line_baseline

        def update(num):
            line_processed.set_data(all_smoothed_data[num].two_theta, all_smoothed_data[num].intensity)
            line_peaks.set_data(all_smoothed_data[num].two_theta[all_peaks[num]], all_smoothed_data[num].intensity[all_peaks[num]])
            line_raw.set_data(all_raw_data[num].two_theta.copy(), all_raw_data[num].intensity)
            line_baseline.set_data(all_raw_data[num].two_theta.copy(), all_raw_data[num].get_baseline())
            plt.title(f'{self.title}-{num}')
            return line_processed, line_peaks, line_raw, line_baseline

        ani = FuncAnimation(fig, update, init_func=init, frames=total_data_number, interval=25, blit=False)

        if self.save_dir:
            ani.save(f"{self.save_dir}/{self.title}_animation.gif", fps=10, writer="pillow", dpi=dpi)
        else:
            ani.save(f"{self.title}_animation.gif", fps=10, writer="pillow", dpi=dpi)
        plt.close(fig)

    def plot_spectrum(self, index_to_plot: list[int]=None, plot_raw: bool=False, plot_peaks: bool=True, plot_unsmoothed: bool=False, plot_type: str='combine', ax: mpl.axes.Axes=None, mask: list[list[float, float]]=None, height: float=0.06, mask_height: float=0.12, window: int=101, factor: float=0.5, lam: int=200, if_show: bool=True, if_save: bool=False, save_path: str='', v_margin: float=0.2, dpi: int=600) -> mpl.axes.Axes:
        '''plot_spectrum plots the xrd spectrums for data points of the given index.

        Args:
            index_to_plot (list[int], optional): The index of data points to plot. Defaults to None. If None, plot all the data.
            plot_raw (bool, optional): If true, plot raw data. Defaults to False.
            plot_peaks (bool, optional): If true, mark peaks. Defaults to True.
            plot_unsmoothed (bool, optional): If true, plot substracted but unsmoothed data. Defaults to False.
            plot_type (str, optional): Set the showing method of multiple data lines. Defaults to 'combine'.
            ax (plt.axes._axes.Axes, optional): If given, plot in given axis. Defaults to None. If None, create a new figure.
            mask (list[list[float, float]], optional): Mask information in the form of [[mask1_left, mask1_right], [mask2_left, mask2_right] ...]. Defaults to None.
            height (float, optional): The height above which peaks could be detected. Defaults to 0.06.
            mask_height (float, optional): The height above which peaks could be detected in masked area. Defaults to 0.12.
            window (int, optional): The parameter of smooth. Defaults to 101.
            factor (float, optional): The parameter of smooth. Defaults to 0.5.
            lam (int, optional): The parameter of substract_baseline. Defaults to 200.
            if_show (bool, optional): If true, show the figure immediately. Defaults to True.
            if_save (bool, optional): If ture, save the figure to self.save_dir. Defaults to True.
            save_dir (str, optional): Set the unique save path for current figure. Defaults to ''. Then save to self.save_dir.
            v_margin (float, optional): Set the vertical spacing between lines in 'stack' mode. Defaults to 0.2.

        Raises:
            ValueError: Plot type not supported! Only combine and stack are allowed.
            ValueError: Index out of range! There are only {len(self.left_xy)} data points.
            ValueError: Cannot save the figure when ax is given!

        Returns:
            plt.axes._axes.Axes: The axis of the plot.
        '''
        if plot_type not in ['combine', 'stack']:
            raise ValueError('Plot type not supported! Only combine and stack are allowed.')
        if if_save and ax is not None:
            raise ValueError('Cannot save the figure when ax is given!')
        # Set font
        plt.rc('font', family='Calibri')
        # If list is empty, plot all the data
        if index_to_plot is None:
            index_to_plot = range(len(self.left_xy))
        if max(index_to_plot) >= len(self.left_xy):
            raise ValueError(f'Index out of range! There are only {len(self.left_xy)} data points.')
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        if plot_type == 'combine':
            for index_of_plot, index in enumerate(index_to_plot):
                left_path = self.file_dir[0] + self.left_xy[index]
                right_path = self.file_dir[1] + self.right_xy[index]
                # Process data
                original_data = XrdProcess([left_path, right_path])
                substracted_data = original_data.substract_baseline(lam=lam)
                smoothed_data = substracted_data.smooth(window=window, factor=factor)
                _, peaks_index, _ = smoothed_data.peaks(mask=mask, height=height, mask_height=mask_height)
                # Plot the data and the peak
                ax.plot(smoothed_data.two_theta, smoothed_data.intensity, label="Processed", color=self.colormap(index_of_plot))
                if plot_raw:
                    ax.plot(original_data.two_theta, original_data.intensity, 'o', markersize=0.3, label='Raw data', color=self.colormap(index_of_plot))
                if plot_unsmoothed:
                    ax.plot(substracted_data.two_theta, substracted_data.intensity, '+', markersize=0.3, label='Substracted', color=self.colormap(index_of_plot))
                if plot_peaks:
                    ax.plot(smoothed_data.two_theta[peaks_index], smoothed_data.intensity[peaks_index], "x", label="Peaks", color=self.colormap(index_of_plot))
            # Set the title and labels
            if len(index_to_plot) == 1:
                ax.set_title(f'{self.title}-{index_to_plot[0]}')
            else:
                ax.set_title(f'{self.title}')
            ax.set_xlim(smoothed_data.two_theta[0], smoothed_data.two_theta[-1])
            ax.set_xlabel(r'2$\theta$')
            ax.set_ylabel('Intensity')
        elif plot_type == 'stack':
            for index_of_plot, index in enumerate(index_to_plot):
                left_path = self.file_dir[0] + self.left_xy[index]
                right_path = self.file_dir[1] + self.right_xy[index]
                # Process data
                original_data = XrdProcess([left_path, right_path])
                substracted_data = original_data.substract_baseline(lam=lam)
                smoothed_data = substracted_data.smooth(window=window, factor=factor)
                _, peaks_index, _ = smoothed_data.peaks(mask=mask, height=height, mask_height=mask_height)
                ax.plot(smoothed_data.two_theta, smoothed_data.intensity + index_of_plot * v_margin, color=self.colormap(index_of_plot))
                if plot_peaks:
                    ax.plot(smoothed_data.two_theta[peaks_index], smoothed_data.intensity[peaks_index] + index_of_plot * v_margin, "x", label="Peaks", color=self.colormap(index_of_plot))
                # Plot a text containing the index of the data just above the line
                ax.text(smoothed_data.two_theta[-1] - 5, 0 + index_of_plot * v_margin + v_margin / 5, f'point {index}', fontsize=10, color=self.colormap(index_of_plot))
            # Set the title and labels
            if len(index_to_plot) == 1:
                ax.set_title(f'{self.title}-{index_to_plot[0]}')
            else:
                ax.set_title(f'{self.title}')
            ax.set_xlim(smoothed_data.two_theta[0], smoothed_data.two_theta[-1])
            ax.set_ylim(0, len(index_to_plot) * v_margin)
            ax.set_xlabel(r'2$\theta$')
            ax.set_ylabel('Intensity')
        if if_save:
            if save_path:
                fig.savefig(save_path, dpi=dpi)
            else:
                fig.savefig(f'{self.save_dir}/{self.title}_spectrum', dpi=dpi)
        if if_show:
            plt.show()
        else:
            if fig:
                plt.close(fig)
        return ax

    def plot_spectrum_with_ref(self, pattern: list[dict[str, any]]=None, database_dir: str='', **kwargs) -> mpl.axes.Axes:
        '''plot_spectrum_with_ref plots the xrd spectrums for data points of the given index with reference patterns.

        Args:
            pattern (list[dict[str, any]]): The list of reference patterns. Each element should be a dict containing 'two_theta' and 'intensity'.
            **kwargs: The arguments passed to plot_spectrum.

        Returns:
            mpl.axes.Axes: The axis of the plot.

        Raises:
            ValueError: No reference pattern or database directory provided! At least one of them should be provided.
            ValueError: Cannot save the figure when ax is given!
        '''
        if pattern is None and not database_dir:
            raise ValueError('No reference pattern or database directory provided!')
        if 'if_save' in kwargs and 'ax' in kwargs and kwargs['if_save'] and kwargs['ax'] is not None:
            raise ValueError('Cannot save the figure when ax is given!')
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            fig = None
        else:
            fig, ax = plt.subplots()
        if 'if_show' in kwargs:
            if_show = kwargs.pop('if_show')
        else:
            if_show = True
        if 'if_save' in kwargs:
            if_save = kwargs.pop('if_save')
        else:
            if_save = False
        ax = self.plot_spectrum(**kwargs, ax=ax, if_show=False, if_save=False)
        if pattern is None:
            database = XrdDatabase(file_dir=database_dir)
            pattern, structure = database.process(if_save=False)
        # Get the start index of colormap
        if 'index_to_plot' not in kwargs:
            index_to_plot = range(len(self.left_xy))
        else:
            index_to_plot = kwargs['index_to_plot']
        number_of_lines = len(index_to_plot)
        v_margin = kwargs.get('v_margin', 0.2)
        for index, (ref_pattern, ref_structure) in enumerate(zip(pattern, structure)):
            spg_symbol, spg_number = ref_structure.get_space_group_info()
            label = f'{ref_structure.formula} ({spg_symbol}, {spg_number})'
            # Plot the reference pattern as vertical lines from top to bottom, across the whole ylim, x is the two_theta
            ax.vlines(ref_pattern['two_theta'], 0, number_of_lines * v_margin, linestyle='--', label=label, color=self.colormap(index + number_of_lines), linewidth=1)
        # Plot legend right outside the plot and set the margin of the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if 'dpi' not in kwargs:
            kwargs['dpi'] = 600
        if if_save:
            if 'save_path' in kwargs:
                fig.savefig(kwargs['save_path'], dpi=kwargs['dpi'])
            else:
                fig.savefig(f'{self.save_dir}/{self.title}_spectrum_with_ref', dpi=kwargs['dpi'])
        if if_show:
            plt.show()
        else:
            if fig:
                plt.close(fig)
        return ax


def plot_ternary_diagram(phase_type: list[str],
                         phase_index: list[int],
                         labels: list[str],
                         title: str,
                         color: dict[str, str],
                         rotation: dict[str, float]=None,
                         ax: mpl.axes.Axes=None,
                         if_save: bool=False,
                         if_show: bool=True,
                         if_legend: bool=True) -> None:
    '''plot_ternary_diagram plots a ternary diagram with given width and phase information.

    Args:
        width (int): The width of the diagram.
        phase_type (list[str]): All the name of the phases. Defaults to None.
        phase_index (list[int]): Define the phase of each point. Defaults to None.
        labels (list[str]): Labels of the three axis. Defaults to None.
        title (str): Title of the plot. Defaults to None.
        ax (mpl.axes.Axes, optional): If need to plot in existing ax, define this parameter. Defaults to None.
        color (dict[str, str]): The color of certain phase. Defaults to None.
    '''
    # Set font and ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = None
    plt.rcParams['font.family'] = 'Times New Roman'
    # Generate the points
    width = _get_width(phase_index)
    points = _get_points_position(width)
    # Plot the ternary diagram
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)
    tax.set_title(title, fontsize=15, fontweight='bold')
    # Plot the scatter of each phase
    for current_index in range(max(phase_index) + 1):
        current_type = phase_type[current_index]
        current_points = [points[i] for i in range(len(points)) if current_index == phase_index[i]]
        if ' + ' in current_type:
            temp_current_type = current_type
            current_type = current_type.split(' + ')
            if rotation is not None and temp_current_type in rotation:
                _ternary_scatter(tax,
                                 current_points,
                                 color[current_type[0]],
                                 color[current_type[1]],
                                 8,
                                 rotation[temp_current_type])
            else:
                _ternary_scatter(tax,
                                 current_points,
                                 color[current_type[0]],
                                 color[current_type[1]],
                                 8)
        else:
            tax.scatter(current_points, color=color[current_type], s=40)
    # Set legends
    if if_legend:
        legend_elements = []
        for current_index in range(max(phase_index) + 1):
            current_type = phase_type[current_index]
            if ' + ' in current_type:
                current_type = current_type.split(' + ')
                # If legend doesn't exist, add it
                if current_type[0] not in [element.get_label() for element in legend_elements]:
                    legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', color='w', label=current_type[0], markerfacecolor=color[current_type[0]], markersize=10))
                if current_type[1] not in [element.get_label() for element in legend_elements]:
                    legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', color='w', label=current_type[1], markerfacecolor=color[current_type[1]], markersize=10))
            else:
                if current_type not in [element.get_label() for element in legend_elements]:
                    legend_elements.append(mpl.lines.Line2D([0], [0], marker='o', color='w', label=current_type, markerfacecolor=color[current_type], markersize=10))
        ax.legend(handles=legend_elements, loc='upper right')
    # Set the axis and labels
    tax.ticks(axis='lbr', linewidth=1, multiple=20)
    tax.bottom_axis_label(labels[0], fontsize=15)
    tax.right_axis_label(labels[1], fontsize=15)
    tax.left_axis_label(labels[2], fontsize=15)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    if if_show:
        plt.show()
        if fig:
            plt.close(fig)
    if if_save:
        # pylint: disable=W0212
        tax._redraw_labels()
        # pylint: enable=W0212
        tax.savefig(f'{title}.png', dpi=600)


def _get_points_position(width: int) -> list[tuple[int, int, int]]:
    '''_get_points_position returns the position of points in ternary diagram.

    Args:
        points_number (int): The number of points in the diagram.

    Returns:
        list[tuple[int, int, int]]: The position of points in the diagram. In snake form from bottom to top.
    '''
    # Generate the points, each point position is defined by a tuple contaning 3 int
    points = []
    # Define the margin between points by width
    margin = 100 / (width - 1)
    # Generate the points in snake form from bottom to top
    for row in range(width):
        # Find the middle point of that row
        if (width - row) % 2 == 0:
            middle = int((width - row) / 2) - 1
        else:
            middle = (width - row) // 2
        y = row * margin
        middle_value = 50 - y / 2
        # if the row is even, left to right; if odd, right to left
        if row % 2 == 0:
            for column in range(width - row):
                if (width - row) % 2 != 0:
                    x = middle_value - (middle - column) * margin
                else:
                    x = middle_value - (middle - column) * margin - margin / 2
                z = 100 - x - y
                points.append((x, y, z))
        else:
            for column in range(width - row - 1, -1, -1):
                if (width - row) % 2 != 0:
                    x = middle_value - (middle - column) * margin
                else:
                    if column < middle:
                        x = middle_value - (middle - column) * margin + margin / 2
                    else:
                        x = middle_value - (middle - column) * margin - margin / 2
                z = 100 - x - y
                points.append((x, y, z))
    return points


def _ternary_scatter(tax,
                     points: list[tuple[int, int, int]],
                     left_color: str,
                     right_color: str,
                     marker_size: int,
                     rotation: float=0) -> None:
    '''_ternary_scatter _summary_

    Args:
        tax (_type_): _description_
        points (list[tuple[int, int, int]]): _description_
        left_color (str): _description_
        right_color (str): _description_
        marker_size (int): _description_
    '''
    t = Affine2D().rotate_deg(rotation)
    filled_marker_style = {'marker': MarkerStyle('o', 'left', t),
                           'markersize': marker_size,
                           'markerfacecolor': left_color,
                           'markerfacecoloralt': right_color,
                           'markeredgecolor': '#F7F7F7',
                           'markeredgewidth': 0,
                           }
    tax.plot(points, linewidth=0, **filled_marker_style)


def rotate_phase_index(phase_index: list[int]) -> list[int]:
    '''_rotate_phase_index rotate the whole ternary diagram by 60 degrees counter-clockwise.

    Args:
        width (int): The width of the diagram.
        phase_index (list[int]): Define the phase of each point.

    Returns:
        list[int]: _description_
    '''
    # Create mapping relationship of rotation
    mapping = []
    width = _get_width(phase_index)
    for row in range(width):
        current_width = width - row
        # Extend the index of the right row into mapping
        for column in range(current_width):
            mapping.append(current_width - 1 + column * (width - 1 + width - column) // 2)
    # Map the phase index
    phase_index = _snake_to_serial(width, phase_index.copy())
    rotated_phase_index = [phase_index[mapping[i]] for i in range(len(phase_index))]
    rotated_phase_index = [rotated_phase_index[mapping[i]] for i in range(len(phase_index))]
    rotated_phase_index = _snake_to_serial(width, rotated_phase_index)
    return rotated_phase_index


def _snake_to_serial(width: int, phase_index: list[int]) -> list[int]:
    '''snake_to_serial _summary_

    Args:
        width (int): _description_
        phase_index (list[int]): _description_

    Returns:
        list[int]: _description_
    '''
    serial_phase_index = phase_index.copy()
    # Reverse the index of even rows
    for row in range(width):
        if row % 2 != 0:
            start_index = (row + 1) * (width + width - row) // 2 - 1
            end_index = start_index - width + row + 1
            for column in range(end_index, start_index + 1):
                serial_phase_index[column] = phase_index[start_index - column + end_index]
    return serial_phase_index


def _get_width(index: list[int]) -> int:
    '''_get_width return the width of the ternary diagram.

    Args:
        index (list[int]): The index of the points.

    Returns:
        int: The width of the diagram.

    Raises:
        ValueError: The index length is not a triangle number!
    '''
    # Set the initial value as the approximate value of the width by solving width * (width + 1) / 2 == length
    length = len(index)
    width = (-1 + (1 + 8 * length) ** 0.5) / 2
    width = int(width) - 2
    # check if there exist a width such that width * (width + 1) / 2 == length, if not, raise error
    while width * (width + 1) / 2 < length:
        width += 1
    if width * (width + 1) / 2 != length:
        raise ValueError('The index length is not a triangle number!')
    return width
