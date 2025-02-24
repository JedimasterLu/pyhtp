# -*- coding: utf-8 -*-
"""
This module provides a class to process reference phase data from .cif files.
- Author: Junyuan Lu
- E-mail: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
import re
import pickle
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from ..typing import AngleRange, CIFData, MillerIndice, LatticeParam


class CIFDatabase:
    """This class can process reference phase data from .cif files.

    Please put all the cif files into a directory to construct the database.
    The class has the following methods
    - process: generate list[CIFData] from .cif files.
    - index: get the index of the data in database by several conditions.
    - plot: plot the pattern with vertical lines.

    Note:
        The tested cif files of this class are currently all from ICSD.
        Errors may occur if the cif files are not from ICSD.

    Attributes:
        data (list[CIFData]): The reference phase data from all the .cif files in given directory.
    """
    def __init__(
            self,
            file_dir: str,
            **kwargs):
        """Initialize the reference phase database from a folder with .cif files.

        Args:
            file_dir (str): The directory of the .cif files.
                If non cif files are in the directory, they will be ignored.
            **kwargs: The keyword arguments for cif file processing,
                which is automatically performed when the class is initialized.
                - if_save (bool, optional): This parameter is passed into process function.
                    If true, a pkl file containing the class will be saved to the file_dir
                    to speed up the initialization next time. Defaults to True.
                - wavelength (str | float, optional): The wavelength of the X-ray. Defaults to 'CuKa'.
                    Please refer to the pymatgen documentation for the available wavelength strings.
        """
        self._file_dir = file_dir
        self.data: list[CIFData] = []
        # Load the data from the pickle file if it exists
        if os.path.exists(os.path.join(self._file_dir, 'icsd.pkl')):
            with open(os.path.join(self._file_dir, 'icsd.pkl'), 'rb') as f:
                self.data = pickle.load(f)
        else:
            self._process(
                if_save=kwargs.get('if_save', True),
                wavelength=kwargs.get('wavelength', 'CuKa'))

    def _process(
            self,
            wavelength: str | float = 'CuKa',
            if_save: bool = True):
        """Process the cif files in the file_dir and generate the data.
        """
        cif_files = os.listdir(self._file_dir)
        data: list[CIFData] = []
        for file_name in cif_files:
            if file_name.split('.')[-1] != 'cif':
                continue
            file_path = os.path.join(self._file_dir, file_name)
            try:
                parser = CifParser(file_path)
                cif_data = parser.as_dict()
                # Because the initial key is a long string, we need to get the first key.
                cif_data = cif_data[str(list(cif_data.keys())[0])]
                # Remove the spaces in cif_data['_space_group_name_H-M_alt']
                cif_data['_space_group_name_H-M_alt'] = cif_data['_space_group_name_H-M_alt'].replace(' ', '')
                structure = Structure.from_file(file_path, primitive=False, merge_tol=0.01)
                xrd = XRDCalculator(wavelength=wavelength)  # type: ignore
                pattern = xrd.get_pattern(structure)
            except ValueError as e:
                print(f'Error in processing {file_name}: {e}')
                continue
            icsd_data = {
                'name': file_name,
                'two_theta': pattern.x,
                'intensity': pattern.y,
                'hkl': [MillerIndice(hkls[0]['hkl'][0], hkls[0]['hkl'][1], hkls[0]['hkl'][-1])
                        for hkls in pattern.hkls],
                'space_group': cif_data['_space_group_name_H-M_alt'],
                'space_group_number': int(cif_data['_space_group_IT_number']),
                'formula': cif_data['_chemical_formula_structural'],
                'lattice_parameter': LatticeParam(
                    a=cif_data['_cell_length_a'],
                    b=cif_data['_cell_length_b'],
                    c=cif_data['_cell_length_c'],
                    alpha=cif_data['_cell_angle_alpha'],
                    beta=cif_data['_cell_angle_beta'],
                    gamma=cif_data['_cell_angle_gamma']),
                'icsd_code': int(cif_data['_database_code_ICSD']),
                'structure': structure
            }
            # Convert icsd_data to IcsdData
            icsd_data = CIFData(**icsd_data)
            data.append(icsd_data)
        self.data = data
        if if_save:
            save_path = os.path.join(self._file_dir, 'icsd.pkl')
            pickle.dump(self.data, open(save_path, 'wb'))

    def index(
            self,
            cif_name: str | list[str] | None = None,
            icsd_code: int | list[int] | None = None,
            space_group: str | list[str] | None = None,
            space_group_number: int | list[int] | None = None,
            element: str | list[str] | None = None,
            peak_two_theta: float | tuple[float, float] | list[float | tuple[float, float]] | None = None,
            two_theta_range: float = 0.2,
            mode: Literal['and', 'or', 'strict'] = 'or') -> list[int]:
        """Get the index of reference phase data in database by some conditions.

        Note:
            - The parameters have certain order of priority. The selection policy among different \
            parameters is always ``and``. For example, if ``cif_name`` and ``icsd_code`` are both given, \
            the data that meet both conditions will be appended to the result.

            - The ``mode`` is for ``peak_two_theta`` and ``element`` conditions. DO NOT set ``mode`` to \
            ``and`` or ``strict`` when using ``cif_name``, ``icsd_code``, ``space_group`` or \
            ``space_group_number`` conditions. An error will be raised if you do so.

        Args:
            cif_name (str | list[str] | None, optional): The name of the cif file without .cif. Defaults to None.
            icsd_code (int | list[int] | None, optional): The ICSD code of the data. Defaults to None.
            space_group (str | list[str] | None, optional): The space group of the data. Defaults to None.
            space_group_number (int | list[int] | None, optional): The space group number of the data. Defaults to None.
            element (str | list[str] | None, optional): The element in the formula of the data. Defaults to None.
            peak_two_theta (float | tuple[float, float] | list[float | tuple[float, float]] | None, optional):
                The required two_theta of the data. Defaults to None.
            two_theta_range (float, optional): The range of two_theta for float input of peak_two_theta.
                The range is (float_two_theta - range / 2, float_two_theta + range / 2). Defaults to 0.2.
            mode (Literal['and', 'or', 'strict'], optional): The mode of the search. Defaults to 'or'.
                The mode is for element and peak_two_theta conditions. For the other 4 conditions,
                since each reference phase data has only one cif_name, icsd_code, spcae_group and space_group_number,
                setting mode to ``and`` or ``strict`` is meaningless, and an error will be raise.
                    - ``or``: As long as one condition is met, the data will be appended to the result.
                    - ``and``: Only when all conditions are met, the data will be appended to the result.
                    - ``strict``: The data must perfectly match all conditions to be appended to the result.

                For example, if the input elements = ['Ge', 'Se']
                    - ``or``: If Ge or Se is in the data.formula, the data will be appended to the result.
                    - ``and``: If both Ge and Se are in the data.formula, the data will be appended to the result.
                    - ``strict``: If the data.formula is exactly 'Gex Sey', the data will be appended to the result.

        Returns:
            list[int]: The index of the reference phase data in the self.data list.

        Raises:
            ValueError: If no reference phase data found based on the conditions.
        """
        # Process the input parameters
        if isinstance(cif_name, str):
            cif_name = [cif_name]
        if isinstance(icsd_code, int):
            icsd_code = [icsd_code]
        if isinstance(space_group, str):
            space_group = [space_group]
        if isinstance(space_group_number, int):
            space_group_number = [space_group_number]
        if isinstance(element, str):
            element = [element]
        if isinstance(peak_two_theta, (float, tuple)):
            peak_two_theta = [peak_two_theta]
            for i, two_theta in enumerate(peak_two_theta):
                if isinstance(two_theta, float):
                    peak_two_theta[i] = (two_theta - two_theta_range / 2, two_theta + two_theta_range / 2)
        # Check the mode
        if (cif_name is not None
                or icsd_code is not None
                or space_group is not None
                or space_group_number is not None) and mode != 'or':
            raise ValueError('Each icsd data has only one cif_name, icsd_code, spcae_group and space_group_number. \
                              Setting mode to "and" or "strict" is meaningless. \
                              If you want to find data with element or peak_two_theta conditions, please use multiple \
                              index functions and combine the results.')
        # Find the index of the data
        # If the data meet the conditions, append the index to the result_index
        result_index: list[int] = []
        for i, data in enumerate(self.data):
            if cif_name:
                if data.name not in cif_name:
                    continue
            if icsd_code:
                if data.icsd_code not in icsd_code:
                    continue
            if space_group:
                if data.space_group not in space_group:
                    continue
            if space_group_number:
                if data.space_group_number not in space_group_number:
                    continue
            if element:
                # Process the formula to get icsd element list
                # e.g. 'Ge0.15 Se0.11 H0.1' -> ['Ge', 'Se', 'H']
                # e.g. '123123' -> [] (If no element found, raise error)
                icsd_element: list[str] = re.findall(r'[A-Z][a-z]*', data.formula)
                if not icsd_element:
                    raise ValueError(f"No element found in data[{i}]'s formula: {data.formula}!")
                if mode == 'or':
                    if not set(icsd_element).issubset(set(element)):
                        continue
                elif mode == 'and':
                    if not set(element).issubset(set(icsd_element)):
                        continue
                elif mode == 'strict':
                    if set(icsd_element) != set(element):
                        continue
            if peak_two_theta:
                assert (isinstance(peak_two_theta, list)
                        and all(isinstance(two_theta, tuple)
                                for two_theta in peak_two_theta))
                if mode == 'or':
                    # As long as one peaks of the data lies in any given two_theta range, else continue.
                    continue_flag = True
                    for two_theta in peak_two_theta:
                        assert isinstance(two_theta, tuple)
                        if any(two_theta[0] <= angle < two_theta[1] for angle in data.two_theta):
                            continue_flag = False
                            break
                    if continue_flag:
                        continue
                elif mode == 'and':
                    # Only all peaks of the data lies in all given two_theta range, else continue.
                    continue_flag = False
                    for two_theta in peak_two_theta:
                        assert isinstance(two_theta, tuple)
                        if not any(two_theta[0] <= angle < two_theta[1] for angle in data.two_theta):
                            continue_flag = True
                            break
                    if continue_flag:
                        continue
                elif mode == 'strict':
                    # The two_theta range perfectly match all peaks of the data. Else continue.
                    if len(peak_two_theta) != len(data.two_theta):
                        continue
                    continue_flag = False
                    for two_theta in peak_two_theta:
                        assert isinstance(two_theta, tuple)
                        if not any(two_theta[0] <= angle < two_theta[1] for angle in data.two_theta):
                            continue_flag = True
                            break
                    if continue_flag:
                        continue
            result_index.append(i)
        if not result_index:
            raise ValueError('No reference phase data found! Please check the conditions.')
        return result_index

    def plot(
            self,
            two_theta_range: AngleRange = AngleRange(left=0, right=90),
            ax: Axes | None = None,
            if_show: bool = True,
            cmap: str = '',
            color: str | tuple[float, float, float, float] = 'tab:blue',
            **kwargs) -> None:
        """Plot the diffraction pattern of the reference phases.

        The peaks are shown as vertical lines. Miller indices are displayed above the peaks.

        Args:
            two_theta_range (AngleRange, optional): The plot two theta limit.
                Peaks out of the range will not be shown. Defaults to (0, 90).
            ax (Axes | None, optional): The matplotlib axes of the plot. If provided,
                the pattern will be plotted inside the ax. Try to use it for second
                development, such as plot reference in subplots or change styles.
                If none, a new figure and axes will be created. Defaults to None.
            if_show (bool, optional): Whether to show the plot immediately. Defaults to True.
            cmap (str, optional): The colormap of the plot. If more than one
                reference phases is selected, use the colormap. If index_to_plot is less than 10,
                default to 'tab10', if 20, default to 'tab20', else default to 'viridis'.
                Defaults to ''.
            color (Union[str, tuple[float, float, float, float]], optional):
                For single reference phase plot, directly set the color. Defaults to 'tab:blue'.
            **kwargs: The keyword arguments for index function and plot. If no keyword arguments,
                all the reference phases will be plotted. The keyword arguments are as follows.
                Please refer to the docstring of CIFDatabase.index() for more details.

                For index function
                    - ``cif_name``: The name of the cif file without .cif. Defaults to None.
                    - ``icsd_code``: The ICSD code of the data. Defaults to None.
                    - ``space_group``: The space group of the data. Defaults to None.
                    - ``space_group_number``: The space group number of the data. Defaults to None.
                    - ``element``: The element in the formula of the data. Defaults to None.
                    - ``peak_two_theta``: The required two_theta of the data. Defaults to None.
                    - ``two_theta_range``: The range of two_theta for float input of peak_two_theta.
                            The range is ``(float_two_theta - range / 2, float_two_theta + range / 2)``.
                            Defaults to 0.2.
                    - ``mode``: The mode of the search. Defaults to ``or``.

                For plot
                    - ``linewidth``: The width of the vertical lines. Defaults to 2.
                    - ``fontsize``: The fontsize of the Miller indices. Defaults to 7.
                    - ``rotation``: The rotation of the Miller indices. Defaults to 90.

        Raises:
            ValueError: If no reference phase data found based on the conditions.
        """

        if not kwargs:
            index_to_plot = list(range(len(self.data)))
        else:
            index_to_plot = self.index(**kwargs)
        if two_theta_range is None:
            two_theta_range = AngleRange(left=0, right=90)
        if ax is None:
            _, ax = plt.subplots()
        assert isinstance(ax, Axes)

        for i, index in enumerate(index_to_plot):
            data = self.data[index]
            two_theta = [angle for angle in data.two_theta
                         if two_theta_range.left <= angle <= two_theta_range.right]
            intensity = [intensity for angle, intensity
                         in zip(data.two_theta, data.intensity)
                         if two_theta_range.left <= angle <= two_theta_range.right]
            label = f'{data.formula}-{data.space_group}-{data.icsd_code}'
            linewidth = kwargs.get('linewidth', 2)
            if len(index_to_plot) > 1:
                if cmap:
                    colormap = plt.get_cmap(cmap)
                else:
                    if len(index_to_plot) <= 10:
                        colormap = plt.get_cmap('tab10')
                    elif len(index_to_plot) <= 20:
                        colormap = plt.get_cmap('tab20')
                    else:
                        colormap = plt.get_cmap('viridis')
                colors = colormap(range(len(index_to_plot)))

                # If there are multiple data, we use the colormap.
                ax.vlines(
                    two_theta, 0, intensity, color=colors[i],
                    linewidth=linewidth, label=label)
                ax.legend()
            else:
                # If there is only one data, we use the color parameter, with text.
                ax.vlines(
                    two_theta, 0, intensity, color=color, linewidth=linewidth, label=label)
                ax.text(
                    0.1, 0.8, label, transform=ax.transAxes, color=color)
                # Also add text indicate the miller indexs on each vline.
                for j, hkl in enumerate(data.hkl):
                    if two_theta_range.left <= data.two_theta[j] <= two_theta_range.right:
                        ax.text(
                            data.two_theta[j], data.intensity[j],
                            f'({hkl.h} {hkl.k} {hkl.l})', fontsize=kwargs.get('fontsize', 7),
                            rotation=kwargs.get('rotation', 90), color=color)

        ax.set_xlim(two_theta_range.left, two_theta_range.right)

        if if_show:
            plt.show()
