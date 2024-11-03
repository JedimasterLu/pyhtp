# -*- coding: utf-8 -*-
"""
Filename: database.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
import pickle
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from .infotuple import AngleRange, IcsdData, MillerIndice, Latticeabc, Latticeangles


class ICSD:
    ''' A class that generated pattern database and structure database from .cif files.
    '''
    def __init__(self, file_dir: str):
        '''__init__ create a instance of XrdDatabase.

        Args:
            file_dir (str): The directory of .cif files.
        '''
        self._file_dir = file_dir
        self.data: list[IcsdData] = []
        # Load the data from the pickle file if it exists
        if os.path.exists('icsd.pkl'):
            self.data = pickle.load(open('icsd.pkl', 'rb'))
        else:
            self.process(if_save=True)

    def process(self, if_save: bool=True):
        '''process generate pattern database and structure database from .cif files.
        '''
        cif_files = os.listdir(self._file_dir)
        data: list[IcsdData] = []
        for file_name in cif_files:
            if file_name.split('.')[-1] != 'cif':
                continue
            file_path = self._file_dir + file_name
            parser = CifParser(file_path)
            cif_data = parser.as_dict()
            # Because the initial key is a long string, we need to get the first key.
            cif_data = cif_data[str(list(cif_data.keys())[0])]
            # Remove the spaces in cif_data['_space_group_name_H-M_alt']
            cif_data['_space_group_name_H-M_alt'] = cif_data['_space_group_name_H-M_alt'].replace(' ', '')
            structure = Structure.from_file(file_path, primitive=False, merge_tol=0.01)
            xrd = XRDCalculator(wavelength='CuKa')
            pattern = xrd.get_pattern(structure)
            # pattern = pattern.as_dict()
            icsd_data = {
                'name': file_name,
                'two_theta': pattern.x,
                'intensity': pattern.y,
                'hkl': [MillerIndice(*hkls['hkl'])
                        for hkls in pattern.hkls],
                'space_group': cif_data['_space_group_name_H-M_alt'],
                'space_group_number': int(cif_data['_space_group_IT_number']),
                'formula': cif_data['_chemical_formula_structural'],
                'lattice_abc': Latticeabc(cif_data['_cell_length_a'],
                                          cif_data['_cell_length_b'],
                                          cif_data['_cell_length_c']),
                'lattice_angles': Latticeangles(cif_data['_cell_angle_alpha'],
                                                cif_data['_cell_angle_beta'],
                                                cif_data['_cell_angle_gamma']),
                'icsd_code': int(cif_data['_database_code_ICSD']),
                'structure': structure
            }
            # Convert icsd_data to IcsdData
            icsd_data = IcsdData(**icsd_data)
            data.append(icsd_data)
        self.data = data
        if if_save:
            pickle.dump(self.data, open('icsd.pkl', 'wb'))

    def index(self,
              file_name: Optional[str]=None,
              icsd_code: Optional[int]=None,
              space_group: Optional[str]=None,
              space_group_number: Optional[int]=None,
              element: Optional[list[str]]=None) -> list[int]:
        """Get the index from the data in database.

        Args:
            file_name (Optional[str], optional): The file name without .cif. Defaults to None.
            icsd_code (Optional[int], optional): ICSD code. Defaults to None.
            space_group (Optional[str], optional): Space group. Defaults to None.
            space_group_number (Optional[int], optional): No of space group. Defaults to None.
            element (Optional[list[str]], optional): A list, - means mustn't contain. Defaults to None.

        Returns:
            list[int]: The index of the data.

        Raises:
            ValueError: No data found!
        """
        result_index: list[int] = []
        for i, data in enumerate(self.data):
            if file_name and file_name != data.name:
                continue
            if icsd_code and icsd_code != data.icsd_code:
                continue
            if space_group and space_group != data.space_group:
                continue
            if space_group_number and space_group_number != data.space_group_number:
                continue
            if element:
                add_flag = True
                for e in element:
                    if e.startswith('-') and e[1:] in data.formula:
                        add_flag = False
                        break
                    if not e.startswith('-') and e not in data.formula:
                        add_flag = False
                        break
                if not add_flag:
                    continue
            result_index.append(i)
        if not result_index:
            raise ValueError('No data found!')
        return result_index

    def get_code(self, **kwargs) -> list[int]:
        """Get the ICSD code from the data in database.

        Args:
            **kwargs: The keyword arguments for finding the data.

        Returns:
            list[int]: The ICSD code of the data.
        """
        return [self.data[i].icsd_code for i in self.index(**kwargs)]

    def plot(self,
             angle_range: Optional[AngleRange]=None,
             ax: Optional[Axes]=None,
             if_show: bool=True,
             cmap: str='tab10',
             color: Union[str, tuple[float, float, float, float]]='b',
             **kwargs) -> Axes:
        """Plot the diffraction pattern of cif file.

        Args:
            angle_range (Optional[AngleRange], optional): The angle limit. Defaults to None.
            ax (Optional[Axes], optional): The axes of the plot. Defaults to None.
            if_show (bool, optional): Whether to show the plot. Defaults to True.
            cmap (str, optional): The colormap of the plot. Defaults to 'tab10'.
            color (Union[str, tuple[float, float, float, float]], optional): The color of the plot. Defaults to 'b'.
            **kwargs: The keyword arguments for finding index of the data.

        Raises:
            ValueError: Please input the keyword arguments for finding the data.
            ValueError: The ax is not correctly set.
        """
        if not kwargs:
            raise ValueError('Please input the keyword arguments for finding the data.')
        index = self.index(**kwargs)
        if angle_range is None:
            angle_range = AngleRange(left=0, right=90)
        if ax is None:
            _, ax = plt.subplots()
        if ax is None:
            raise ValueError('The ax is not correctly set!')
        colormap = plt.cm.get_cmap(cmap)
        for i in index:
            if len(index) > 1:
                # If there are multiple data, we use the colormap.
                ax.vlines(self.data[i].two_theta, 0,
                          self.data[i].intensity,
                          color=colormap(i % 10), linewidth=1)
            else:
                # If there is only one data, we use the color parameter, with text.
                ax.vlines(self.data[i].two_theta, 0,
                          self.data[i].intensity,
                          color=color, linewidth=1)
                ax.text(0.5, 0.5, f'{self.data[i].formula}-{self.data[i]}', transform=ax.transAxes)
                # Also add text indicate the miller indexs on each vline.
                for j, hkl in enumerate(self.data[i].hkl):
                    ax.text(self.data[i].two_theta[j], self.data[i].intensity[j],
                            f'({hkl.h} {hkl.k} {hkl.l})', fontsize=8, rotation=90, color=color)
        ax.set_xlim(angle_range.left, angle_range.right)
        if if_show:
            plt.show()
        return ax
