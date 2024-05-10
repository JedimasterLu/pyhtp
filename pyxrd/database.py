# -*- coding: utf-8 -*-
"""
Filename: database.py
Author: Junyuan Lu
Contact: Lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
import os
import pickle
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser


class XrdDatabase:
    ''' A class that generated pattern database and structure database from .cif files.
    '''
    def __init__(self, file_dir: str, save_dir: str=''):
        '''__init__ create a instance of XrdDatabase.

        Args:
            file_dir (str): The directory of .cif files.
            save_path (str, optional): The path to save database files. Defaults to ''.
        '''
        self.file_dir = file_dir
        if save_dir:
            self.save_dir = save_dir
        else:
            self.save_dir = file_dir
        self.pattern_database = None
        self.structure_database = None

    def process(self, if_save: bool=True) -> tuple[list[dict[str, any]], list[Structure]]:
        '''process generate pattern database and structure database from .cif files.
        '''
        cif_files = os.listdir(self.file_dir)
        pattern_database = []
        structure_database = []
        for file_name in cif_files:
            if file_name.split('.')[-1] != 'cif':
                continue
            file_path = self.file_dir + file_name
            parser = CifParser(file_path)
            cif_data = parser.as_dict()
            # Because the initial key is a long string, we need to get the first key.
            cif_data = cif_data[str(list(cif_data.keys())[0])]
            # Remove the spaces in cif_data['_space_group_name_H-M_alt']
            cif_data['_space_group_name_H-M_alt'] = cif_data['_space_group_name_H-M_alt'].replace(' ', '')
            structure = Structure.from_file(file_path, primitive=False, merge_tol=0.01)
            structure_database.append(structure)
            xrd = XRDCalculator(wavelength='CuKa')
            pattern = xrd.get_pattern(structure)
            pattern = pattern.as_dict()
            xrd_data = {
                'name': file_name,
                'two_theta': pattern['x'],
                'intensity': pattern['y'],
                'space_group': cif_data['_space_group_name_H-M_alt'],
                'space_group_number': cif_data['_space_group_IT_number'],
                'formula': cif_data['_chemical_formula_structural'],
                'lattice_abc': [cif_data['_cell_length_a'], cif_data['_cell_length_b'], cif_data['_cell_length_c']],
                'lattice_angles': [cif_data['_cell_angle_alpha'], cif_data['_cell_angle_beta'], cif_data['_cell_angle_gamma']],
                'icsd_code': cif_data['_database_code_ICSD'],
            }
            pattern_database.append(xrd_data)
        if if_save:
            with open(f'{self.save_dir}pattern.pkl', 'wb') as db:
                pickle.dump(pattern_database, db)
            with open(f'{self.save_dir}structure.pkl', 'wb') as db:
                pickle.dump(structure_database, db)
        self.pattern_database = pattern_database
        self.structure_database = structure_database
        return pattern_database, structure_database

    def get_pattern(self) -> list[dict[str, any]]:
        '''get_pattern return pattern database from file.

        Returns:
            list[dict[str, any]]: pattern database
                element: {
                    'name': file name of .cif file (str),
                    'two_theta': two theta angles of peaks (np.ndarray),
                    'intensity': intensities of peaks (np.ndarray)
                }
        '''
        pattern_database = self.process(if_save=False)[0]
        return pattern_database

    def plot_pattern(self, file_name: str, two_theta_range: list[float]=None):
        '''plot_pattern plot a pattern from database, use pymatgen.analysis.diffraction

        Args:
            file_name (str): The name of .cif file.

        Raises:
            ValueError: If no such file in database.
        '''
        if not self.pattern_database:
            self.process(if_save=False)
        if file_name not in self.pattern_database['name']:
            raise ValueError('No such file in database.')
        if two_theta_range is None:
            two_theta_range = [0, 90]
        for pattern in self.pattern_database:
            if pattern['name'] == file_name:
                xrd = XRDCalculator()
                xrd.get_plot(
                    structure=self.structure_database[self.pattern_database.index(pattern)],
                    two_theta_range=tuple(two_theta_range),
                    fontsize=12
                )
                break
