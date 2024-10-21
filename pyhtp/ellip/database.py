# -*- coding: utf-8 -*-
"""
Define a class to read file and store ellipsometry data.
"""
import re
import os
from typing import Literal
import numpy as np
import pandas as pd


class EllipDatabase:
    """A class to read file and store ellipsometry data."""
    def __init__(self, file_path: str | list[str]):
        """Initialize the class from a xlsx or csv file of the sample.

        Args:
            file_dir (str | list[str]): The path of the file. Input 2 files to calculate FOM.
        """
        self.file_path = file_path
        if isinstance(file_path, str):
            file_path = [file_path]
        self.data = {}
        for p in file_path:
            data_ini = self._read_file(p)
            p = os.path.basename(p)
            p = p.split('.')[0]
            self.data[p] = data_ini
        self.wavelength = self.get_wavelength()

    def _read_file(self, file_path: str) -> pd.DataFrame:
        """Read the xlsx or csv file of the sample.

        Args:
            file_path (str): The path of the file.

        Returns:
            pd.DataFrame: The data of the sample.
        """
        data_ini = pd.read_excel(file_path)
        # Process the Dataframe to change its columns to a standard form
        pattern = r"([nk]) of B-Spline @ (\d+)\.\d+ nm vs. Position"
        finded_index = []
        columns = data_ini.columns.values
        # Change the columns with n/k and wavelength
        for index, column in enumerate(columns):
            # maches = [[n/k, wavelength]]
            matches = re.findall(pattern, column)
            if matches:
                columns[index] = f"{matches[0][0]}_{matches[0][1]}"
                finded_index.append(index)
        # Change other columns start with "Unnamed" to the same form
        for index in finded_index:
            while index + 1 < len(columns) and columns[index + 1].startswith("Unnamed"):
                columns[index + 1] = columns[index]
                index += 1
        # Add the first row to the columns
        xyz = data_ini.iloc[0, :].values
        columns = [f"{column}_{xyz[index]}" for index, column in enumerate(columns)]
        columns = [column.lower().rstrip(' (cm)') for column in columns]
        # Change the columns of the Dataframe
        data_ini.columns = columns
        # Delete the first row
        data_ini.drop(index=0, inplace=True)
        return data_ini

    def copy(self):
        """Return a copy of the class."""
        return EllipDatabase(self.file_path)

    def get_param(self, param: Literal['n', 'k'], wavelength: int) -> np.ndarray:
        """Return the n/k data of the sample at a specific wavelength.

        Args:
            param (Literal['n', 'k']): The optical parameter of the sample.
            wavelength (int): The wavelength of the data.

        Returns:
            np.ndarray: The n/k data of the sample at a specific wavelength.
        """
        column = f"{param}_{wavelength}_z"
        result = []
        for _, data in self.data.items():
            result.append(data[column].values)
        result = np.array(result).squeeze()
        return result

    def get_fom(self, wavelength: int) -> np.ndarray:
        """Return the figure of merit of the sample at a specific wavelength.

        Args:
            wavelength (int): The wavelength of the data.

        Returns:
            np.ndarray: The figure of merit of the sample at a specific wavelength.
        """
        if len(self.data) != 2:
            raise ValueError("Input 2 files to calculate FOM.")
        ns = self.get_param('n', wavelength)
        ks = self.get_param('k', wavelength)
        return np.abs(ns[0] - ns[1]) / (ks[0] + ks[1])

    def get_len(self) -> int:
        """Return the number of data in the database."""
        return len(self.data[list(self.data.keys())[0]])

    def get_file_num(self) -> int:
        """Return the number of files in the database."""
        return len(self.data)

    def get_wavelength(self) -> np.ndarray:
        """Return the wavelength of the data."""
        wavelength = np.array([int(column.split('_')[1])
                               for column in self.data[list(self.data.keys())[0]].columns])
        # Remove repeated wavelength
        wavelength = np.unique(wavelength)
        return wavelength
