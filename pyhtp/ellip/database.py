# -*- coding: utf-8 -*-
"""
Define a class to read file and store ellipsometry data.
"""
import re
from typing import Literal, Union
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from ..typing import SampleInfo


class EllipDatabase:
    """A class to read file and store ellipsometry data."""
    def __init__(
            self,
            file_path: Union[str, tuple[str, str]],  # type: ignore
            info: SampleInfo):
        """Initialize the class from a xlsx or csv file of the sample.

        Args:
            file_dir (str | list[str]): The path of the file. Input 2 files to calculate FOM.
            info (SampleInfo): The information of the sample.

        Raises:
            ValueError: Please input the film thickness of the sample.
        """
        self.file_path = file_path
        if isinstance(file_path, str):
            file_path: list[str] = [file_path]
        if isinstance(file_path, tuple):
            file_path: list[str] = list(file_path)
        self.data: dict[str, DataFrame] = {}
        for index, p in enumerate(file_path):
            if index == 0:
                key = 'amorphous'
            elif index == 1:
                key = 'crystalline'
            else:
                raise ValueError("Input up to 2 files to calculate FOM.")
            self.data[key] = self._read_file(p)
        self.wavelength = self.get_wavelength()
        if info.film_thickness is None:
            raise ValueError("Please input the film thickness of the sample.")
        self.info = info

    def _read_file(self, file_path: str) -> DataFrame:
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
        columns: list[str] = data_ini.columns.values.tolist()
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
        return EllipDatabase(self.file_path, self.info)

    def get_data(self, param: Literal['n', 'k'], wavelength: int) -> NDArray:
        """Return the n/k data of the sample at a specific wavelength.

        Args:
            param (Literal['n', 'k']): The optical parameter of the sample.
            wavelength (int): The wavelength of the data.

        Returns:
            NDArray: The n/k data of the sample at a specific wavelength.
        """
        column = f"{param}_{wavelength}_z"
        result = []
        for _, data in self.data.items():
            result.append(data[column].values)
        result = np.array(result).squeeze()
        return result

    def get_fom(self, wavelength: int) -> NDArray:
        """Return the figure of merit of the sample at a specific wavelength.

        Args:
            wavelength (int): The wavelength of the data.

        Returns:
            NDArray: The figure of merit of the sample at a specific wavelength.
        """
        if len(self.data) != 2:
            raise ValueError("Input 2 files to calculate FOM.")
        ns = self.get_data('n', wavelength)
        ks = self.get_data('k', wavelength)
        return np.abs(ns[0] - ns[1]) / (ks[0] + ks[1])

    def get_len(self) -> int:
        """Return the number of data in the database."""
        return len(self.data[list(self.data.keys())[0]])

    def get_file_num(self) -> int:
        """Return the number of files in the database."""
        return len(self.data)

    def get_wavelength(self) -> NDArray:
        """Return the wavelength of the data."""
        wavelength = np.array([int(column.split('_')[1])
                               for column in self.data[list(self.data.keys())[0]].columns])
        # Remove repeated wavelength
        wavelength = np.unique(wavelength)
        return wavelength

    def get_alpha(self, wavelength: float) -> NDArray:
        """Return the absorption coefficient of the sample at a specific wavelength.

        Args:
            wavelength (float): The wavelength of the data.

        Returns:
            NDArray: The absorption coefficient of the sample at a specific wavelength.
        """
        ks = self.get_data('k', int(wavelength))
        return 4 * np.pi * ks / wavelength


def tauc_plot(k, wavelength):
    """_summary_

    Args:
        alpha (_type_): _description_
        wavelength (_type_): _description_
    """
    alpha = 4 * np.pi * k / wavelength
    import matplotlib.pyplot as plt
    plank = 6.62607015e-34
    c = 3e8
    nu = c / (wavelength * 1e-9)
    hnu = plank * nu
    ahnu = alpha * hnu
    plt.plot(hnu, ahnu, 'o')

