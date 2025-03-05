# -*- coding: utf-8 -*-
"""
Define a class to read file and store ellipsometry data.
Arthur: Junyuan Lu
Email: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.typing import NDArray
from .spectrum import EllipSpectrum
from ..typing import SampleInfo, SpectrumInfo


class EllipDatabase:
    """A class to process the ellipsometry data of a high-throughput sample.
    """
    def __init__(
            self,
            file_path: str,
            info: SampleInfo):
        """Create an instance of EllipDatabase from a csv file of the sample.

        Args:
            file_path (str): The path of the csv file. The file should be directly
                generated from Complete EASE software. The first row should be the
                information of the sample, and the first column should be the wavelength.
                Then, the data is like n1, k1, n2, k2, ...
            info (SampleInfo): The information of the sample. Please refer to the
                SampleInfo class.

        Raises:
            ValueError: Please input the film thickness of the sample.
        """
        if not file_path.endswith('.csv'):
            raise ValueError("The file should be a csv file.")
        self.data = []
        wavelength, dataset = self._read_file(file_path)
        self.wavelength = wavelength
        info = info._replace(
            point_number=len(dataset.columns) // 2,
            wavelength_range=(wavelength[0], wavelength[-1]))
        assert info.wavelength_range is not None
        for i in range(info.point_number):
            self.data.append(
                EllipSpectrum(
                    n=dataset[f'n_{i}'].to_numpy(),
                    k=dataset[f'k_{i}'].to_numpy(),
                    wavelength=self.wavelength,
                    info=SpectrumInfo(
                        name=info.name, index=i,
                        wavelength_range=info.wavelength_range,
                        element=info.element,
                        temperature=info.temperature)))
        self._file_path = file_path
        self.info = info

    def __len__(self) -> int:
        return len(self.data)

    def _read_file(self, file_path: str) -> tuple[NDArray, DataFrame]:
        """Read the csv file of the sample.
        """
        dataset = pd.read_csv(file_path, skiprows=1)
        # The first column is the wavelength
        wavelength = dataset.iloc[:, 0].to_numpy()
        # Then, the data is like n1, k1, n2, k2, ...
        # The column names are like [n | k]: (x,y), x and y is the coordinate of the point
        # Rename them to [n | k]_i, i is the index of the point
        dataset = dataset.drop(columns=dataset.columns[0])
        dataset.columns = [f"n_{i // 2}" if i % 2 == 0 else f"k_{(i - 1) // 2}"
                           for i in range(len(dataset.columns))]
        return wavelength, dataset

    def copy(self):
        """Return a copy of the class."""
        return EllipDatabase(self._file_path, self.info)

    def get_property(
            self,
            property_name: Literal['n', 'k', 'absorp'],
            wavelength: float | list[float] | NDArray[np.float64] | None = None,
            index: int | list[int] | NDArray[np.int_] = -1) -> NDArray[np.float64]:
        """Get specific property of the sample.

        Args:
            property_name (Literal['n', 'k', 'absorp']): The property to get. 'n' for
                refractive index, 'k' for extinction coefficient, 'absorp' for absorption coefficient.
            wavelength (float | list[float] | NDArray[np.float64], optional): The wavelength
                to get the property. If None, return the whole property. Defaults to None.
                If multiple values are provided, return the property of these wavelengths. The shape
                of the return value is (len(index), len(wavelength)).
            index (int | list[int] | NDArray[np.int_], optional): The index of the point to get
                the property. If index < 0, return the property of all points. Defaults to -1.
        """
        # Check if all the wavelength are in the range
        if wavelength:
            assert self.info.wavelength_range is not None
            temp_w = wavelength if isinstance(wavelength, list) else [wavelength]
            for w in temp_w:
                if not self.info.wavelength_range[0] <= w <= self.info.wavelength_range[1]:
                    raise ValueError(f"The wavelength {w} is out of the range \
                                     {self.info.wavelength_range}.")
        # Process index
        if isinstance(index, int) and index < 0:
            point_index = list(range(self.info.point_number))
        elif isinstance(index, int):
            point_index = [index]
        elif isinstance(index, np.ndarray):
            point_index = index.tolist()
        else:
            point_index = index
        assert isinstance(point_index, list) and len(point_index) > 0
        # Get the wavelength index
        if wavelength is None:
            wavelength_index = list(range(len(self.wavelength)))
        else:
            if isinstance(wavelength, (float, int)):
                wavelength_index = [np.argmin(np.abs(self.wavelength - wavelength))]
            else:
                wavelength_index = [np.argmin(np.abs(self.wavelength - w)) for w in wavelength]
        assert len(wavelength_index) > 0
        # Get the property
        if property_name == 'n':
            return np.array([self.data[i].n[wavelength_index] for i in point_index])  # type: ignore
        if property_name == 'k':
            return np.array([self.data[i].k[wavelength_index] for i in point_index])  # type: ignore
        if property_name == 'absorp':
            return np.array([self.data[i].absorp[wavelength_index] for i in point_index])  # type: ignore

    @staticmethod
    def fom(
        crystalline_data: EllipDatabase,
        amorphous_data: EllipDatabase,
        wavelength: float | list[float] | NDArray[np.float64] | None = None,
        index: int | list[int] | NDArray[np.int_] = -1,
    ) -> NDArray[np.float64]:
        """Calculate the figure of merit of the sample from two EllipDatabase.

        The FOM is defined as: FOM = (n_cry - n_amo) / (k_cry + k_amo)

        Args:
            crystalline_data (EllipDatabase): The ellipsometry data of the crystalline sample.
            amorphous_data (EllipDatabase): The ellipsometry data of the amorphous sample.
            wavelength (float | list[float] | NDArray[np.float64] | None, optional):
                The parameter pass to get_property. Defaults to None.
            index (int | list[int] | NDArray[np.int_], optional):
                The parameter pass to get_property. Defaults to -1.

        Returns:
            NDArray[np.float64]: The figure of merit of the sample.
                Shape: (len(index), len(wavelength)).
        """
        n_cry = crystalline_data.get_property('n', wavelength, index)
        k_cry = crystalline_data.get_property('k', wavelength, index)
        n_amo = amorphous_data.get_property('n', wavelength, index)
        k_amo = amorphous_data.get_property('k', wavelength, index)
        return np.abs((n_cry - n_amo) / (k_cry + k_amo))
