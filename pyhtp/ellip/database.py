# -*- coding: utf-8 -*-
"""
Define a class to read file and store ellipsometry data.
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.typing import NDArray
from .spectrum import EllipSpectrum, EllipPointSet
from ..typing import SampleInfo, SpectrumInfo


class EllipDatabase:
    """A class to read file and store ellipsometry data."""
    def __init__(
            self,
            file_path: str | tuple[str, str],  # type: ignore
            info: SampleInfo):
        """Initialize the class from csv file(s) of the sample.

        Args:
            file_path (str | tuple[str, str]): The path of the file. Input up to 2 files to calculate FOM.
            info (SampleInfo): The information of the sample. The length of info.temperature should be the same as the file_path.

        Raises:
            ValueError: Please input the film thickness of the sample.
        """
        if not all(p.endswith('.csv') for p in file_path):
            raise ValueError("The file should be csv file.")
        self.data = []
        if isinstance(info.temperature, float) and isinstance(file_path, str):
            wavelength, dataset = self._read_file(file_path)
            self._data_set = dataset
            self.wavelength = wavelength.astype(float)
            info = info._replace(point_number=len(dataset.columns) // 2,
                                 wavelength_range=(wavelength[0], wavelength[-1]))
            assert (info.wavelength_range is not None
                    and isinstance(info.temperature, float))
            for i in range(info.point_number):
                self.data.append(
                    EllipSpectrum(
                        refrac=dataset[f'n_{i}'].to_numpy(),
                        extinc=dataset[f'k_{i}'].to_numpy(),
                        wavelength=self.wavelength,
                        info=SpectrumInfo(
                            name=info.name,
                            index=i,
                            wavelength_range=info.wavelength_range,
                            element=info.element,
                            temperature=info.temperature)))
        elif isinstance(info.temperature, tuple) and isinstance(file_path, tuple):
            wavelength1, dataset1 = self._read_file(file_path[0])
            wavelength2, dataset2 = self._read_file(file_path[1])
            self._data_set = dataset1, dataset2
            if not np.array_equal(wavelength1, wavelength2):
                raise ValueError("The wavelength of the two files should be the same.")
            self.wavelength = wavelength1.astype(float)
            info = info._replace(point_number=len(dataset1.columns) // 2,
                                 wavelength_range=(wavelength1[0], wavelength1[-1]))
            assert (info.wavelength_range is not None
                    and isinstance(info.temperature, tuple))
            for i in range(info.point_number):
                spec1 = EllipSpectrum(
                    refrac=dataset1[f'n_{i}'].to_numpy(),
                    extinc=dataset1[f'k_{i}'].to_numpy(),
                    wavelength=self.wavelength,
                    info=SpectrumInfo(
                        name=info.name,
                        index=i,
                        wavelength_range=info.wavelength_range,
                        element=info.element,
                        temperature=info.temperature[0]))
                spec2 = EllipSpectrum(
                    refrac=dataset2[f'n_{i}'].to_numpy(),
                    extinc=dataset2[f'k_{i}'].to_numpy(),
                    wavelength=self.wavelength,
                    info=SpectrumInfo(
                        name=info.name,
                        index=i,
                        wavelength_range=info.wavelength_range,
                        element=info.element,
                        temperature=info.temperature[1]))
                # Make sure the first one is the higher temperature
                # The higher temperature is defined as the crystalline phase
                if spec1.info.temperature < spec2.info.temperature:
                    spec1, spec2 = spec2, spec1
                self.data.append(EllipPointSet(spec1, spec2))
        else:
            raise ValueError("The type and length of info should be the same as the file_path.")
        self._file_path = file_path
        self.info = info

    def __len__(self) -> int:
        return len(self.data)

    def _read_file(self, file_path: str) -> tuple[NDArray, DataFrame]:
        """Read the xlsx or csv file of the sample.

        Args:
            file_path (str): The path of the file.

        Returns:
            pd.DataFrame: The data of the sample.
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

    @property
    def fom(self) -> NDArray:
        """Return the figure of merit of the sample at a specific wavelength.

        Args:
            wavelength (int): The wavelength of the data.

        Returns:
            NDArray: The figure of merit of the sample at a specific wavelength.
        """
        if not all([isinstance(d, EllipPointSet) for d in self.data]):
            raise ValueError("Input a crystalline and a amorphous phase to calculate the figure of merit.")
        amo_n = np.array([d.amorphous.refrac for d in self.data])
        cry_n = np.array([d.crystalline.refrac for d in self.data])
        amo_k = np.array([d.amorphous.extinc for d in self.data])
        cry_k = np.array([d.crystalline.extinc for d in self.data])
        return np.abs(amo_n - cry_n) / (cry_k + amo_k)
