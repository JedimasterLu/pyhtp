# -*- coding: utf-8 -*-
"""
This module contains the class for XRF composition database.
- Author: Junyuan Lu
- E-mail: lujunyuan@sjtu.edu.cn
"""
from __future__ import annotations
from typing import Literal
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.widgets import RectangleSelector

MaskVertices = tuple[tuple[float, float], tuple[float, float],
                     tuple[float, float], tuple[float, float]]


class XRFDatabase:
    """A Class that load and process XRF data.

    The class is designed to correct the compositional position
    of plots with high-throughput XRF data. The data is stored in txt files,
    and each file contains the composition of a single element.
    The class is current designed for square sample. If the sample shape is triangle,
    the class should be modified.
    """
    def __init__(
            self,
            file_dir: str,
            elements: list[str] | NDArray[np.str_] | None = None,
            if_mask: bool = True,
            mask_vertices: MaskVertices | list[MaskVertices] | None = None,
            vert_path: str | None = None,
            data: list[NDArray] | None = None):
        """Create an instance of XRFDatabase.

        The file_dir contains the path to the directory of the XRF data files.
        The elements is a list of elements that are included in the database.
        Please make sure the order of the elements is consistent with the
        order of the data files.

        Args:
            file_dir (str): The directory of the XRF data files.
            elements (list[str] | NDArray[np.str_] | None, optional):
                The elements included in the database.
                If None, the elements will be read from the name of data files.
            if_mask (bool, optional): If True, the data will be masked. Defaults to True.
                If true and mask_vertices is None, the vertices will be obtained from
                the user interactively. If if_mask is False, the mask_vertices should be None.
            mask_vertices (MaskVertices | list[MaskVertices] | None, optional):
                The vertices of the rectangle area that contains the valid data.
                The form is ((x1, y1), (x2, y2), (x3, y3), (x4, y4)).
                Point 1-4 are the lower left, lower right, upper right, and upper left.
                If None, no rectangle mask will be applied.
            vert_path (str | None, optional):
                The path of the pkl file that contains the vertices of the rectangle area.
            data (list[NDArray] | None, optional):
                The data of the XRF data. If not None, the data will be directly used.
        """
        self._file_dir = file_dir
        # Get the elements from the file names
        if elements is None:
            self.elements = [
                file_path.split('.')[0] for file_path in sorted(os.listdir(file_dir))
                if file_path.endswith('.txt')]
        else:
            self.elements = list(elements)
        # If data is directly provided, skip the reading process
        if data is not None:
            self.data = data
            self._vertices = mask_vertices
            return
        # Read the data from the file
        self.data = []
        self._vertices = mask_vertices
        if not if_mask:
            self._read_dir(file_dir)
        else:
            if self._vertices is None and vert_path is None:
                self._get_vertices()
            elif self._vertices is None and vert_path is not None:
                with open(vert_path, 'rb') as f:
                    self._vertices = pickle.load(f)
            elif isinstance(self._vertices, tuple):
                self._vertices = [self._vertices] * len(self.elements)
            assert isinstance(self._vertices, list)
            self._read_dir(file_dir, self._vertices)

    def save_vertices(self, file_name: str = 'maskverts.pkl') -> None:
        """Save the vertices of the rectangle area that contains the valid data into
        a pkl file in the file_dir.

        Args:
            file_name (str, optional): The file name. Defaults to 'maskverts.pkl'.
        """
        vert_path = os.path.join(self._file_dir, file_name)
        with open(vert_path, 'wb') as f:
            pickle.dump(self._vertices, f)

    def _get_vertices(self) -> None:
        """Get the vertices of the rectangle area that contains the valid data.

        The function will display the data of each element and ask the user to
        drag a rectangle area that contains the valid data.
        """
        raw_data = []
        self._vertices = []
        # Firstly, implort the raw data
        for file_path in sorted(os.listdir(self._file_dir)):
            if not file_path.endswith('.txt'):
                continue
            data = self._read_data(os.path.join(self._file_dir, file_path))
            data = data + np.abs(data.min()) + 0.1
            raw_data.append(data)
        # Then, plot the data and ask the user to drag a rectangle area
        for element, data in zip(self.elements, raw_data):
            _, ax = plt.subplots()
            ax.imshow(data, origin='lower', norm=LogNorm(vmin=data.min(), vmax=data.max()))
            ax.set_title(f'Drag a rectangle area that contains the valid data of {element}')
            ax.axis('off')
            vertices = self._get_rectangle_vertices(ax)
            self._vertices.append(vertices)

    @staticmethod
    def _get_rectangle_vertices(ax: Axes) -> MaskVertices:
        """Get the vertices of the rectangle area that contains the valid data."""
        vertices = []

        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            vertices.append((x1, y1))
            vertices.append((x2, y1))
            vertices.append((x2, y2))
            vertices.append((x1, y2))
            plt.close()
        _ = RectangleSelector(ax, onselect)
        plt.show()
        return tuple(vertices)

    def copy(self) -> XRFDatabase:
        """Return a deep copy of the XRFDatabase instance."""
        return XRFDatabase(
            file_dir=self._file_dir,
            elements=self.elements.copy(),
            data=self.data.copy(),
            mask_vertices=self._vertices)

    def _read_dir(self, file_dir: str, vertices: list[MaskVertices] | None = None):
        """Set the self.data using _read_data."""
        txt_files = [file_path for file_path in sorted(os.listdir(file_dir))
                     if file_path.endswith('.txt')]
        if vertices is None:
            for file_path in txt_files:
                if not file_path.endswith('.txt'):
                    continue
                self.data.append(
                    self._read_data(os.path.join(file_dir, file_path)))
        else:
            for file_path, vert in zip(txt_files, vertices):
                if not file_path.endswith('.txt'):
                    continue
                self.data.append(
                    self._read_data(os.path.join(file_dir, file_path), vert))

    @staticmethod
    def _read_data(file_path: str, vertices: MaskVertices | None = None) -> NDArray:
        """Read the XRF data from a txt file.

        The txt file should have no skiprows, and the delimeter should be ';'.

        Args:
            file_path (str): The path of the XRF data file.
            vertices (VerticesType | None, optional):
                The vertices of the rectangle area that contains the valid data.
                The form is ((x1, y1), (x2, y2), (x3, y3), (x4, y4)).
                Point 1-4 are the lower left, lower right, upper right, and upper left.
                If None, no rectangle mask will be applied.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip(';\n') for line in lines]
            lines = [line.split(';') for line in lines]
            data = np.array(lines, dtype=float)
        # The vertices can form a rectangle area.
        # Only the data within the rectangle area is valid.
        # The data outside the rectangle area is invalid.
        # Get the valid data.
        # Detect the vertices of the rectangle area based on 4 vertices
        if vertices:
            lower_left, lower_right, upper_right, upper_left = vertices
            x_min = np.mean([lower_left[0], upper_left[0]]) + 3
            x_max = np.mean([upper_right[0], lower_right[0]]) - 3
            y_min = np.mean([lower_left[1], lower_right[1]]) + 3
            y_max = np.mean([upper_right[1], upper_left[1]]) - 3
            # Only remain the valid data
            data = data[int(y_min):int(y_max), int(x_min):int(x_max)]
        # All data should be positive
        data[data < 0] = 0
        return data

    def smooth(
            self,
            method: Literal['gaussian', 'median'] = 'gaussian',
            element: str | None = None,
            **kwargs) -> XRFDatabase:
        """Smoothing data with gaussian or median algorithm.

        Args:
            method (Literal['gaussian', 'median'], optional): The method of denoising.
                Defaults to 'gaussian'. Please refer to ``scipy.ndimage.gaussian_filter``
                and ``scipy.ndimage.median_filter`` for more details.
            element (str | None, optional): The data of the element to be denoised.
                If None, all data will be denoised. Defaults to None.
            **kwargs: The parameters of the denoising method.
                - For 'gaussian', the parameter is 'sigma'. The default value is 1.
                - For 'median', the parameter is 'size'. The default value is 3.
        """
        new_db = self.copy()
        index_to_be_denoised = []
        if element:
            index_to_be_denoised.append(new_db.elements.index(element))
        else:
            index_to_be_denoised = range(len(new_db.elements))
        for index in index_to_be_denoised:
            if method == 'gaussian':
                new_db.data[index] = gaussian_filter(
                    new_db.data[index], sigma=kwargs.get('sigma', 1))
            elif method == 'median':
                new_db.data[index] = median_filter(
                    new_db.data[index], size=kwargs.get('size', 3))
        return new_db

    def get_composition(
            self,
            x: float | list[float] | NDArray[np.float64],
            y: float | list[float] | NDArray[np.float64],
            order: tuple[str, str, str, str] | None = None) -> NDArray:
        """Get the composition of the XRF data.

        Args:
            x (float | list[float] | NDArray[np.float64]):
                The x coordinate of the point. The range of x is [0, 1].
            y (float | list[float] | NDArray[np.float64]):
                The y coordinate of the point. The range of y is [0, 1].
            order (tuple[str, str, str, str] | None, optional):
                The order of the elements. If None, the order of the elements will be
                the same as the order of the data files. Defaults to None.
            normalize (bool, optional): If True, the composition will be normalized,
                so that the sum of the composition is 1. Defaults to True.

        Returns:
            NDArray: The composition of the XRF data. The shape is (x.shape[0], 4).
        """
        x = np.array(x)
        y = np.array(y)
        comp = []
        for data in self.data:
            column = x * data.shape[1]
            column = column.astype(int)
            column[column >= data.shape[1]] = data.shape[1] - 1
            column[column <= 0] = 0
            row = (1 - y) * data.shape[0]
            row = row.astype(int)
            row[row >= data.shape[0]] = data.shape[0] - 1
            row[row <= 0] = 0
            comp.append(data[row, column])
        comp = np.array(comp).T
        # Normalize the composition
        comp /= np.sum(comp, axis=1)[:, None]
        if order:
            order_index = [self.elements.index(element) for element in order]
            comp = comp[:, order_index]
        return comp

    def get_composition_map(
            self,
            side_number: int,
            order: tuple[str, str, str, str] | None = None) -> NDArray:
        """Get the composition of the XRF data.

        Args:
            side_number (int): The number of points in each side of the rectangle area.
            order (tuple[str, str, str, str] | None, optional):
                The order of the elements. If None, the order of the elements will be
                the same as the order of the data files. Defaults to None.

        Returns:
            NDArray: The composition of the XRF data. The shape is (side_number**2, 4).
        """
        # 修改网格生成时的索引顺序，确保与raw数据保持一致
        i, j = np.meshgrid(np.arange(side_number), np.arange(side_number))
        # Normalize the coordinates
        abs_coord = np.stack(
            (i / (side_number - 1), j / (side_number - 1)), axis=-1).reshape(-1, 2)
        x = abs_coord[:, 0]
        y = abs_coord[:, 1]
        comp = self.get_composition(x, y, order)
        # 确保reshape后的数据方向与raw数据一致
        return comp

    def get_composition_df(self, side_number: int) -> pd.DataFrame:
        """Get the composition of the XRF data into a pd.DataFrame.

        Args:
            side_number (int): The number of points in each side of the rectangle area.

        Returns:
            pd.DataFrame: The composition of the XRF data. The shape is (side_number**2, 4).
                The column name is the elements.
        """
        composition = self.get_composition_map(side_number)
        return pd.DataFrame(composition, columns=self.elements)

    def plot(
            self,
            element: str | list[str] | None = None,
            ax: Axes | list[Axes] | None = None,
            vlim: tuple[float, float] | None = None,
            data_type: Literal['raw', 'composition'] = 'raw',
            plot_type: Literal['imshow', 'contourf'] = 'imshow',
            sharev: bool = False,
            **kwargs) -> None:
        """Plot the 2d imshow of the XRF data of a single element.

        Args:
            element (str | list[str] | None, optional): The element to be plotted.
                If None, all elements will be plotted. Defaults to None.
            ax (Axes | None, optional): matplotlib Axes. For second development.
                Defaults to None.
            cmap (str, optional): Defaults to 'viridis'.
            vlim (tuple[float, float] | None, optional): The limit of the colorbar.
                Defaults to None. If None, the colorbar will be automatically set.
            **kwargs: Other parameters of the imshow function and get_composition_df.

        Returns:
            tuple[AxesImage, Colorbar]: The AxesImage and Colorbar of the plot.
        """
        if element is None:
            element = self.elements
        elif isinstance(element, str):
            element = [element]
        assert isinstance(element, list) and element
        if ax is None:
            _, ax = plt.subplots(
                nrows=1, ncols=len(element), figsize=(4 * len(element), 4))
            if len(element) > 1:
                ax = ax.flatten().tolist()  # type: ignore
        if isinstance(ax, Axes):
            ax = [ax]
        assert isinstance(ax, list)
        data = []
        if data_type == 'raw':
            data = [self.data[self.elements.index(e)] for e in element]
        elif data_type == 'composition':
            for e in element:
                side_number = kwargs.get('side_number', 100)
                composition = self.get_composition_map(side_number)
                data.append(
                    np.flipud(composition[:, self.elements.index(e)].reshape(
                        side_number, side_number)))
        assert isinstance(data, list)
        if vlim is None:
            vlim = (min(data.min() for data in data),
                    max(data.max() for data in data))
        for d, e, a in zip(data, element, ax):
            if not sharev:
                temp_vlim = (d.min(), d.max())
            else:
                temp_vlim = vlim
            im = None
            if plot_type == 'imshow':
                im = a.imshow(
                    d, origin='upper', vmin=temp_vlim[0], vmax=temp_vlim[1], **kwargs)
            elif plot_type == 'contourf':
                im = a.contourf(
                    d, origin='upper', vmin=temp_vlim[0], vmax=temp_vlim[1], **kwargs)
                a.contour(
                    d, origin='upper', vmin=temp_vlim[0], vmax=temp_vlim[1],
                    linewidths=1, levels=kwargs.get('levels', None),
                    colors='black', alpha=0.6)
            a.set_title(e)
            a.axis('off')
            cbar = plt.colorbar(im, ax=a, fraction=0.045, pad=0.04)
            cbar.mappable.set_clim(temp_vlim)
        plt.tight_layout()

    def max(self) -> float:
        """Get the maximum value of the data."""
        return max(data.max() for data in self.data)

    def min(self) -> float:
        """Get the minimum value of the data."""
        return min(data.min() for data in self.data)

    def rotate90(self, times: int = 1) -> XRFDatabase:
        """Rotate the data 90 degrees clockwise.

        Args:
            times (int, optional): The times of rotation. Defaults to 1.

        Returns:
            XRFDatabase: The rotated XRFDatabase.
        """
        new_db = self.copy()
        for i, data in enumerate(new_db.data):
            new_db.data[i] = np.rot90(data, times)
        return new_db
