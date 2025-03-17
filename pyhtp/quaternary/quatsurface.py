# -*- coding: utf-8 -*-
'''
Define a class for quaternary surface plot.
'''
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..xrf import XRFDatabase
from ..ellip import SEDatabase
from .quatplot import QuatPlot
from .utils import get_coord


class QuatSurface(QuatPlot):
    """A class for quaternary surface plots.

    Attributes:
    """
    def __init__(
            self,
            value: list[float] | NDArray[np.float64],
            axis_label: tuple[str, str, str, str] | None = None,
            database: SEDatabase | XRFDatabase | None = None,
            coords: NDArray[np.float64] | None = None,
            path_type: Literal['normal', 'snakelike'] = 'snakelike',
            composition_type: Literal['atomic', 'volumetric'] = 'atomic',
            ax: Axes3D | None = None):
        """_summary_

        Args:
            value (list[str  |  int] | NDArray[np.str_  |  np.int_]):
                _description_
            axis_label (tuple[str, str, str, str] | None, optional):
                _description_. Defaults to None.
            database (SEDatabase | XRFDatabase | None, optional):
                _description_. Defaults to None.
            coords (NDArray[np.float64] | None, optional): _description_.
                Defaults to None.
            path_type (Literal['normal', 'snakelike'], optional):
                _description_. Defaults to 'snakelike'.
            composition_type (Literal['atomic', 'volumetric'], optional):
                _description_. Defaults to 'atomic'.
            ax (Axes3D | None, optional): _description_. Defaults to None.
        """
        self.database = database
        if axis_label is None:
            if isinstance(database, XRFDatabase):
                assert len(database.elements) == 4
                axis_label = tuple(database.elements)  # type: ignore
            elif isinstance(database, SEDatabase):
                assert len(database.info.element) == 4
                axis_label = tuple(database.info.element)  # type: ignore
            else:
                raise ValueError('Please provide the axis label if database is not given.')
        assert isinstance(axis_label, tuple)

        super().__init__(axis_label=axis_label, ax=ax)

        # Process the value
        self.value = np.array(value).flatten()
        self.side_number = int(np.sqrt(self.value.shape[0]))
        assert isinstance(self.value, np.ndarray)

        # Process the coords
        if coords is None:
            if isinstance(database, XRFDatabase):
                coords = get_coord(
                    element_order=self.axis_label,
                    side_number=self.side_number,
                    composition_type=composition_type,
                    xrf_database=database)
            elif isinstance(database, SEDatabase):
                coords = get_coord(
                    element_order=self.axis_label,
                    side_number=self.side_number,
                    composition_type=composition_type,
                    info=database.info)
            else:
                raise ValueError('Please provide the coords if database is not given.')
        assert isinstance(coords, np.ndarray)
        assert coords.shape[0] == len(self.value) and coords.shape[1] == 4
        self.coords = coords

        index_map = np.arange(len(self.value))
        if path_type == 'snakelike':
            index_map = index_map.reshape(self.side_number, self.side_number)
            index_map[1::2] = index_map[1::2, ::-1]
            index_map = index_map.flatten()
        self.index_map = index_map
        self.value = self.value[index_map]

    def rotate90(self, times: int = 1) -> None:
        """Rotate the coordinates 90 degrees.

        Please note that the direction of rotation depends on
        the self.axis_label.

        Args:
            times (int, optional): Defaults to 1.
        """
        self.coords = np.rot90(
            self.coords.reshape(
                self.side_number, self.side_number, 4), times).reshape(-1, 4)

    def plot(
            self,
            cmap: str = 'viridis',
            color: list[str] | NDArray[np.str_] | None = None,
            vlim: tuple[float, float] | float | None = None
    ) -> Poly3DCollection:
        """Plot the quaternary surface.

        Args:
            cmap (str, optional): _description_. Defaults to 'viridis'.
            color (list[str] | NDArray[np.str_] | None, optional): _description_. Defaults to None.
            vlim (tuple[float, float] | float | None, optional): _description_. Defaults to None.

        Returns:
            Poly3DCollection: _description_
        """
        artist = self.surface(
            coords=self.coords,
            value=self.value,
            cmap=cmap,
            color=color,
            vlim=vlim, artist_name='surface_main')
        return artist
