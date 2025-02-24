# -*- coding: utf-8 -*-
'''
Define the surf plot for quaternary phase diagram.
'''
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from .utils import tet_to_car, get_coord, build_tetrahedron
from ..xrf import XRFDatabase
from ..typing import SampleInfo


def plot_quat_surface(
        value: list[float] | NDArray[np.float_],
        axis_label: tuple[str, str, str, str],
        coord: list[tuple[float, float, float, float]] | NDArray | None = None,  # type: ignore
        info: SampleInfo | None = None,
        xrf_database: XRFDatabase | None = None,
        ax: Axes3D | None = None,
        tick_number: int = 5,
        cmap: str = 'viridis',
        path_type: Literal['normal', 'snakelike'] = 'normal',
        composition_type: Literal['atomic', 'volumetric'] = 'atomic',
        vlim: tuple[float, float] | None = None,
        vlim_percentile: tuple[float, float] | None = None,
        **kwargs) -> None:
    """Plot the quaternary surface phase diagram.

    Args:
        coord (list[tuple[float, float, float, float]]): Compositional coordinates of the points.
        value (list[float]): Values of the points.
        label (tuple[str, str, str, str]): Labels of the vertices.
        ax (Optional[Axes3D], optional): Defaults to None.
        ticknum (int, optional): Number of ticks on the edges. Defaults to 5.
        cmap (str, optional): Colormap. Defaults to 'summer'.
        path_type (Literal['normal', 'snakelike'], optional): Path type. Defaults to 'normal'.
        **kwargs: Additional keyword arguments for plot, such as fontfamily.

    Returns:
        Axes: The 3D axes.

    Raises:
        TypeError: If the ax is not an instance of Axes3D.
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
    assert isinstance(ax, Axes3D)

    # Process coordinates from database
    if coord is None:  # Generate coordinates from database
        if info is not None and xrf_database is None:
            coord = get_coord(
                element_order=axis_label,
                side_number=int(np.sqrt(len(value))),
                info=info,
                composition_type=composition_type)
        elif xrf_database is not None:
            coord = get_coord(
                element_order=axis_label,
                side_number=int(np.sqrt(len(value))),
                xrf_database=xrf_database,
                composition_type=composition_type)
        else:
            raise ValueError("Either XRD or XRF database should be provided.")
    if coord is not None:  # Use the provided coordinates
        coord = np.array(coord)
    assert isinstance(coord, np.ndarray)

    # Construct tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    build_tetrahedron(
        ax, vertices, axis_label, tick_number,
        fontfamily=kwargs.get('fontfamily', 'Calibri'))

    # The original coords are in tetrahedron coordinates
    # Convert to cartesian coordinates
    car_coords = tet_to_car(vertices, coord)
    car_coords -= np.mean(vertices, axis=0)

    # Reshape value to square form
    side_num = int(np.sqrt(len(coord)))
    value = np.array(value).reshape(side_num, side_num)
    if path_type == 'snakelike':
        value[1::2] = value[1::2, ::-1]
    # Color
    colormap = plt.get_cmap(cmap)

    if vlim is None:
        # Vmax is the top 5% of the values
        # Vmin is the bottom 5% of the values
        if vlim_percentile is None:
            vlim_percentile = (5, 95)
        temp = np.percentile(value, list(vlim_percentile))
        vlim = (temp[0], temp[1])

    # Normalize value to 0 to 1
    color = colormap(Normalize(*vlim)(value))
    # Plot the surface
    ax.plot_surface(
        X=car_coords[:, 0].reshape(side_num, side_num),
        Y=car_coords[:, 1].reshape(side_num, side_num),
        Z=car_coords[:, 2].reshape(side_num, side_num),
        facecolors=color, shade=False, antialiased=True,
        rstride=1, cstride=1)
    # Plot colorbar and set the value range from value.min() to value.max()
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=Normalize(*vlim))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=-0.05)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
             fontfamily=kwargs.get('fontfamily', 'Calibri'))

    plt.tight_layout()
