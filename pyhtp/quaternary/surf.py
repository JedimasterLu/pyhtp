# -*- coding: utf-8 -*-
'''
Define the surf plot for quaternary phase diagram.
'''
from typing import Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .utils import tet_to_car


def surf_quaternary(
        coord: list[tuple[float, float, float, float]],
        value: list[float],  # type: ignore
        label: tuple[str, str, str, str],
        ax: Optional[Axes3D] = None,
        ticknum: int = 5,
        cmap: str = 'summer',
        path_type: Literal['normal', 'snakelike'] = 'normal',
        **kwargs) -> tuple[Axes3D, Colorbar]:
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
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if not isinstance(ax, Axes3D):
        raise TypeError("The ax should be an instance of Axes3D.")
    # Construct tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    # Move the center of the tetrahedron to the origin
    vertices -= np.mean(vertices, axis=0)
    faces = [[vertices[j] for j in [0, 1, 2]],
             [vertices[j] for j in [0, 1, 3]],
             [vertices[j] for j in [0, 2, 3]],
             [vertices[j] for j in [1, 2, 3]]]
    ax.add_collection3d(
        Poly3DCollection(faces, alpha=0.05, linewidths=1,
                         edgecolors='darkslategray'))
    # Add labels to the vertices
    for i, txt in enumerate(label):
        ax.text(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                txt, size=20, color='darkslategray',
                fontfamily=kwargs.get('fontfamily', 'Calibri'))
    # Set aspect and lims and axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)
    ax.set_axis_off()
    ax.set_proj_type('persp')
    # Add ticks on the edges
    for start, end in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        for i in range(1, ticknum):
            tick = vertices[start] + (vertices[end] - vertices[start]) * i / ticknum
            ax.plot([tick[0]], [tick[1]], [tick[2]], 'ko', markersize=2)
    # Convert coord to Cartesian
    car_coord = tet_to_car(vertices, np.array(coord))
    # Reshape value to square form
    side_num = int(np.sqrt(len(coord)))
    value: NDArray = np.array(value).reshape(side_num, side_num)
    if path_type == 'snakelike':
        value[1::2] = value[1::2, ::-1]
    # Color
    colormap = plt.get_cmap(cmap)
    color = colormap(Normalize(min(value), max(value))(value))
    # Plot the surface
    ax.plot_surface(X=car_coord[:, 0].reshape(side_num, side_num),
                    Y=car_coord[:, 1].reshape(side_num, side_num),
                    Z=car_coord[:, 2].reshape(side_num, side_num),
                    facecolors=color, shade=False, antialiased=True,
                    rstride=1, cstride=1)
    # Plot colorbar and set the value range from value.min() to value.max()
    sm = plt.cm.ScalarMappable(cmap=colormap,
                               norm=Normalize(value.min(), value.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
             fontfamily=kwargs.get('fontfamily', 'Calibri'))
    return ax, cbar
