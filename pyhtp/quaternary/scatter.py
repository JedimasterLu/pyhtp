# -*- coding: utf-8 -*-
'''
Define the scatter plot for quaternary phase diagram.
'''
from typing import Optional, Literal, Union
import io
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..xrd import XrdDatabase, ICSD
from .utils import get_coord, tet_to_car


def _bi_scatter(
        ax: Axes3D,
        points: NDArray,
        left_color: str,
        right_color: str,
        marker_size: int,
        vertices: NDArray,
        rotation: float = 0) -> None:
    '''Plot scatters with pie markers that has 2 colors'''
    car_points = tet_to_car(vertices, points)
    t = Affine2D().rotate_deg(rotation)
    filled_marker_style = {
        'marker': MarkerStyle('o', 'left', t),
        'markersize': marker_size,
        'markerfacecolor': left_color,
        'markerfacecoloralt': right_color,
        'markeredgecolor': '#F7F7F7',
        'markeredgewidth': 0}
    ax.scatter(car_points,
               linewidth=0,
               **filled_marker_style)


def _multi_scatter(
        ax: Axes3D,
        points: NDArray,
        color: list[str],
        marker_size: int,
        vertices: NDArray,
        rotation: float = 0) -> None:
    """Plot scatters with pie markers that has more than 2 colors"""
    # Convert points to cartisian coordinates
    car_points = tet_to_car(vertices, points)
    # Generate pie image
    slice_num = len(color)
    sizes = np.ones(slice_num) / slice_num
    fig, ax0 = plt.subplots()
    ax0.pie(sizes, explode=(0, 0, 0), colors=color,
            autopct='', shadow=False, startangle=rotation)
    plt.axis('equal')
    # fig.set_size_inches(marker_size / fig.dpi,
    #                     marker_size / fig.dpi)
    # Save a temp figure and read it as image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    # Read the image from the bytes buffer
    pie_image = OffsetImage(plt.imread(buf, format="png"), zoom=marker_size / fig.dpi / 3.3)
    for point in car_points:
        ab = AnnotationBbox(pie_image, point, frameon=False)
        ax.add_artist(ab)


def scatter_quaternary(
        value: Union[list[str | int], NDArray],  # type: ignore
        label: tuple[str, str, str, str],
        database: Optional[XrdDatabase] = None,
        icsd: Optional[ICSD] = None,
        coord: Optional[list[tuple[float, float, float, float]]] = None,  # type: ignore
        color: Optional[dict[str, str]] = None,
        rotation: Optional[dict[str, float]] = None,
        ticknum: int = 5,
        path_type: Literal['normal', 'snakelike'] = 'normal',
        composition_type: Literal['atomic', 'volumetric'] = 'atomic',
        interactive: bool = False,
        ax: Optional[Axes3D] = None,
        **kwargs) -> None:
    """Plot the quaternary scatter phase diagram.

    Args:
        value (Union[list[str | int], np.ndarray]): The phase of each point.
        label (tuple[str, str, str, str]): The label of each element.
        database (Optional[XrdDatabase], optional): The XRD database. Defaults to None.
        icsd (Optional[ICSD], optional): The ICSD database. If defined, the plots in interactive mode will display icsd diffraction lines. Defaults to None.
        coord (Optional[list[tuple[float, float, float, float]]], optional): The coordinates of each point. Defaults to None.
        color (Optional[dict[str, str]], optional): The color of each phase. Defaults to None.
        rotation (Optional[dict[str, float]], optional): The rotation of each phase. Defaults to None.
        ticknum (int, optional): The number of ticks on the edges. Defaults to 5.
        path_type (Literal['normal', 'snakelike'], optional): The type of path. Defaults to 'normal'.
        composition_type (Literal['atomic', 'volumetric'], optional): The type of composition. Defaults to 'atomic'.
        interactive (bool, optional): Whether to enable the interactive mode. Defaults to False.
        ax (Optional[Axes3D], optional): The axes. Defaults to None.

    Raises:
        TypeError: If the ax is not an instance of Axes3D.
    """
    # Convert the value to list[str]
    value: list[str] = [str(i) for i in value]  # type: ignore
    # Check if the ax is None
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if not isinstance(ax, Axes3D):
        raise TypeError("The ax should be an instance of Axes3D.")
    # Process coordinates from database
    if database is None and coord is None:
        raise ValueError("The database or coordinates should be provided.")
    if database is not None and coord is None:
        coord: NDArray = get_coord(
            label=label,
            side_num=int(np.sqrt(len(value))),
            info=database.info,
            composition_type=composition_type)
    elif database is None and coord is not None:
        coord: NDArray = np.array(coord)
    if coord is None:
        raise ValueError("The database or coordinates should be provided.")
    if isinstance(coord, list):
        coord = np.array(coord)
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
                fontfamily=kwargs.pop('fontfamily', 'Calibri'))
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
    # Reshape value to square form
    side_num = int(np.sqrt(len(coord)))
    value: NDArray = np.array(value).reshape(side_num, side_num)
    if path_type == 'snakelike':
        value[1::2] = value[1::2, ::-1]
    # Color
    if color is None:
        if len(np.unique(value)) > 10:
            cmap = plt.cm.get_cmap('tab20')
        else:
            cmap = plt.cm.get_cmap('tab10')
        color = {}
        for i, v in enumerate(np.unique(value)):
            color[v] = to_hex(cmap(i), keep_alpha=True)
    # Plot scatters
    for phase in np.unique(value.flatten()):
        coo = coord[(value == phase).flatten()]
        if '+' in phase:
            rot = 0
            if rotation is not None and phase in rotation:
                rot = rotation[phase]
            sep_phase = phase.split('+')
            sep_phase = [i.strip() for i in sep_phase]
            if len(sep_phase) == 2:
                _bi_scatter(ax, coo, color[sep_phase[0]], color[sep_phase[1]],
                            8, vertices, rot)
            else:
                _multi_scatter(ax, coo, [color[i] for i in sep_phase],
                               8, vertices, rot)
        else:
            car_coords = tet_to_car(vertices, coo)
            ax.scatter(car_coords[:, 0], car_coords[:, 1], car_coords[:, 2],  # type: ignore
                       s=25, c=color[phase], picker=interactive)
    # Interactive mode
    if interactive:
        if database is None:
            raise ValueError("The database should be provided for interactive mode.")

        def _onpick(event):
            """Scatter plot pick event."""
            # Make a copy of the kwargs
            # Avoid modifying the original kwargs
            kw = kwargs.copy()
            # Get the index of the selected point
            index = event.ind
            if len(index) > 1:
                print("Multiple points selected. Zoom in to select a single point.")
                return
            # Get the xrd pattern of the specific point
            pattern = database.data[index[0]]
            # Perform postprocessing based on kwargs
            pattern = pattern.subtract_baseline(lam=kw.pop('lam', -1))
            pattern = pattern.smooth(
                window=kw.pop('window', -1),
                factor=kw.pop('factor', -1))
            # Plot the xrd pattern of the specific point
            if icsd is None:
                pattern.plot(if_peak=True, **kw)
                plt.title(f"{pattern.info.name}-{pattern.info.index}")
                plt.show()
            else:
                pattern.plot_with_icsd(
                    icsd=icsd,
                    number=kw.pop('number', 5),
                    cmap=kw.pop('cmap', 'tab10'),
                    title=f"{pattern.info.name}-{pattern.info.index}",
                    if_peak=True, **kw)

        fig.canvas.mpl_connect('pick_event', _onpick)
    # Legend
    ax.legend([Line2D([0], [0], marker='o', color='w', label=i,
                      markerfacecolor=color[i], markersize=10)
               for i in color], list(color.keys()), loc='right')
    plt.show()
