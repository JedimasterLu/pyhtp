# -*- coding: utf-8 -*-
'''
Define the scatter plot for quaternary phase diagram.
'''
from typing import Literal
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba_array
from mpl_toolkits.mplot3d import Axes3D
from ..xrd import XRDDatabase, CIFDatabase
from ..xrf import XRFDatabase
from .utils import get_coord, tet_to_car, build_tetrahedron


def _bi_scatter(
        ax: Axes3D,
        points: NDArray,
        left_color: str,
        right_color: str,
        markersize: int,
        vertices: NDArray,
        rotation: float = 0) -> None:
    '''Plot scatters with pie markers that has 2 colors'''
    car_points = tet_to_car(vertices, points)
    t = Affine2D().rotate_deg(rotation)
    filled_marker_style = {
        'marker': MarkerStyle('o', 'left', t),
        'markersize': markersize,
        'markerfacecolor': left_color,
        'markerfacecoloralt': right_color,
        'markeredgecolor': '#F7F7F7',
        'markeredgewidth': 0}
    ax.scatter(car_points,
               linewidth=0,
               **filled_marker_style)


def _scatter(
        ax: Axes3D,
        label: NDArray,
        coords: NDArray,
        color: dict[str, str],
        markersize: int,
        vertices: NDArray,
        rotation: dict[str, float] | None = None,
        picker: bool = False) -> list:
    """Plot scatters. Multi phase points are still treated as single phase points."""
    # The original coords are in tetrahedron coordinates
    # Convert to cartesian coordinates
    car_coords = tet_to_car(vertices, coords)
    car_coords -= np.mean(vertices, axis=0)

    # Set transform list of the MarkerStyle
    if rotation is None:
        rotation = {i: 0 for i in np.unique(label)}
    else:
        rotation = {i: rotation.get(i, 0) for i in np.unique(label)}
    transform = {i: Affine2D().rotate_deg(rotation[i])
                 for i in np.unique(label)}

    # Set the marker style list
    # If single, fullstyle=True, if double, fullstyle=False
    marker = [MarkerStyle('o', 'full', transform[i]) if '+' not in i
              else MarkerStyle('o', 'left', transform[i]) for i in label]

    # Split the label, each element is a list of phases, len=1 or 2
    split_label = [i.split('+') for i in label]

    label_len = [len(i) for i in split_label]
    if any(i > 2 for i in label_len):
        raise NotImplementedError("Multiple phases are not implemented.")

    facecolor = to_rgba_array(np.array([color[i[0]] for i in split_label]))
    facecoloralt = to_rgba_array(np.array([color[i[-1]] for i in split_label]))

    points = []
    for i in range(len(car_coords)):
        point = ax.plot(
            car_coords[i, 0], car_coords[i, 1], car_coords[i, 2],
            marker=marker[i], markersize=markersize,
            markerfacecolor=facecolor[i],
            markerfacecoloralt=facecoloralt[i],
            markeredgecolor='#F7F7F7',
            markeredgewidth=0,
            picker=picker)
        points.append(point)
    return points


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


def label_modify(
        label: NDArray,
        file_name: str,
        index_map: NDArray[np.int_] | None = None) -> NDArray:
    """Modify the label based on a json file.
    The json file contains key-value pairs of 3 kinds:
    1. "a": [j1, j2, j3, j4] to tansform the label of j1-j4 to a.
    2. "b": ["a", "c"] to change the phase label name of phase a and c to b.
    Please make sure that all case 1 is before case 2.

    Args:
        label (NDArray): The label of each point.
        file_name (str): The json file name.
        index_map (NDArray[np.int_] | None, optional): The index mapping
            for the label. Defaults to None. If None, the index mapping
            will be the same as the label.

    Returns:
        NDArray: The modified label.
    """
    if not file_name.endswith('.json'):
        raise ValueError("The file name should end with '.json'.")
    # Set the dtype of label to str
    label = label.astype('<U100')
    with open(file_name, 'r', encoding='utf-8') as f:
        modify = json.load(f)
    for key, value in modify.items():
        if not isinstance(value, list):
            raise ValueError("The value should be a list.")
        # Case 1
        if isinstance(value[0], int):
            if index_map is not None:
                transform_value = index_map[value]
            else:
                transform_value = value
            label[transform_value] = key
        # Case 2
        if isinstance(value[0], str):
            for i in value:
                label[label == i] = key
    return label


def _single_scatter(
        ax: Axes3D,
        color: dict[str, str],
        label: NDArray[np.str_],
        car_coords: NDArray[np.float64],
        interactive: bool):
    markercolor = [color[i] for i in label]
    ax.scatter(
        car_coords[:, 0], car_coords[:, 1], car_coords[:, 2],  # type: ignore
        c=markercolor, s=25, picker=interactive)
    ax.legend([Line2D([0], [0], marker='o', color='w', label=i,
                      markerfacecolor=color[i], markersize=10)
               for i in color], list(color.keys()),
              loc='center left', bbox_to_anchor=(0.7, 0.8))


def plot_quat_scatter(
        label: list[str | int] | NDArray,  # type: ignore
        axis_label: tuple[str, str, str, str],
        xrd_database: XRDDatabase | None = None,
        cif_database: CIFDatabase | None = None,
        xrf_database: XRFDatabase | None = None,
        coord: list[tuple[float, float, float, float]] | NDArray | None = None,  # type: ignore
        color: dict[str, str] | None = None,
        tick_number: int = 5,
        path_type: Literal['normal', 'snakelike'] = 'normal',
        composition_type: Literal['atomic', 'volumetric'] = 'atomic',
        interactive: bool = False,
        ax: Axes3D | None = None,
        json_path: str = '',
        ylim: tuple[float | int, float | int] | None = None,
        rotate90: int = 0,
        **kwargs) -> None:
    """Plot the scatter plot of quaternary phase diagram.

    Args:
        label (list[str | int] | NDArray): The label of each point. The label should be
            a 1D list or 1D array.
        xrd_database (XRDDatabase | None, optional): The XRDDatabase for coordinate
            generation. Defaults to None.
        cif_database (CIFDatabase | None, optional): The CIFDatabase for interactive mode.
            Defaults to None.
        xrf_database (XRFDatabase | None, optional): The XRFDatabase for XRF based
            coordinate generation. Defaults to None.
        coord (list[tuple[float, float, float, float]] | NDArray | None, optional):
            The quaternary coordinate for each point. Defaults to None. If None, the
            coordinate will be generated from either xrd_database or xrf_database.
            If xrf_database is provided, the coordinate will be generated from XRF.
        tick_number (int, optional): The number of ticks on each side of the quaternary plot.
            Defaults to 5.
        path_type (Literal[&#39;normal&#39;, &#39;snakelike&#39;], optional):
            For normal type, the points of odd and even lines are in the same order.
            For snakelike type, the odd and even lines are in the opposite order.
            Defaults to 'normal'.
            For example
            - normal: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
            - snakelike: [[0, 1, 2, 3], [7, 6, 5, 4], [8, 9, 10, 11], [15, 14, 13, 12]]
        composition_type (Literal[&#39;atomic&#39;, &#39;volumetric&#39;], optional):
            The composition to display. Defaults to 'atomic'.
        interactive (bool, optional): Defaults to False. In the interactive mode,
            you can double left click on a point to see the XRD pattern of the specific point.
            You can also double right click to refresh the scatter plot with new labels.
        ax (Axes3D | None, optional): Defaults to None.
        json_path (str, optional): The modifier file for plot refresh in interactive mode.
            Defaults to 'modifier.json'.
        ylim (tuple[float | int, float | int] | None, optional): The ylim of the left click
            displayed pattern in interactive mode. Defaults to None. If None, the ylim will
            be automatically set by matplotlib.
        rotate_label (int, optional): The times to rotate 90 degrees for the labels.
            Defaults to 0. The rotate direction is depending on the sequence of axis_label
            and xrd_database.info.elements, which is current a bug.
        **kwargs: Additional keyword arguments for the scatter plot.

    Raises:
        ValueError: Either XRD or XRF database should be provided.
        ValueError: XRDDatabase should be provided for interactive mode.
    """
    # Check if the ax is None
    if ax is None or 'fig' not in kwargs:
        fig = plt.figure(figsize=(8, 6))
        kwargs['fig'] = fig
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
    else:
        # This is for interactive mode
        # To pass fig into event functions
        fig = kwargs.get('fig')
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes3D)

    # Process coordinates from database
    if coord is None:  # Generate coordinates from database
        if xrd_database is not None and xrf_database is None:
            coord = get_coord(
                element_order=axis_label,
                side_number=int(np.sqrt(len(label))),
                info=xrd_database.info,
                composition_type=composition_type)
        elif xrf_database is not None:
            coord = get_coord(
                element_order=axis_label,
                side_number=int(np.sqrt(len(label))),
                xrf_database=xrf_database,
                composition_type=composition_type)
        else:
            raise ValueError("Either XRD or XRF database should be provided.")
    if coord is not None:  # Use the provided coordinates
        coord = np.array(coord)
    assert isinstance(coord, np.ndarray)

    # Build the tetrahedron in ax
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    build_tetrahedron(
        ax, vertices, axis_label, tick_number,
        fontfamily=kwargs.get('fontfamily', 'DejaVu Sans'))

    side_num = int(np.sqrt(len(coord)))

    # If in snakelike mode, the index of the picked point needs to be converted
    index_map = np.arange(len(label))
    if path_type == 'snakelike':
        index_map = index_map.reshape(side_num, side_num)
        index_map[1::2] = index_map[1::2, ::-1]
        index_map = index_map.flatten()

    # Convert the label to NDArray[str] to avoid type error
    label = np.array(label).astype(str)

    # Normal path to snake like path transition if required
    if path_type == 'snakelike':
        label = label[index_map]

    # Rotate the label
    if rotate90 != 0:
        coord = np.rot90(
            coord.reshape(side_num, side_num, 4), rotate90, (0, 1)).reshape(-1, 4)

    # The original coords are in tetrahedron coordinates
    # Convert to cartesian coordinates
    car_coords = tet_to_car(vertices, coord)
    car_coords -= np.mean(vertices, axis=0)

    # Modify the label based on the json file
    if json_path:
        label = label_modify(label, json_path, index_map)

    # Color: {'phase_name': '#RRGGBBAA'}
    if color is None:
        cmap = _pick_cmap(len(np.unique(label)))
        color = {}
        for i, v in enumerate(np.unique(label)):
            color[v] = cmap[i]

    # Plot scatters
    _single_scatter(ax, color, label, car_coords, interactive)

    # Interactive mode
    if interactive:
        if xrd_database is None:
            raise ValueError("The xrd_database should be provided for interactive mode.")

        def _onpick(event):
            """Scatter plot pick event."""
            if not (event.mouseevent.dblclick and event.mouseevent.button == 1):
                return
            # Make a copy of the kwargs
            # Avoid modifying the original kwargs
            kw = kwargs.copy()
            kw.pop('fig', None)
            # Get the index of the selected point
            index = index_map[event.ind]
            if len(index) > 1:
                print("Multiple points selected. Zoom in to select a single point.")
                return
            # Get the xrd pattern of the specific point
            pattern = xrd_database.data[index[0]]
            # Plot the xrd pattern of the specific point
            if cif_database is None:
                _, ax = plt.subplots()
                pattern.plot(ax=ax, max_intensity=xrd_database.intensity.max(), **kw)
                ax.set_title(f"{pattern.info.name}-{pattern.info.index}")
                if ylim:
                    ax.set_ylim(*ylim)
                plt.show()
            else:
                pattern.plot_with_ref(
                    cif_database=cif_database,
                    title=f"{pattern.info.name}-{pattern.info.index}",
                    max_intensity=xrd_database.intensity.max(), ylim=ylim, **kw)

        def _refresh(event):
            """Read json file and refresh the scatter with new labels."""
            # Detect double right click
            if not (event.dblclick and event.button) == 3:
                return
            # Read the json file
            if not json_path:
                return
            new_label = label_modify(label, json_path, index_map)
            # Detect if new labels are added, generate color for them
            new_phase = np.unique(new_label)
            cmap = _pick_cmap(len(new_phase))
            for index, phase in enumerate(new_phase):
                if phase not in color:
                    color[phase] = cmap[index]
            # If a color is not used, remove it
            for i in color.copy():
                if i not in new_phase:
                    color.pop(i)
            # Clear the current scatter
            ax.clear()
            # Redraw the scatter with new labels
            build_tetrahedron(
                ax, vertices, axis_label, tick_number,
                fontfamily=kwargs.get('fontfamily', 'DejaVu Sans'))
            _single_scatter(ax, color, new_label, car_coords, interactive)
            # Refresh the plot
            plt.tight_layout()
            plt.draw()

        fig.canvas.mpl_connect('pick_event', _onpick)
        fig.canvas.mpl_connect('button_press_event', _refresh)
    plt.tight_layout()
    plt.show()


def _pick_cmap(number: int) -> list[str]:
    """Pick a colormap based on the number of phases.

    - <= 10: tab10
    - 10 < x <= 20: tab20
    - > 20: viridis (change to discrete colors)

    Args:
        number (int): The number of phases.

    Returns:
        list[str]: The list of hex colors.
    """
    if number <= 10:
        cmap = plt.cm.get_cmap('tab10')
        result = [to_hex(cmap(i), keep_alpha=True) for i in range(number)]
    elif 10 < number <= 20:
        cmap = plt.cm.get_cmap('tab20')
        result = [to_hex(cmap(i), keep_alpha=True) for i in range(number)]
    else:
        cmap = plt.cm.get_cmap('viridis')
        result = [to_hex(cmap(i / number), keep_alpha=True) for i in range(number)]
    return result
