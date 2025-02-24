# -*- coding: utf-8 -*-
'''
Define the scatter plot for ternary phase diagram.
'''
import io
from typing import Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from ternary import TernaryAxesSubplot
from pyhtp.xrd import XRDDatabase


def _ter_to_car(point: tuple[float, float, float]) -> tuple[float, float]:
    '''This function converts the ternary coordinates to cartesian coordinates.

    Args:
        point (tuple[float, float, float]): _description_

    Returns:
        tuple[float, float]: _description_
    '''
    x = 0.5 * (2 * point[0] + point[1]) / (point[0] + point[1] + point[2]) * 100
    y = 0.5 * 3 ** 0.5 * point[1] / (point[0] + point[1] + point[2]) * 100
    return (x, y)


def _get_points_position(
        width: int,
        path_type: Literal['normal', 'snakelike'] = 'normal') -> list[tuple[float, float, float]]:
    '''_get_points_position returns the position of points in ternary diagram.

    Args:
        points_number (int): The number of points in the diagram.
        path_type (Literal['normal', 'snakelike']): The path type of the diagram.

    Returns:
        list[tuple[float, float, float]]: The position of points in the diagram. In snake form from bottom to top.
    '''
    # Generate the points, each point position is defined by a tuple contaning 3 int
    points = []
    # Define the margin between points by width
    margin = 100 / (width - 1)
    # Generate the points in snake form from bottom to top
    for row in range(width):
        # Find the middle point of that row
        if (width - row) % 2 == 0:
            middle = int((width - row) / 2) - 1
        else:
            middle = (width - row) // 2
        y = row * margin
        middle_value = 50 - y / 2
        # Generate the points in that row
        if row % 2 == 0:
            for column in range(width - row):
                if (width - row) % 2 != 0:
                    x = middle_value - (middle - column) * margin
                else:
                    x = middle_value - (middle - column) * margin - margin / 2
                z = 100 - x - y
                points.append((x, y, z))
        elif row % 2 != 0 and path_type == 'snakelike':
            # if the row is even, left to right; if odd, right to left
            for column in range(width - row - 1, -1, -1):
                if (width - row) % 2 != 0:
                    x = middle_value - (middle - column) * margin
                else:
                    if column < middle:
                        x = middle_value - (middle - column) * margin + margin / 2
                    else:
                        x = middle_value - (middle - column) * margin - margin / 2
                z = 100 - x - y
                points.append((x, y, z))
        else:
            for column in range(width - row):
                if (width - row) % 2 != 0:
                    x = middle_value - (middle - column) * margin
                else:
                    if column < middle:
                        x = middle_value - (middle - column) * margin + margin / 2
                    else:
                        x = middle_value - (middle - column) * margin - margin / 2
                z = 100 - x - y
                points.append((x, y, z))
    return points


def _rotate_mapping(width: int) -> list[int]:
    """_summary_

    Args:
        width (int): _description_

    Returns:
        list[int]: _description_
    """
    mapping = []
    for row in range(width):
        current_width = width - row
        # Extend the index of the right row into mapping
        for column in range(current_width):
            mapping.append(current_width - 1 + column * (width - 1 + width - column) // 2)
    return mapping


def _get_width(index: list) -> int:
    '''_get_width return the width of the ternary diagram.

    Args:
        index (list): The index of the points.

    Returns:
        int: The width of the diagram.

    Raises:
        ValueError: The index length is not a triangle number!
    '''
    # Set the initial value as the approximate value of the width by solving width * (width + 1) / 2 == length
    length = len(index)
    width = (-1 + (1 + 8 * length) ** 0.5) / 2
    width = int(width) - 2
    # check if there exist a width such that width * (width + 1) / 2 == length, if not, raise error
    while width * (width + 1) / 2 < length:
        width += 1
    if width * (width + 1) / 2 != length:
        raise ValueError('The index length is not a triangle number!')
    return width


def scatter_ternary(
        value: list[str],  # type: ignore
        color: dict[str, str],
        label: tuple[str, str, str],
        coord: Optional[list[tuple[float, float, float]]] = None,  # type: ignore
        rotation: Optional[dict[str, float]] = None,
        path_type: Literal['normal', 'snakelike'] = 'normal',
        ax: Optional[Axes] = None) -> TernaryAxesSubplot:
    """Plot the ternary diagram with given data

    Args:
        data (list[str]): The data of each points.
        color (dict[str, str]): Color for each structure.
        label (tuple[str, str, str]): Labels of the three axis.
        coord (Optional[list[tuple[float, float, float]]], optional): The coordinates of each point. Defaults to None.
        rotation (Optional[dict[str, float]], optional): Rotation for each class. Defaults to None.
        path_type (Literal['normal', 'snakelike'], optional): The path type. Defaults to 'normal'.
        ax (Optional[Axes], optional): The axis. Defaults to None.

    Returns:
        Axes: The axis of the plot.
    """
    def _bi_scatter(
            tax: TernaryAxesSubplot,
            points: list[tuple[float, float, float]],
            left_color: str,
            right_color: str,
            marker_size: int,
            rotation: float=0) -> None:
        '''Plot scatters with pie markers that has 2 colors'''
        t = Affine2D().rotate_deg(rotation)
        filled_marker_style = {
            'marker': MarkerStyle('o', 'left', t),
            'markersize': marker_size,
            'markerfacecolor': left_color,
            'markerfacecoloralt': right_color,
            'markeredgecolor': '#F7F7F7',
            'markeredgewidth': 0}
        tax.plot(points, linewidth=0, **filled_marker_style)

    def _multi_scatter(
            ax: Axes,
            points: list[tuple[float, float, float]],
            color: list[str],
            marker_size: int,
            rotation: float=0) -> None:
        """Plot scatters with pie markers that has more than 2 colors"""
        # Convert points to cartisian coordinates
        car_points = [_ter_to_car(point) for point in points]

        # Generate pie image
        slice_num = len(color)
        sizes = np.ones(slice_num) / slice_num
        explode = (0, 0, 0)
        fig, ax0 = plt.subplots()
        ax0.pie(sizes, explode=explode, colors=color,
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

    # Set the figure
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if ax is None:
        raise ValueError('The ax is not correctly defined!')
    if coord is None:
        coord: list[tuple[float, float, float]] = \
            _get_points_position(_get_width(value), path_type)
    # If snake like, modify the data
    value: NDArray = np.array(value)
    # Set the axis
    tax = TernaryAxesSubplot(ax=ax, scale=100)
    # Plot the scatter of each phase
    for phase in list(set(value)):
        coo = [coord[i] for i in range(len(value)) if value[i] == phase]
        if '+' in phase:
            sep_phase: list[str] = phase.split('+')
            sep_phase = [i.strip() for i in sep_phase]
            rot = 0
            if rotation is not None and phase in rotation:
                rot = rotation[phase]
            if len(sep_phase) == 2:
                _bi_scatter(tax, coo, color[sep_phase[0]], color[sep_phase[1]], 8, rot)
            if len(sep_phase) > 2:
                current_color = [color[phase] for phase in phase]
                _multi_scatter(ax, coo, current_color, 8, rot)
        else:
            tax.scatter(coo, color=color[phase], s=40)
    # Set legends
    legend_elements = []
    for lab, color_name in color.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      label=lab, markerfacecolor=color_name,
                                      markersize=15))
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), handles=legend_elements, fontsize=16)
    # Set the axis and labels
    tax.ticks(axis='lbr', linewidth=1, multiple=25, fontsize=12, offset=0.03)
    tax.bottom_axis_label(label[0], fontsize=16, offset=0.20, fontweight='bold')
    tax.right_axis_label(label[1], fontsize=16, offset=0.20, fontweight='bold')
    tax.left_axis_label(label[2], fontsize=16, offset=0.20, fontweight='bold')
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    return tax


def rotate_data(data: list[str]) -> list[str]:
    '''_rotate_phase_index rotate the whole ternary diagram by 60 degrees counter-clockwise.

    Args:
        width (int): The width of the diagram.
        phase_index (list[int]): Define the phase of each point.

    Returns:
        list[int]: _description_
    '''
    def _snake_to_serial(width: int, data: list[str]) -> list[str]:
        '''Convert the snake form index to serial form index.'''
        serial_phase_index = data.copy()
        # Reverse the index of even rows
        for row in range(width):
            if row % 2 != 0:
                start_index = (row + 1) * (width + width - row) // 2 - 1
                end_index = start_index - width + row + 1
                for column in range(end_index, start_index + 1):
                    serial_phase_index[column] = data[start_index - column + end_index]
        return serial_phase_index
    # Create mapping relationship of rotation
    mapping = []
    width = _get_width(data)
    for row in range(width):
        current_width = width - row
        # Extend the index of the right row into mapping
        for column in range(current_width):
            mapping.append(current_width - 1 + column * (width - 1 + width - column) // 2)
    # Map the phase index
    data = _snake_to_serial(width, data)
    rotated_data = [data[mapping[i]] for i in range(len(data))]
    rotated_data = [rotated_data[mapping[i]] for i in range(len(data))]
    rotated_data = _snake_to_serial(width, rotated_data)
    return rotated_data


def pattern_on_ternary_line(
        xrd_data: XRDDatabase,
        start_point: tuple[float, float, float],
        end_point: tuple[float, float, float],
        detect_radius: float = -1,
        rotate_times: int = 0,
        ax: Optional[Axes] = None,
        **kwargs) -> None:
    """_summary_

    Args:
        xrd_data (XrdDatabase): _description_
        start_point (tuple[float, float, float]): _description_
        end_point (tuple[float, float, float]): _description_
        detect_radius (float, optional): _description_. Defaults to -1.
        rotate_times (int, optional): _description_. Defaults to 0.
        **kwargs: args for XrdDatabase.plot

    """
    def _distance_to_line(point: tuple[float, float, float],
                          start_point: tuple[float, float, float],
                          end_point: tuple[float, float, float]) -> float:
        """This function calculates the distance of a point to a line."""
        # Convert the ternary coordinates to cartesian coordinates
        point_cartesian = _ter_to_car(point)
        start_point_cartesian = _ter_to_car(start_point)
        end_point_cartesian = _ter_to_car(end_point)
        # Calculate the distance
        distance = abs((end_point_cartesian[1] - start_point_cartesian[1]) * point_cartesian[0]
                       - (end_point_cartesian[0] - start_point_cartesian[0]) * point_cartesian[1]
                       + end_point_cartesian[0] * start_point_cartesian[1]
                       - end_point_cartesian[1] * start_point_cartesian[0]) \
            / ((end_point_cartesian[1] - start_point_cartesian[1]) ** 2
               + (end_point_cartesian[0] - start_point_cartesian[0]) ** 2) ** 0.5
        return distance

    def _distance_between_points(point1: tuple[float, float, float],
                                 point2: tuple[float, float, float]) -> float:
        """This function calculates the distance between two points."""
        # Convert the ternary coordinates to cartesian coordinates
        point1_cartesian = _ter_to_car(point1)
        point2_cartesian = _ter_to_car(point2)
        # Calculate the distance
        distance = ((point1_cartesian[0] - point2_cartesian[0]) ** 2
                    + (point1_cartesian[1] - point2_cartesian[1]) ** 2) ** 0.5
        return distance

    if ax is None:
        _, ax = plt.subplots()
    # Get the index of the xrd data points on the line
    # If the distance of a point to the line is less than detect_radius, it is considered on the line
    width = _get_width(xrd_data.data)
    coordinates = _get_points_position(width)
    # Decide detect radius based on the distance between two points
    if detect_radius == -1:
        detect_radius = 0.5 * 3 ** 0.5 * 100 / width
    index_to_plot = []
    for index, coo in enumerate(coordinates):
        if _distance_to_line(coo, start_point, end_point) < detect_radius:
            index_to_plot.append(index)
    # Sort the index_to_plot based on the distance to the start point
    index_to_plot.sort(key=lambda x: _distance_between_points(coordinates[x], start_point))
    # Rotate the phase index if needed
    mapping = _rotate_mapping(width)
    for _ in range(rotate_times):
        index_to_plot = [mapping[i] for i in index_to_plot]
    # Plot the xrd data
    ax = xrd_data.plot(index_to_plot, style='stack', ax=ax, **kwargs)
