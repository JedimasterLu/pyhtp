# -*- coding: utf-8 -*-
'''
Define the utility functions for quaternary phase diagram.
'''
from typing import Literal
import numpy as np
import periodictable
from numpy.typing import NDArray
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ..typing import SampleInfo
from ..xrf import XRFDatabase


def get_coord(
        element_order: tuple[str, str, str, str],
        side_number: int,
        composition_type: Literal['atomic', 'volumetric'] = 'atomic',
        xrf_database: XRFDatabase | None = None,
        info: SampleInfo | None = None) -> NDArray:
    """Get the compositional coordinates of each point.

    Either info or xrf_database should be provided.

    Args:
        element_order (tuple[str, str, str, str]): The order of elements in the database.
        side_number (int): The number of points in each side.
        info (SampleInfo): The information of the sample.
        composition_type (Literal['atomic', 'volumetric'], optional):
            The type of composition. Defaults to 'atomic'.
        xrf_database (XRFDatabase, optional): The XRF database. Defaults to None.

    Raises:
        ValueError: Raise error if the type is invalid.

    Returns:
        np.ndarray: The compositional coordinates of each point.
    """
    if (info is None and xrf_database is None) \
            or (info is not None and xrf_database is not None):
        raise ValueError("Either info or xrf_database should be provided.")
    if info:
        return _get_coord_from_thickness(
            element_order, side_number, info, composition_type)
    if xrf_database:
        return _get_coord_from_xrf(
            element_order, side_number, xrf_database, composition_type)
    raise ValueError("Invalid type of input.")


def _get_coord_from_thickness(
        element_order: tuple[str, str, str, str],
        side_number: int,
        info: SampleInfo,
        composition_type: Literal['atomic', 'volumetric'] = 'atomic') -> NDArray:
    """Get the compositional coordinates of each point from thickness."""
    # Make sure the element in database.info.element is the same to the label
    if set(info.element) != set(element_order):
        raise ValueError("The elements in database.info should be the same to the label.")
    # Film thickness is necessary for coordinates
    if info.film_thickness is None:
        raise ValueError("The film thickness is not provided in Sampleinfo.")
    # x and y coordinates,from 0 to 1, equivalent to:
    # abs_coord = np.zeros((side_num ** 2, 2))
    # for i in range(side_num):
    #     for j in range(side_num):
    #         x = 1 / (side_num - 1) * i
    #         y = 1 / (side_num - 1) * j
    #         abs_coord[i * side_num + j] = [x, y]
    i, j = np.meshgrid(np.arange(side_number), np.arange(side_number), indexing='ij')
    # Normalize the coordinates
    abs_coord = np.stack((i / (side_number - 1),
                          j / (side_number - 1)), axis=-1).reshape(-1, 2)
    # Thickness of each layer as the function of x and y
    # Layer 1: max at the top boundary
    # Layer 2: max at the right boundary
    # Layer 3: max at the bottom boundary
    # Layer 4: max at the left boundary
    # Volumetric composition of each point
    point_thickness = np.stack((info.film_thickness[0] * abs_coord[:, 1],
                                info.film_thickness[1] * abs_coord[:, 0],
                                info.film_thickness[2] * (1 - abs_coord[:, 1]),
                                info.film_thickness[3] * (1 - abs_coord[:, 0])), axis=1)
    if composition_type == "volumetric":
        # Normalize the volumetric composition
        comp = point_thickness / np.sum(point_thickness, axis=1)[:, None]
    if composition_type == "atomic":
        density = np.array(
            [periodictable.elements.symbol(element).density  # type: ignore
             for element in info.element])
        atomic_weight = np.array(
            [periodictable.elements.symbol(element).mass  # type: ignore
             for element in info.element])
        comp = point_thickness * density / atomic_weight
        # Normalize the atomic composition
        comp /= np.sum(comp, axis=1)[:, None]
    # Change the columns of comp from original database.info.element to label
    label_dict = {info.element[i]: comp[:, i] for i in range(4)}
    comp = np.stack([label_dict[l] for l in element_order], axis=1)  # noqa=E741
    return comp


def _get_coord_from_xrf(
        element_order: tuple[str, str, str, str],
        side_number: int,
        xrf_database: XRFDatabase,
        composition_type: Literal['atomic', 'volumetric'] = 'atomic') -> NDArray:
    """Get the compositional coordinates of each point from XRF database."""
    # Get the coordinates from the XRF database
    comp = xrf_database.get_composition_map(
        side_number=side_number, order=element_order)
    if composition_type == "atomic":
        density = np.array(
            [periodictable.elements.symbol(element).density  # type: ignore
             for element in element_order])
        atomic_weight = np.array(
            [periodictable.elements.symbol(element).mass  # type: ignore
             for element in element_order])
        comp = comp * density / atomic_weight
        # Normalize the atomic composition
        comp /= np.sum(comp, axis=1)[:, None]
    return comp


def tet_to_car(
        vertices: NDArray,
        coord: NDArray) -> NDArray:
    """Convert the tetrahedral coordinates to Cartesian coordinates."""
    return np.dot(coord, vertices)


def build_tetrahedron(
        ax: Axes3D,
        vertices: NDArray,
        axis_label: tuple[str, str, str, str],
        tick_number: int,
        fontfamily: str) -> None:
    """Build a tetrahedron axis in 3D plot."""
    # Construct tetrahedron
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    # Move the center of the tetrahedron to the origin
    vertices -= np.mean(vertices, axis=0)
    # Add the tetrahedron to the plot
    faces = [[vertices[j] for j in [0, 1, 2]],
             [vertices[j] for j in [0, 1, 3]],
             [vertices[j] for j in [0, 2, 3]],
             [vertices[j] for j in [1, 2, 3]]]
    ax.add_collection3d(
        Poly3DCollection(faces, alpha=0.05, linewidths=1,
                         edgecolors='darkslategray'))
    # Add labels to the vertices
    pad = 0.08
    for i, txt in enumerate(axis_label):
        if i == 0:
            coord = vertices[i] + np.array([-np.sqrt(3) * pad / 2, - pad / 2, - pad / 2])
        elif i == 1:
            coord = vertices[i] + np.array([np.sqrt(3) * pad / 2, - pad / 2, - pad / 2])
        elif i == 2:
            coord = vertices[i] + np.array([0, pad, - pad / 2])
        else:
            coord = vertices[i] + np.array([0, 0, pad / 2])
        ax.text(coord[0], coord[1], coord[2], txt, size=18,
                color='darkslategray', fontfamily=fontfamily)
    # Set aspect and lims and axes
    ax.set_box_aspect([1, 1, 1])
    lim_value = 0.38
    ax.set_xlim(-lim_value, lim_value)
    ax.set_ylim(-lim_value, lim_value)
    ax.set_zlim(-lim_value, lim_value)
    ax.set_axis_off()
    ax.set_proj_type('persp')
    # Add ticks on the edges
    for start, end in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        for i in range(1, tick_number):
            tick = vertices[start] + (vertices[end] - vertices[start]) * i / tick_number
            ax.plot([tick[0]], [tick[1]], [tick[2]], 'ko', markersize=2)


def get_chemical_formula(
        element_order: list[str],
        coord: NDArray | list[float],
        if_subscript: bool = True) -> str:
    """Get the chemical formula of each point.

    Args:
        element_order (tuple[str, str, str, str]): The order of elements in the database.
        coord (np.ndarray): The compositional coordinates of each point.
        if_subscript (bool, optional): If the subscript is needed. Defaults to True.
            If true, the number will be subscripted.
    """
    # Check if the length of element_order and coord is the same
    if len(element_order) != len(coord):
        raise ValueError("The length of element_order and coord should be the same.")
    if not if_subscript:
        return ''.join([f"{element_order[i]}{round(coord[i], 2)}"
                        for i in range(len(element_order))])
    return ''.join([f"{element_order[i]}$_{{{round(coord[i], 2)}}}$"
                    for i in range(len(element_order))])
