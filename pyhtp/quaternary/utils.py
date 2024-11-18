# -*- coding: utf-8 -*-
'''
Define the utility functions for quaternary phase diagram.
'''
from typing import Literal
import numpy as np
import periodictable
from numpy.typing import NDArray
from ..typing import SampleInfo


def get_coord(
        label: tuple[str, str, str, str],
        side_num: int,
        info: SampleInfo,
        composition_type: Literal['atomic', 'volumetric'] = 'atomic') -> NDArray:
    """Get the compositional coordinates of each point.

    Args:
        composition_type (Literal['atomic', 'volumetric'], optional): The type of composition. Defaults to 'atomic'.

    Raises:
        ValueError: Raise error if the type is invalid.

    Returns:
        np.ndarray: The compositional coordinates of each point.
    """
    # Make sure the element in database.info.element is the same to the label
    if set(info.element) != set(label):
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
    i, j = np.meshgrid(np.arange(side_num), np.arange(side_num), indexing='ij')
    # Normalize the coordinates
    abs_coord = np.stack((i / (side_num - 1),
                          j / (side_num - 1)), axis=-1).reshape(-1, 2)
    # Thickness of each layer as the function of x and y
    # Layer 1: max at the top boundary
    # Layer 2: max at the right boundary
    # Layer 3: max at the bottom boundary
    # Layer 4: max at the left boundary
    # Volumetric composition of each point
    point_thickness = np.stack((info.film_thickness[0] * abs_coord[:, 0],
                                info.film_thickness[1] * abs_coord[:, 1],
                                info.film_thickness[2] * (1 - abs_coord[:, 0]),
                                info.film_thickness[3] * (1 - abs_coord[:, 1])), axis=1)
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
    comp = np.stack([label_dict[l] for l in label], axis=1)  # noqa=E741
    return comp


def tet_to_car(
        vertices: NDArray,
        coord: NDArray) -> NDArray:
    """Convert the tetrahedral coordinates to Cartesian coordinates."""
    return np.dot(coord, vertices)
