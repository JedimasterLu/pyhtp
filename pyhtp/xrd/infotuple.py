# -*- coding: utf-8 -*-
"""
Define the NamedTuple for the information of a diffraction pattern.
"""
from __future__ import annotations
from typing import NamedTuple, Union
from pymatgen.core.structure import Structure


class AngleRange(NamedTuple):
    """A NamedTuple that contains the information of a mask area.

    Args:
        NamedTuple (_type_): The left and right angle of the mask area.
    """
    left: float
    right: float


class SampleInfo(NamedTuple):
    """A NamedTuple that contains the information of a sample.

    Args:
        NamedTuple (_type_): The name, formula, and description of the sample.
    """
    name: str
    angle_range: Union[AngleRange, list[AngleRange]]
    element: list[str]
    temperature: float = 25.0
    description: str = ''


class PatternInfo(NamedTuple):
    """A NamedTuple that contains the information of a diffraction pattern.

    Args:
        NamedTuple (_type_): The name, index, elements, and temperature of the pattern.
    """
    name: str = 'default'
    index: int = 0
    angle_range: AngleRange = AngleRange(left=0, right=90)
    element: list[str] = []
    temperature: float = 25.0


class Latticeabc(NamedTuple):
    """A NamedTuple that contains the information of a lattice.

    Args:
        NamedTuple (_type_): The a, b, and c lattice parameters of the lattice.
    """
    a: float
    b: float
    c: float


class Latticeangles(NamedTuple):
    """A NamedTuple that contains the information of a lattice.

    Args:
        NamedTuple (_type_): The alpha, beta, and gamma lattice angles of the lattice.
    """
    alpha: float
    beta: float
    gamma: float


class MillerIndice(NamedTuple):
    """A NamedTuple that contains the information of a Miller indice.

    Args:
        NamedTuple (_type_): The h, k, and l of the Miller indice.
    """
    h: int
    k: int
    l: int


class IcsdData(NamedTuple):
    """A NamedTuple that contains the information of a ICSD data.

    Args:
        NamedTuple (_type_): The ICSD code, space group, space group number, and formula of the data.
    """
    name: str
    two_theta: list[float]
    intensity: list[float]
    hkl: list[MillerIndice]
    space_group: str
    space_group_number: int
    formula: str
    lattice_abc: Latticeabc
    lattice_angles: Latticeangles
    icsd_code: int
    structure: Structure
