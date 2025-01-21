# -*- coding: utf-8 -*-
"""
Define the NamedTuple for the information of a diffraction pattern.
"""
from __future__ import annotations
from typing import NamedTuple, Union, Optional
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
        name (str): The name of the sample.
        element (list[str]): The elements of the sample.
        film_thickness (Optional[list[float]]): The film thickness of the sample.
        angle_range (Optional[Union[AngleRange, list[AngleRange]]]): The angle range of the sample.
        temperature (float): The temperature of the sample.
        description (str): The description of the sample.
    """
    name: str
    element: list[str]
    point_number: int = -1
    film_thickness: Optional[list[float]] = None
    angle_range: Optional[Union[AngleRange, list[AngleRange]]] = None
    wavelength_range: Optional[tuple[float, float]] = None
    temperature: float | tuple[float, float] = 25.0


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


class SpectrumInfo(NamedTuple):
    """A NamedTuple that contains the information of a spectrum.

    Args:
        NamedTuple (_type_): The name, index, elements, and temperature of the spectrum.
    """
    name: str
    index: int
    wavelength_range: tuple[float, float]
    element: list[str]
    temperature: float


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


class BandGap(NamedTuple):
    """4 band gaps of a material."""
    direct_allowed: tuple[float, float]  # value, error
    direct_forbidden: tuple[float, float]
    indirect_allowed: tuple[float, float]
    indirect_forbidden: tuple[float, float]


class BandGapLegacy(NamedTuple):
    """4 band gaps of a material."""
    direct_allowed: float  # value
    direct_forbidden: float
    indirect_allowed: float
    indirect_forbidden: float


class PeakParam(NamedTuple):
    """The parameters for finding peaks in a diffraction pattern.
    - Please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html for the meaning of each parameter.
    - Unlike the original scipy.signal.find_peaks, the units of parameters are transformed to percentage and angle.

    Args:
        height (float): The minimum height of the peak (percentage of the maximum intensity).
        distance (float): The minimum distance between peaks (two theta angle).
        prominence (float): The minimum prominence of the peak (percentage of the maximum intensity).
    """
    height: float = 0.01
    distance: float = 0.5
    prominence: float = 0.05
