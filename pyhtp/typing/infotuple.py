# -*- coding: utf-8 -*-
"""
Define the NamedTuple for the information of a diffraction pattern.
"""
from __future__ import annotations
from typing import NamedTuple
from pymatgen.core.structure import Structure


class AngleRange(NamedTuple):
    """A NamedTuple that contains the information of an angle range

    Args:
        left (float): The left boundary of the angle range.
        right (float): The right boundary of the angle range.
    """
    left: float
    right: float


class SampleInfo(NamedTuple):
    """A NamedTuple that contains the information of a sample.

    Args:
        name (str): The name of the sample.
        element (list[str]): The elements in the sample. If the sample is a multilayer one,
            please enter the elements in the order of the layers.
        film_thickness (list[float]): The film thickness of the sample. Similar to
            elements, please enter the thicknesses in the order of the layers. If some layers
            contain multiple elements, please calculate the volmetric equivalent thickness of
            each element and enter them in this list, according to the order of the elements.
            The film_thickness for XRD and SE are all necessary.
        point_number (int): The number of points in the sample. Default is -1. This value will
            be automatically set when the data is loaded.
        two_theta_range (AngleRange | list[AngleRange] | None): The angle range of the sample.
            For XRD analysis, two_theta_range must be set. Also, the length of the list[AngleRange]
            should be the same to number of directories for .xy data. Please note that the angle
            range should be next to each other, leading to a continuous range. Default to None.
        wavelength_range (tuple[float, float] | None): The wavelength range of the sample. For SE
            analysis, wavelength_range must be set. Default to None.
        temperature (float | tuple[float, float]): The temperature of the sample. For XRD analysis,
            only one temperature is needed. For SE analysis, two temperatures entry is an option.
    """
    name: str
    element: list[str]
    film_thickness: list[float]
    point_number: int = -1
    two_theta_range: AngleRange | list[AngleRange] | None = None
    wavelength_range: tuple[float, float] | None = None
    temperature: float = 25.0


class PatternInfo(NamedTuple):
    """A NamedTuple that contains the information of a diffraction pattern.

    Args:
        name (str): The name of the diffraction pattern.
        two_theta_range (AngleRange): The angle range of the diffraction pattern.
        element (list[str]): The elements in the diffraction pattern.
        temperature (float): The temperature of the diffraction pattern.
        index (int): The index of the diffraction pattern. Default to 0. The index is used
            in XRDDatabase to identify the diffraction pattern.
    """
    name: str
    two_theta_range: AngleRange
    element: list[str]
    temperature: float
    index: int = 0


class SpectrumInfo(NamedTuple):
    """A NamedTuple that contains the information of a spectroscopic ellipsometry (SE) spectrum.

    Args:
        name (str): The name of the SE spectrum.
        wavelength_range (tuple[float, float]): The wavelength range of the SE spectrum.
        element (list[str]): The elements in the sample.
        temperature (float): The temperature of the SE spectrum.
        index (int): The index of the SE spectrum. Default to 0. The index is used
            in EllipDatabase to identify the SE spectrum.
    """
    name: str
    wavelength_range: tuple[float, float]
    element: list[str]
    temperature: float
    index: int = 0


class LatticeParam(NamedTuple):
    """A NamedTuple that contains the lattice parameters of a bravais lattice.

    Args:
        a (float): The length of the a side.
        b (float): The length of the b side.
        c (float): The length of the c side.
        alpha (float): The alpha angle of the lattice.
        beta (float): The beta angle of the lattice.
        gamma (float): The gamma angle of the lattice.
    """
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


class MillerIndice(NamedTuple):
    """A NamedTuple that contains the information of a Miller indice.

    Args:
        h (int): The h Miller indice.
        k (int): The k Miller indice.
        l (int): The l Miller indice.
    """
    h: int
    k: int
    l: int


class CIFData(NamedTuple):
    """A NamedTuple that contains some information of a phase from a cif file.

    Args:
        name (str): The name of the cif file without .cif.
        two_theta (list[float]): The two theta angles of the phase's diffraction peaks.
        intensity (list[float]): The intensities of the phase's diffraction peaks.
        hkl (list[MillerIndice]): The Miller indices of the phase's diffraction peaks.
        space_group (str): The space group of the phase.
        space_group_number (int): The space group number of the phase.
        formula (str): The formula of the phase, which should be in the form
            of '[element][int or float] [element][int or float] ...'.
        lattice_parameter (LatticeParam): The lattice parameters of the phase.
        icsd_code (int): The ICSD code of the phase.
        structure (Structure): The structure of the phase. It is a pymatgen Structure object.
            Please refer to https://pymatgen.org/pymatgen.core.structure.html for more information.
    """
    name: str
    two_theta: list[float]
    intensity: list[float]
    hkl: list[MillerIndice]
    space_group: str
    space_group_number: int
    formula: str
    lattice_parameter: LatticeParam
    icsd_code: int
    structure: Structure


class BandGap(NamedTuple):
    """4 band gaps of a material.

    Contains the value and error.
    The first element is the value and the second element is the error.

    Args:
        direct_allowed (tuple[float, float]):
            The value and error of the direct allowed band gap.
        direct_forbidden (tuple[float, float]):
            The value and error of the direct forbidden band gap.
        indirect_allowed (tuple[float, float]):
            The value and error of the indirect allowed band gap.
        indirect_forbidden (tuple[float, float]):
            The value and error of the indirect forbidden band gap.
    """
    direct_allowed: tuple[float, float]  # value, error
    direct_forbidden: tuple[float, float]
    indirect_allowed: tuple[float, float]
    indirect_forbidden: tuple[float, float]


class BandGapLegacy(NamedTuple):
    """4 band gaps of a material.

    Args:
        direct_allowed (float): The value of the direct allowed band gap.
        direct_forbidden (float): The value of the direct forbidden band gap.
        indirect_allowed (float): The value of the indirect allowed band gap.
        indirect_forbidden (float): The value of the indirect forbidden band gap.
    """
    direct_allowed: float  # value
    direct_forbidden: float
    indirect_allowed: float
    indirect_forbidden: float


class PeakParam(NamedTuple):
    """The parameters for finding peaks in a diffraction pattern.
    - Please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        for the meaning of each parameter.
    - Unlike the original scipy.signal.find_peaks, the units of parameters are transformed
        to percentage and angle.

    Args:
        height (float): The minimum height of the peak (percentage of the maximum intensity).
        distance (float): The minimum distance between peaks (two theta angle).
        prominence (float): The minimum prominence of the peak
            (percentage of the maximum intensity).
    """
    height: float = 0.01
    distance: float = 0.5
    prominence: float = 0.05
