# -*- coding: utf-8 -*-
"""
Define a class to plot ellipsometry data in 2D or 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional
import periodictable
from pyhtp.ellip.database import EllipDatabase
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import Colorbar


class EllipPlotter:
    """A class to plot ellipsometry data in 2D or 3D."""
    def __init__(self,
                 file_path: str | list[str],
                 element: list[str],
                 thickness: list[float]):
        """Add file_path and sample information to the class.

        Args:
            file_path (str | list[str]): The path of the file. Input 2 files to calculate FOM.
            element (list[str]): The elements of each layer.
            thickness (list[float]): The maximum thickness of each layer.
        """
        self.database = EllipDatabase(file_path)
        self.element = element
        self.thickness = np.array(thickness)

    def plot_3d(self,
                param: Literal['n', 'k', 'fom'],
                wavelength: float,
                path_type: Literal['normal', 'snakelike'] = 'snakelike',
                composition_type: Literal['atomic', 'volumetric'] = 'atomic',
                structure_type: Literal['amorphous', 'crystalline'] = 'amorphous',
                num_ticks: int = 5,
                ax: Optional[Axes3D] = None,
                if_show: bool = True) -> tuple[Axes3D, Colorbar]:
        """Plot the 3D figure of the ellipsometry data.

        Args:
            param (Literal['n', 'k', 'fom']): The parameter to plot.
            wavelength (float): The wavelength of the light.
            path_type (Literal['normal', 'snakelike'], optional): Type of scan path. Defaults to 'snakelike'.
            composition_type (Literal['atomic', 'volumetric'], optional): Atomic or volumetric composition. Defaults to 'atomic'.
            structure_type (Literal['amorphous', 'crystalline'], optional): Only valid when file_path has 2 elements. Defaults to 'amorphous'.
            num_ticks (int, optional): Number of ticks on each side. Defaults to 5.
            ax (Optional[Axes3D], optional): If not given, create a new figure. Defaults to None.
            if_show (bool, optional): If show the figure. Defaults to True.

        Returns:
            Axes3D: The 3D axis of the ellipsometry data.
        """
        # Get the value of the parameter
        if self.database.get_file_num() == 2 and structure_type == 'amorphous' and param in ['n', 'k']:
            file_name = self.database.data.keys()
            # Find the index of the file with 'as' in the name
            index = [i for i, name in enumerate(file_name) if 'as' in name]
            if len(index) != 1:
                raise ValueError("One of the files should have 'as' in the name.")
            index = index[0]
            value = self.database.get_param(param, wavelength)[index]
        elif self.database.get_file_num() == 2 and structure_type == 'crystalline' and param in ['n', 'k']:
            file_name = self.database.data.keys()
            # Find the index of the file with no 'as' in the name
            index = [i for i, name in enumerate(file_name) if 'as' not in name]
            if len(index) != 1:
                raise ValueError("One of the files should have 'as' in the name.")
            index = index[0]
            value = self.database.get_param(param, wavelength)[index]
        elif param == 'fom':
            value = self.database.get_fom(wavelength)
        else:
            # Only 1 file, structure_type is invalid, get the file directly
            value = self.database.get_param(param, wavelength)
        # The number of points in each side
        side_num = int(np.sqrt(self.database.get_len()))
        value = value.reshape(side_num, side_num).astype(float)
        # If snakelike path, reverse the order of the second half
        if path_type == "snakelike":
            value[1::2] = value[1::2, ::-1]

        # The endpoints of the tetrahedron
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]
        ])
        # Move the center of the tetrahedron to the origin
        vertices -= np.mean(vertices, axis=0)
        # Create the figure if ax is not given
        fig = None
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # Add the tetrahedron
        faces = [[vertices[j] for j in [0, 1, 2]],
                 [vertices[j] for j in [0, 1, 3]],
                 [vertices[j] for j in [0, 2, 3]],
                 [vertices[j] for j in [1, 2, 3]]]
        ax.add_collection3d(
            Poly3DCollection(faces, alpha=0.05, linewidths=1,
                             edgecolors='darkslategray'))
        # Add labels to the vertices
        for i, txt in enumerate(self.element):
            ax.text(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                    txt, size=20, color='darkslategray')
        # Set the aspect of the plot
        ax.set_box_aspect([1, 1, 1])
        # Set the limits of the plot
        ax.set_xlim([-0.35, 0.35])
        ax.set_ylim([-0.35, 0.35])
        ax.set_zlim([-0.35, 0.35])
        # Remove cartesian axes
        ax.set_axis_off()
        # Add ticks on the edges
        for start, end in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            for i in range(1, num_ticks):
                tick = vertices[start] + (vertices[end] - vertices[start]) * i / num_ticks
                ax.plot([tick[0]], [tick[1]], [tick[2]], 'ko', markersize=2)
        # Set projection type
        ax.set_proj_type('persp')

        # Add the data point
        coord = self.get_coord(composition_type)
        # Convert the tetrahedral coordinates to Cartesian coordinates
        car_coord = np.array([self._tetrahedral_to_cartesian(vertices, c) for c in coord])
        # Set the color of the surface
        colors = plt.cm.summer(plt.Normalize(value.min(), value.max())(value))  # pylint: disable=all
        # Plot the surface
        ax.plot_surface(X=car_coord[:, 0].reshape(side_num, side_num),
                        Y=car_coord[:, 1].reshape(side_num, side_num),
                        Z=car_coord[:, 2].reshape(side_num, side_num),
                        facecolors=colors, shade=False, antialiased=True,
                        rstride=1, cstride=1)

        # Plot colorbar and set the value range from value.min() to value.max()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.summer,
                                   norm=plt.Normalize(value.min(), value.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)

        # Show the figure
        if if_show:
            plt.show()
        return ax, cbar

    def get_coord(self, composition_type: Literal['atomic', 'volumetric'] = 'atomic') -> np.ndarray:
        """Get the compositional coordinates of each point.

        Args:
            composition_type (Literal['atomic', 'volumetric'], optional): The type of composition. Defaults to 'atomic'.

        Raises:
            ValueError: Raise error if the type is invalid.

        Returns:
            np.ndarray: The compositional coordinates of each point.
        """
        # The number of points in each side
        side_num = int(np.sqrt(self.database.get_len()))
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
        point_thickness = np.stack((self.thickness[0] * abs_coord[:, 0],
                                    self.thickness[1] * abs_coord[:, 1],
                                    self.thickness[2] * (1 - abs_coord[:, 0]),
                                    self.thickness[3] * (1 - abs_coord[:, 1])), axis=1)
        if composition_type == "volumetric":
            # Normalize the volumetric composition
            vol_comp = point_thickness / np.sum(point_thickness, axis=1)[:, None]
            return vol_comp
        if composition_type == "atomic":
            density = np.array(
                [periodictable.elements.symbol(element).density for element in self.element])
            atomic_weight = np.array(
                [periodictable.elements.symbol(element).mass for element in self.element])
            atom_comp = point_thickness * density / atomic_weight
            # Normalize the atomic composition
            atom_comp /= np.sum(atom_comp, axis=1)[:, None]
            return atom_comp
        # Raise error if the type is invalid
        raise ValueError("Invalid type.")

    def _tetrahedral_to_cartesian(self,
                                  vertices: np.ndarray,
                                  coord: np.ndarray) -> np.ndarray:
        """Convert the tetrahedral coordinates to Cartesian coordinates.

        Args:
            vertices (np.ndarray): Endpoints of the tetrahedron.
            coord (np.ndarray): The tetrahedral coordinates.

        Returns:
            np.ndarray: The Cartesian coordinates.
        """
        cartesian = (coord[0] * vertices[0]
                     + coord[1] * vertices[1]
                     + coord[2] * vertices[2]
                     + coord[3] * vertices[3])
        return cartesian


if __name__ == '__main__':
    plotter = EllipPlotter(
        file_path=['pyhtp/ellip/test/test_data/ellip_data.xlsx'],
        element=['SiO2', 'Si'],
        thickness=[100, 100]
    )
    print(plotter.database.data)
    print(plotter.element)
    print(plotter.thickness)
    print(plotter.density)
    print(plotter.atomic_weight)
    print('All tests passed.')
