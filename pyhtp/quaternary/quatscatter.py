# -*- coding: utf-8 -*-
'''
Define a class for quaternary scattering.
'''
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d import Axes3D
from .quatplot import QuatPlot
from .utils import get_coord, get_chemical_formula
from ..xrd import XRDDatabase, CIFDatabase
from ..xrf import XRFDatabase


class QuatScatter(QuatPlot):
    """A class for quaternary scatter plots

    Attributes:
    """
    def __init__(
            self,
            value: list[str | int] | NDArray[np.str_ | np.int_],
            axis_label: tuple[str, str, str, str] | None = None,
            database: XRDDatabase | XRFDatabase | None = None,
            coords: NDArray[np.float64] | None = None,
            path_type: Literal['normal', 'snakelike'] = 'snakelike',
            composition_type: Literal['atomic', 'volumetric'] = 'atomic',
            ax: Axes3D | None = None):
        """Initializes the QuatScatter class

        Args:
            value (list[str | int] | NDArray[np.str_ | np.int_]): _description_
            axis_label (tuple[str, str, str, str] | None, optional):
                _description_. Defaults to None.
            database (XRDDatabase | XRFDatabase | None, optional):
                _description_. Defaults to None.
            coords (NDArray[np.float64] | None, optional): _description_. Defaults to None.
            path_type (Literal['normal', 'snakelike'], optional):
                _description_. Defaults to 'snakelike'.
            composition_type (Literal['atomic', 'volumetric'], optional):
                _description_. Defaults to 'atomic'.
            ax (Axes3D | None, optional): _description_. Defaults to None.
        """
        self.database = database
        if axis_label is None:
            if isinstance(database, XRFDatabase):
                assert len(database.elements) == 4
                axis_label = tuple(database.elements)  # type: ignore
            elif isinstance(database, XRDDatabase):
                assert len(database.info.element) == 4
                axis_label = tuple(database.info.element)  # type: ignore
            else:
                raise ValueError('Please provide the axis label if database is not given.')
        assert isinstance(axis_label, tuple)

        # The tetrahedron is plotted in the init of QuatPlot
        super().__init__(axis_label=axis_label, ax=ax)

        # Process the value
        self.value = np.array(value).flatten()
        self.side_number = int(np.sqrt(self.value.shape[0]))
        assert isinstance(self.value, np.ndarray)

        # Process the coords
        if coords is None:
            if isinstance(database, XRFDatabase):
                coords = get_coord(
                    element_order=self.axis_label,
                    side_number=self.side_number,
                    composition_type=composition_type,
                    xrf_database=database)
            elif isinstance(database, XRDDatabase):
                coords = get_coord(
                    element_order=self.axis_label,
                    side_number=self.side_number,
                    composition_type=composition_type,
                    info=database.info)
            else:
                raise ValueError('Please provide the coords if database is not given.')
        assert isinstance(coords, np.ndarray)
        assert coords.shape[0] == len(self.value) and coords.shape[1] == 4
        self.coords = coords

        # Process the path mode
        index_map = np.arange(len(self.value))
        if path_type == 'snakelike':
            index_map = index_map.reshape(self.side_number, self.side_number)
            index_map[1::2] = index_map[1::2, ::-1]
            index_map = index_map.flatten()
        self.index_map = index_map
        self.value = self.value[index_map].astype('<U100')

        self._json_path = ''
        self._picked_point_index = []

    def rotate90(self, times: int = 1) -> None:
        """Rotate the coordinates 90 degrees.

        Please note that the direction of rotation depends on
        the self.axis_label.

        Args:
            times (int, optional): Defaults to 1.
        """
        self.coords = np.rot90(
            self.coords.reshape(
                self.side_number, self.side_number, 4), times).reshape(-1, 4)

    def value_modify(
            self,
            json_path: str) -> None:
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
        import json  # pylint: disable=import-outside-toplevel
        if not json_path.endswith('.json'):
            raise ValueError("The file name should end with '.json'.")
        # Set the json path for future use
        if self._json_path == '':
            self._json_path = json_path
        # Set the dtype of label to str
        with open(json_path, 'r', encoding='utf-8') as f:
            modify = json.load(f)
        for target, element in modify.items():
            if not isinstance(element, list):
                raise ValueError("The value should be a list.")
            # Case 1
            if all(isinstance(i, int) for i in element):
                self.value[self.index_map[element]] = target
            # Case 2
            elif all(isinstance(i, str) for i in element):
                for i in element:
                    self.value[self.value == i] = target
            else:
                raise ValueError("The list elements should be all int or str.")

    def set_interactive(
            self,
            xrd_database: XRDDatabase | None = None,
            cif_database: CIFDatabase | None = None,
            ylim: tuple[float, float] | None = None,
            json_path: str = '',
            **kwargs) -> None:
        """Set the interactive mode.

        In the interactive mode, the user can double left click on a point
        to show the pattern of the corresponding phase.
        If cif_database is provided, the displayed pattern will
        accompanied by some cif phase patterns related to the
        clicked phase.

        Double right click to refresh the plot after twicking the json file.

        Args:
            xrd_database (XRDDatabase | None): The xrd database. If the
                self.database is an instance of XRDDatabase, this argument
                can be None. Defaults to None.
            cif_database (CIFDatabase | None, optional): Defaults to None.
            ylim (tuple[float, float] | None, optional): The y-axis limit of
                displayed pattern. Defaults to None. If None, the y-axis limit
                will be set to (0, maximum intensity).
            **kwargs: The keyword arguments for XRDPattern.plot.
        """
        if xrd_database is None:
            if isinstance(self.database, XRDDatabase):
                xrd_database = self.database
            else:
                raise ValueError("Please provide the xrd_database.")
        assert isinstance(xrd_database, XRDDatabase)

        if not json_path and not self._json_path:
            raise ValueError("Please provide the json path.")
        elif json_path:
            self._json_path = json_path
        assert self._json_path.endswith('.json')

        if ylim is None:
            ylim = (0, xrd_database.intensity.max())

        def _onpick(event):
            """Scatter plot pick event."""
            if not (event.mouseevent.dblclick and event.mouseevent.button == 1):
                return
            # Get the index of the selected point
            index = self.index_map[event.ind]
            if len(index) > 1:
                print("Multiple points selected. Zoom in to select a single point.")
                return
            assert len(index) == 1
            index = index[0]
            # Append the index to the picked point list
            if index not in self._picked_point_index:
                self._picked_point_index.append(index)
            self._resize_scatter()
            # Get the xrd pattern of the specific point
            pattern = xrd_database.data[index]
            # Get the chemical formula of the specific point
            formula = get_chemical_formula(
                list(self.axis_label), self.coords[event.ind[0]])
            # Plot the xrd pattern of the specific point
            if cif_database is None:
                fig, ax = plt.subplots()
                pattern.plot(
                    ax=ax,
                    max_intensity=xrd_database.intensity.max(),
                    **kwargs)
                fig.suptitle(f"{formula} - {pattern.info.index}")
                if ylim:
                    ax.set_ylim(*ylim)
            else:
                fig = plt.figure()
                pattern.plot_with_ref(
                    fig=fig,
                    cif_database=cif_database,
                    title=f"{formula} - {pattern.info.index}",
                    max_intensity=xrd_database.intensity.max(),
                    ylim=ylim, **kwargs)
            fig.canvas.mpl_connect('close_event', _onclose)
            plt.show()

        def _refresh(event):
            """Read json file and refresh the scatter with new labels."""
            # Detect if the r is pressed
            if not event.key == 'r':
                return
            # Read the json file
            self.value_modify(self._json_path)
            # Remove the scatter artist in self.artists
            self.artists.pop('scatter_main')
            self.legend_handles = []
            self.legend_labels = []
            # Plot the scatter points
            self.plot(marker=self.params.get('marker', 'o'),
                      markersize=self.params.get('markersize', 10))
            self._resize_scatter()
            # Refresh the plot
            self.refresh()
            # Add legend back
            if self.legend_handles:
                self.legend()

        def _onclose(event):
            """Close event.

            When figure is closed, remove the index from self._picked_point_index.
            """
            # Get the index from title
            fig_handle = event.canvas.manager.canvas.figure
            title = fig_handle._suptitle.get_text()  # pylint: disable=protected-access
            index = int(title.split('-')[1].strip())
            # Remove the index from the list
            while index in self._picked_point_index:
                self._picked_point_index.remove(index)
            # Reset the sizes of the scatter
            self._resize_scatter()

        def _store_view(event):
            """Store the view and lim of the plot when rotated and zoomed."""
            if event.inaxes is None:
                return
            self.params['elev'] = event.inaxes.elev
            self.params['azim'] = event.inaxes.azim
            self.params['roll'] = event.inaxes.roll
            self.params['xlim'] = event.inaxes.get_xlim()
            self.params['ylim'] = event.inaxes.get_ylim()
            self.params['zlim'] = event.inaxes.get_zlim()

        def _default_view(event):
            """Set to default view when press d."""
            if not event.key == 'd':
                return
            self.params['elev'] = None
            self.params['azim'] = None
            self.params['roll'] = None
            self.params['xlim'] = (-0.38, 0.38)
            self.params['ylim'] = (-0.38, 0.38)
            self.params['zlim'] = (-0.38, 0.38)
            self.refresh()
            # Add legend back
            if self.legend_handles:
                self.legend()

        assert isinstance(self.fig, Figure)
        self.fig.canvas.mpl_connect('pick_event', _onpick)
        self.fig.canvas.mpl_connect('key_press_event', _refresh)
        self.fig.canvas.mpl_connect('motion_notify_event', _store_view)
        self.fig.canvas.mpl_connect('key_press_event', _default_view)

        print(
            "Interactive manual:\n"
            "1. Double left click on a point to show the XRD pattern.\n"
            "2. Press 'r' to refresh the plot after twicking the json file.\n"
            "3. Press 'd' to set the plot to default view.\n")

    def _resize_scatter(self) -> None:
        """Resize the scatter points when self._picked_point_index is not empty."""
        if self._picked_point_index:
            # Get the index of the picked points
            index = self.index_map[np.array(self._picked_point_index)]
            # Get the size of the scatter points
            size = self.params.get('markersize', 10)
            sizes = np.full(len(self.value), size)
            sizes[index] = size * 2
            # Update the size of the scatter points
            self.artists['scatter_main'].set(sizes=sizes)
        else:
            # Reset the size of the scatter points
            sizes = np.full(len(self.value), self.params.get('markersize', 10))
            self.artists['scatter_main'].set(sizes=sizes)
        # Refresh the plot
        self.refresh()

    def line(
            self,
            coords: NDArray[np.float64] | list[tuple[float, float, float, float]] | list[int],
            color: str = 'tab:blue',
            linewidth: float = 2,
            linestyle: str = '-',
            label: str | None = None,
            alpha: float = 1,
            artist_name: str | None = None) -> Line2D:
        """Add a line to the quaternary phase diagram.

        Args:
            coords (NDArray[np.float64] | list[tuple[float, float, float, float]]):
                The quatenary coordinates of all the points. Shape (n, 4).
            color (str, optional): Defaults to 'tab:blue'.
            linewidth (float, optional): Defaults to 2.
            linestyle (str, optional): Defaults to '-'.
            label (str | None, optional): Defaults to None. The label of the line.
                If provided, the line will be added to the legend.
            alpha (float, optional): Defaults to 1. The transparency of the line.
            artist_name (str | None, optional): Defaults to None. The name of the artist.

        Returns:
            Line2D: The line artist
        """
        # Deal with the case that coords is a list of int
        if isinstance(coords, list) and isinstance(coords[0], int):
            coords = self.index_map[coords].tolist()  # type: ignore
            coords = self.coords[coords]  # type: ignore
        print(coords)
        assert isinstance(coords, np.ndarray) or isinstance(coords, list)
        return super().line(
            coords=coords,  # type: ignore
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            alpha=alpha,
            artist_name=artist_name)

    def plot(
            self,
            cmap: str = 'viridis',
            group_color: dict[str, str] | None = None,
            marker: str = 'o',
            markersize: int = 10) -> PathCollection:
        """Plot the scatter points.

        This function calls the scatter function of the parent class.

        Args:
            cmap (str, optional): Defaults to 'viridis'.
            group_color (dict[str, str] | None, optional): Defaults to None.
            marker (str, optional): Defaults to 'o'.
            markersize (int, optional): Defaults to 10.

        Returns:
            PathCollection: The artist object.
        """
        artist = self.scatter(
            coords=self.coords,
            value=self.value,
            cmap=cmap,
            group_color=group_color,
            marker=marker,
            markersize=markersize,
            max_legend_number=len(np.unique(self.value)) + 1,
            picker=True, artist_name='scatter_main')
        self.params['marker'] = marker
        self.params['markersize'] = markersize
        return artist

    def get_index_on_line(
            self,
            start: tuple[float, float, float, float] | NDArray[np.float64] | int,
            end: tuple[float, float, float, float] | NDArray[np.float64] | int,
            tol: float = 0.01) -> NDArray[np.int_]:
        """Get the index of points on a line.

        Args:
            start (tuple[float, float, float]): The start point of the line.
            end (tuple[float, float, float]): The end point of the line.
            tol (float, optional): Defaults to 0.01.

        Returns:
            NDArray[np.int_]: The index of points on the line.
        """
        if not isinstance(start, int) and not isinstance(end, int):
            start = self.tet_to_car(np.array(start))
            end = self.tet_to_car(np.array(end))
        elif isinstance(start, int) and isinstance(end, int):
            start = int(self.index_map[start])
            end = int(self.index_map[end])
            start = self.tet_to_car(self.coords[start])
            end = self.tet_to_car(self.coords[end])
        else:
            raise ValueError(
                "The start and end should be both int or both tuple or "
                "both np.ndarray.")
        line = end - start
        line_unit = line / np.linalg.norm(line)
        # Get the index of the points on the line
        points = []
        for i, car_coord in enumerate(self.tet_to_car(self.coords)):
            current_vec = car_coord - start
            t = np.dot(current_vec, line_unit)
            t = np.clip(t, 0, np.linalg.norm(line))
            closest = start + t * line_unit
            if np.linalg.norm(closest - car_coord) > tol:
                continue
            t_closest = np.dot(closest - start, line_unit)
            if t_closest < -tol or t_closest > np.linalg.norm(line) + tol:
                continue
            points.append(i)
        return self.index_map[points]
