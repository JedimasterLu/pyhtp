# -*- coding: utf-8 -*-
'''
Define a class for quaternary phase diagram.
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.ticker import StrMethodFormatter
from matplotlib.figure import Figure
from matplotlib.colors import to_hex, to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class QuatPlot:
    """A class for quaternary phase diagram.

    This class serves as a tool for QuatScatter and QuatSurface.

    Attributes:
        axis_label (tuple[str, str, str, str]): The labels of the axes.
        fig (Figure): The figure of the plot.
        ax (Axes3D): The axes of the plot.
        params (dict): The parameters of the plot.
            - fontfamily: The font family of the labels.
            - axiscolor: The color of the axis.
            - axislabelsize: The size of the axis text.
            - facecolor: The color of the face.
            - facealpha: The alpha of the face.
            - ticksize: The size of the ticks.
            - ticknumber: The number of ticks.
        artists (dict): The artists of the plot. The key is the name of the artist.
            And the value is the artist handle itself.
        legend_handles (list): The handles of the legend.
        legend_labels (list): The labels of the legend.
        _vertices (np.array): The vertices of the tetrahedron.
    """
    def __init__(
            self,
            axis_label: tuple[str, str, str, str],
            ax: Axes3D | None = None,
            **kwargs):
        """Initialize the quaternary phase diagram.

        Args:
            axis_label (tuple[str, str, str, str]): The labels of the axes.
            ax (Axes3D | None, optional): Defaults to None.
                If None, a new figure will be created.
                Otherwise, the plot will be added to the ax.
            **kwargs: Parameters for self.params.
                - fontfamily: The font family of the labels.
                - axiscolor: The color of the axis.
                - axislabelsize: The size of the axis text.
                - facecolor: The color of the face.
                - facealpha: The alpha of the face.
                - ticksize: The size of the ticks.
                - ticknumber: The number of ticks.

        Raises:
            ValueError: If the input ax is not an instance of Axes3D.
        """
        self.axis_label = axis_label

        self._vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3) / 2, 0],
            [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
        self._vertices -= np.mean(self._vertices, axis=0)

        if ax is None:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')  # type: ignore
        elif isinstance(ax, Axes3D):
            self.fig = ax.get_figure()
            self.ax = ax
        else:
            raise ValueError("The input ax must be an instance of Axes3D.")
        assert isinstance(self.ax, Axes3D)
        assert isinstance(self.fig, Figure)

        self.params = {
            'fontfamily': kwargs.pop('fontfamily', 'DejaVu Sans'),
            'axiscolor': kwargs.pop('axiscolor', 'darkslategray'),
            'axislinewidth': kwargs.pop('axislinewidth', 1),
            'axislabelsize': kwargs.pop('axislabelsize', 18),
            'axislabelpad': kwargs.pop('axislabelpad', 0.08),
            'facecolor': kwargs.pop('facecolor', 'tab:blue'),
            'facealpha': kwargs.pop('facealpha', 0.05),
            'ticksize': kwargs.pop('ticksize', 2),
            'ticknumber': kwargs.pop('ticknumber', 5),
            'xlim': kwargs.pop('xlim', (-0.38, 0.38)),
            'ylim': kwargs.pop('ylim', (-0.38, 0.38)),
            'zlim': kwargs.pop('zlim', (-0.38, 0.38)),
        }

        self._build_tetrahedron()

        self.artists = {}

        self.legend_handles = []
        self.legend_labels = []

    def _build_tetrahedron(self):
        """Build a tetrahedron."""
        # Add the tetrahedron to the plot
        faces = [[self._vertices[j] for j in [0, 1, 2]],
                 [self._vertices[j] for j in [0, 1, 3]],
                 [self._vertices[j] for j in [0, 2, 3]],
                 [self._vertices[j] for j in [1, 2, 3]]]
        self.ax.add_collection3d(
            Poly3DCollection(
                faces, alpha=self.params['facealpha'],
                linewidths=self.params['axislinewidth'],
                facecolors=self.params['facecolor'],
                edgecolors=self.params['axiscolor']))
        # Add labels to the vertices
        pad = self.params['axislabelpad']
        for i, txt in enumerate(self.axis_label):
            if i == 0:
                coord = self._vertices[i] + np.array(
                    [-np.sqrt(3) * pad / 2, - pad / 2, - pad / 2])
            elif i == 1:
                coord = self._vertices[i] + np.array(
                    [np.sqrt(3) * pad / 2, - pad / 2, - pad / 2])
            elif i == 2:
                coord = self._vertices[i] + np.array(
                    [0, pad, - pad / 2])
            else:
                coord = self._vertices[i] + np.array(
                    [0, 0, pad / 2])
            self.ax.text(
                coord[0], coord[1], coord[2], txt,
                size=self.params['axislabelsize'],
                color=self.params['axiscolor'],
                fontfamily=self.params['fontfamily'])
        # Set aspect and lims and axes
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(*self.params['xlim'])
        self.ax.set_ylim(*self.params['ylim'])
        self.ax.set_zlim(*self.params['zlim'])
        self.ax.set_axis_off()
        self.ax.set_proj_type('persp')
        self.ax.view_init(
            elev=self.params.get('elev', None),
            azim=self.params.get('azim', None),
            roll=self.params.get('roll', None),
            vertical_axis='z')
        # Add ticks on the edges
        for start, end in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            for i in range(1, self.params['ticknumber']):
                tick = self._vertices[start] + \
                    (self._vertices[end] - self._vertices[start]) * i \
                    / self.params['ticknumber']
                self.ax.plot(
                    [tick[0]], [tick[1]], [tick[2]], 'o',
                    color=self.params['axiscolor'],
                    markersize=self.params['ticksize'])

    def set_params(self, **kwargs) -> None:
        """Set the parameters for the quaternary phase diagram.

        Args:
            **kwargs: The parameters to set.
                - fontfamily: The font family of the labels.
                - axiscolor: The color of the axis.
                - axislabelsize: The size of the axis text.
                - axislinewidth: The width of the axis.
                - facecolor: The color of the face.
                - facealpha: The alpha of the face.
                - ticksize: The size of the ticks.
                - ticknumber: The number of ticks.
        """
        self.params.update(kwargs)
        self.refresh()

    @staticmethod
    def _get_colors(
            color_number: int,
            cmap_for_long: str = 'viridis') -> list[str]:
        """Pick a colormap based on the number of phases.

        - <= 10: tab10
        - 10 < x <= 20: tab20
        - > 20: viridis (change to discrete colors)

        Args:
            number (int): The number of phases.
            cmap_for_long (str, optional):
                The colormap for long list. Defaults to 'viridis'.

        Returns:
            list[str]: The list of hex colors.
        """
        if color_number <= 10:
            cmap = plt.cm.get_cmap('tab10')
            result = [to_hex(cmap(i), keep_alpha=True)
                      for i in range(color_number)]
        elif 10 < color_number <= 20:
            cmap = plt.cm.get_cmap('tab20')
            result = [to_hex(cmap(i), keep_alpha=True)
                      for i in range(color_number)]
        else:
            cmap = plt.cm.get_cmap(cmap_for_long)
            result = [to_hex(cmap(i / color_number), keep_alpha=True)
                      for i in range(color_number)]
        return result

    def tet_to_car(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert tetrahedral coordinates to Cartesian coordinates.

        Args:
            coords (NDArray[np.float64]): The tetrahedral coordinates. Shape (n, 4).

        Returns:
            NDArray[np.float64]: The Cartesian coordinates. Shape (n, 3).
        """
        return np.dot(coords, self._vertices)

    def car_to_tet(self, coords: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert Cartesian coordinates to tetrahedral coordinates.

        Args:
            coords (NDArray[np.float64]): The Cartesian coordinates. Shape (n, 3).

        Returns:
            NDArray[np.float64]: The tetrahedral coordinates. Shape (n, 4).
        """
        return np.dot(coords, np.linalg.inv(self._vertices))

    def scatter(
            self,
            coords: NDArray[np.float64] | list[tuple[float, float, float, float]],
            value: list | NDArray | None = None,
            cmap: str = 'viridis',
            color: list[str] | NDArray[np.str_] | str = 'tab:blue',
            group_color: dict[str, str] | None = None,
            artist_name: str | None = None,
            marker: str = 'o',
            markersize: int = 10,
            max_legend_number: int = 20,
            picker: bool = False) -> PathCollection:
        """Add scatter plot to the quaternary phase diagram.

        If value is provided, the color will be determined by the value.
        If the group number of value is less than max_legend_number, a legend will be added.

        Args:
            coords (NDArray[np.float64]): The coordinates. Shape (n, 4).
            value (list | NDArray | None, optional): The value of the points. Defaults to None.
            cmap (str, optional): The colormap. Defaults to 'viridis'.
            color (list[str] | NDArray[np.str_] | str, optional):
                The color of the points. Defaults to 'tab:blue'. If str, all points
                will have the same color.
            group_color (dict[str, str] | None, optional): The color of the groups.
                Defaults to None.
            artist_name (str | None, optional): The name of the artist. Defaults to None.
            marker (str, optional): The marker of the points. Defaults to 'o'.
            markersize (int, optional): The size of the markers. Defaults to 10.
            max_legend_number (int, optional): The maximum number of legends. Defaults to 20.
            picker (bool, optional): Whether to enable the picker. Defaults to False.

        Returns:
            PathCollection: The scatter plot artist.
        """
        coords = np.array(coords)
        # Normalize the coordinates so that the sum of each row is 1
        coords = coords / np.sum(coords, axis=1)[:, None]
        assert isinstance(coords, np.ndarray) and coords.shape[1] == 4

        artist = None
        if value is None:
            artist = self.ax.scatter(
                *self.tet_to_car(coords).T, marker=marker, c=color, s=markersize)
        elif value is not None:
            if group_color is None:
                color_of_groups = self._get_colors(len(np.unique(value)), cmap)
                group_color = dict(zip(np.unique(value), color_of_groups))
            color = [group_color[i] for i in value]
            artist = self.ax.scatter(
                *self.tet_to_car(coords).T,
                c=color, s=markersize, marker=marker, picker=picker)
            if len(group_color) <= max_legend_number:
                self.legend_handles.extend(
                    [Line2D([0], [0], marker=marker, color='w', label=group_name,
                            markerfacecolor=group_color[group_name], markersize=10)
                     for group_name in group_color])
                self.legend_labels.extend(list(group_color.keys()))
        # Put the handle of the artist into the artists dictionary
        if artist_name is None:
            # Get the number of scatter artists
            current_scatter_num = len([i for i in self.artists if 'scatter' in i])
            artist_name = f'scatter_{current_scatter_num}'
        assert isinstance(artist, PathCollection)
        self.artists[artist_name] = artist
        return artist

    def line(
            self,
            coords: NDArray[np.float64] | list[tuple[float, float, float, float]],
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
        coords = np.array(coords)
        # Normalize the coordinates so that the sum of each row is 1
        coords = coords / np.sum(coords, axis=1)[:, None]
        assert isinstance(coords, np.ndarray) and coords.shape[1] == 4

        # Check if the length of the coordinates is correct
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError("The shape of coords should be (n, 4).")
        artist = self.ax.plot(
            *self.tet_to_car(coords).T, color=color,
            linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        if label is not None:
            self.legend_handles.append(artist[0])
            self.legend_labels.append(label)

        # Put the handle of the artist into the artists dictionary
        if artist_name is None:
            # Get the number of line artists
            current_line_num = len([i for i in self.artists if 'line' in i])
            artist_name = f'line_{current_line_num}'
        self.artists[artist_name] = artist[0]
        return artist[0]

    def surface(
            self,
            coords: NDArray[np.float64] | list[tuple[float, float, float, float]],
            value: NDArray[np.float64] | None = None,
            cmap: str = 'viridis',
            color: list[str] | NDArray[np.str_] | None = None,
            vlim: tuple[float, float] | float | None = None,
            artist_name: str | None = None) -> Poly3DCollection:
        """Add a surface to the quaternary phase diagram.

        Args:
            coords (NDArray[np.float64]): The coordinates of the points. Shape (n, 4).
            value (NDArray[np.float64]): The values of the points. Shape (n,).
            cmap (str, optional): The colormap. Defaults to 'viridis'.
            color (list[str] | NDArray[np.str_] | None, optional):
                The color of the points. Defaults to None. If provided,
                it will overwrite the cmap and value.
            vlim (tuple[float, float] | float | None, optional):
                The value limits. Defaults to None. If tuple, it will be the
                actual value. If float, it will be the percentile.
                For example, vlim=5 means (5%, 95%). Defaults to None.
            artist_name (str | None, optional): The name of the artist. Defaults to None.

        Returns:
            Poly3DCollection: The surface artist.
        """
        coords = np.array(coords)
        # Normalize the coordinates so that the sum of each row is 1
        coords = coords / np.sum(coords, axis=1)[:, None]
        assert isinstance(coords, np.ndarray) and coords.shape[1] == 4

        if color is not None:
            color = np.array(color).flatten()
            assert color.shape[0] == coords.shape[0]
            # The elements in color should be hex colors
            # Convert to RGBA, with shape (n, 4)
            color = to_rgba_array(color)
        elif value is not None:
            if vlim is None:
                vlim = (np.min(value).astype(float),
                        np.max(value).astype(float))
            elif isinstance(vlim, float | int):
                vlim = (np.percentile(value, vlim).astype(float),
                        np.percentile(value, 100 - vlim).astype(float))
            assert isinstance(vlim, tuple)
            assert len(vlim) == 2
            color = plt.get_cmap(cmap)(Normalize(*vlim)(value))
        assert isinstance(color, np.ndarray)

        car_coords = self.tet_to_car(coords)
        side_num = int(np.sqrt(coords.shape[0]))
        artist = self.ax.plot_surface(
            X=car_coords[:, 0].reshape(side_num, side_num),
            Y=car_coords[:, 1].reshape(side_num, side_num),
            Z=car_coords[:, 2].reshape(side_num, side_num),
            cmap=cmap, clim=vlim,
            facecolors=color.reshape(side_num, side_num, 4),
            shade=False, antialiased=True,
            rstride=1, cstride=1)

        # Put the handle of the artist into the artists dictionary
        if artist_name is None:
            # Get the number of surface artists
            current_surface_num = len([i for i in self.artists if 'surface' in i])
            artist_name = f'surface_{current_surface_num}'
        self.artists[artist_name] = artist
        return artist

    def colorbar(
            self,
            artist: Poly3DCollection | None = None,
            artist_name: str | None = None,
            tick_number: int = 5,
            precision: int = 2,
            **kwargs) -> Colorbar:
        """Add colorbar to the quaternary phase diagram.

        Args:
            artist (Poly3DCollection | None, optional):
                If None, the artist will be set to the surface artist
                in self.artists. Defaults to None. If there are multiple
                surface artists, the first one will be used, with a warning
                message.
            artist_name (str | None, optional): The name of the artist.
                Defaults to None. If provided, the artist will be set to
                self.artists[artist_name].
            **kwargs: The arguments for the colorbar.
                - shrink: The shrink of the colorbar. Defaults to 0.5.
                - pad: The pad of the colorbar. Defaults to -0.05.
                - labelsize: The size of the label.
                - colors: The color of the label.
                - labelfontfamily: The font family of the label.
                Other arguments are passed to the tick_params function.

        Returns:
            Colorbar: The colorbar artist.
        """
        if artist is None:
            surface_artists = [i for i in self.artists if 'surface' in i]
            if len(surface_artists) == 0:
                raise ValueError("No surface artist found.")
            elif len(surface_artists) > 1:
                print(f"Multiple surface artists found. Using {surface_artists[0]}.")
            artist = self.artists[surface_artists[0]]
        assert isinstance(artist, Poly3DCollection)
        assert isinstance(self.fig, Figure)

        cmap, clim = artist.get_cmap(), artist.get_clim()
        virtual_artist = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(*clim))
        virtual_artist.set_array([])

        cbar = self.fig.colorbar(
            virtual_artist, ax=self.ax,
            shrink=kwargs.pop('shrink', 0.5),
            pad=kwargs.pop('pad', -0.05))
        cbar.set_ticks(np.linspace(clim[0], clim[1], tick_number))  # type: ignore

        # Format the tick labels to show specified decimal places
        if precision is not None:
            formatter = StrMethodFormatter(f"{{x:.{precision}f}}")
            cbar.ax.yaxis.set_major_formatter(formatter)

        # Set the colorbar properties
        cbar.ax.tick_params(
            labelsize=kwargs.pop('labelsize', self.params['axislabelsize'] - 2),
            colors=kwargs.pop('colors', self.params['axiscolor']),
            labelfontfamily=kwargs.pop('labelfontfamily', self.params['fontfamily']),
            **kwargs)

        # Put the handle of the artist into the artists dictionary
        if artist_name is None:
            # Get the number of colorbar artists
            current_colorbar_num = len([i for i in self.artists if 'colorbar' in i])
            artist_name = f'colorbar_{current_colorbar_num}'

        self.artists[artist_name] = cbar
        return cbar

    def show(self) -> None:
        """Show the quaternary phase diagram."""
        assert isinstance(self.fig, Figure)
        self.fig.tight_layout()
        plt.show()

    def refresh(self) -> None:
        """Refresh the quaternary phase diagram."""
        assert isinstance(self.fig, Figure)
        # Clear the axes and redraw all the artists in the dictionary
        self.ax.clear()
        self._build_tetrahedron()
        for artist in self.artists.values():
            self.ax.add_artist(artist)
        self.fig.tight_layout()
        # Set elev, azim, and lim
        self.ax.view_init(
            elev=self.params.get('elev', None),
            azim=self.params.get('azim', None),
            roll=self.params.get('roll', None),
            vertical_axis='z')
        self.ax.set_xlim(*self.params['xlim'])
        self.ax.set_ylim(*self.params['ylim'])
        self.ax.set_zlim(*self.params['zlim'])
        plt.draw()

    def legend(self, **kwargs) -> None:
        """Add legend to the quaternary phase diagram.

        Args:
            **kwargs: The arguments for the legend.
                - loc: The location of the legend. Defaults to 'center left'.
                - bbox_to_anchor: The bbox_to_anchor of the legend. Defaults to (0.7, 0.8).
                - fontfamily: The font family of the legend. Defaults to 'DejaVu Sans'.
                - fontsize: The font size of the legend. Defaults to 12.
                Other arguments are passed to the legend function.
        """
        self.ax.legend(
            self.legend_handles, self.legend_labels,
            loc=kwargs.pop('loc', 'center left'),
            bbox_to_anchor=kwargs.pop('bbox_to_anchor', (0.7, 0.8)),
            prop={'family': kwargs.pop('fontfamily', self.params['fontfamily']),
                  'size': kwargs.pop('fontsize', 12)},
            **kwargs)

    def save_fig(
            self,
            filename: str,
            dpi: int = 300) -> None:
        """Save the quaternary phase diagram to a file.

        Args:
            filename (str): The filename to save.
            dpi (int, optional): Defaults to 300.
        """
        assert isinstance(self.fig, Figure)
        self.fig.savefig(filename, dpi=dpi)

    def save_animation(
            self,
            filename: str,
            rotation_period: float = 10.0,
            fps: int = 30,
            dpi: int = 300) -> None:
        """Save an animation of the quaternary phase diagram.

        Args:
            filename (str): The filename to save the animation.
            rotation_period_seconds (float, optional):
                Time in seconds for one complete 360° rotation.
                Defaults to 10.0 seconds.
            fps (int, optional): Frames per second. Defaults to 30.
            dpi (int, optional): Dots per inch. Defaults to 300.
        """
        from matplotlib.animation import FuncAnimation  # pylint: disable=import-outside-toplevel

        frames = int(rotation_period * fps)

        def animate(i):
            angle = 360 * i / frames
            self.ax.view_init(elev=20, azim=angle)
            return [self.ax]

        assert isinstance(self.fig, Figure)
        ani = FuncAnimation(
            self.fig, animate, frames=frames,
            blit=True, interval=1000 / fps)

        if filename.endswith('.gif'):
            ani.save(filename, writer='pillow', fps=fps, dpi=dpi)
        else:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=dpi)
