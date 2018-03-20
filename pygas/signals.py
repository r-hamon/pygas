# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2018
# -----------------
#
# * Ronan Hamon r<lastname_AT_protonmail.com>
#
# Description
# -----------
#
# pygas is a python package that regroups tools to transform graphs into a
# collection of signals.
#
# Version
# -------
#
# * pygas version = 0.1
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
"""Definition of a collection of static signals.

Author : Ronan Hamon
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from statsmodels import robust

from scipy.spatial.distance import squareform, pdist

from sklearn.mixture import GaussianMixture

EPSILON = np.finfo(np.float64).eps

SUBPLOTS_LAYOUT = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1),
                   6: (3, 2)}


def gm_thresholding(x):
    """Thresholding using binarization method.

    Parameters
    ----------
    distribution : array-like
        Distribution to consider.

    Returns
    -------
    float
        Threshold of the distribution `x`.
    """
    gm = GaussianMixture(2, n_init=10)

    outliers = np.zeros_like(x, dtype=np.bool)
    tau = np.max(x)

    while (np.sum(x < tau) > len(x) // 2):
        outliers = x > tau
        gm.fit(x[~outliers].reshape(-1, 1))
        tau = np.mean(gm.means_)

    return tau, outliers


class StaticSignals():
    """A collectcon of signals representing static graphs

    Parameters
    ----------
    signals: array-like
        Matrix of coordinates of points in an Euclidean space.

    Attributes
    ----------
    signals : ndarray
        Collection of signals, of shape (n_nodes, n_components).
    fft : ndarray
        FFT of 'signals' of shape (n_frequencies, n_components).
    magnitude : ndarray
        Magnitudes of 'signals'.
    energy : ndarray
        Energy of 'signals'.
    phase : ndarray
        Phase of 'signals'.
    n_nodes : int
        Number of points in a signal.
    n_components : int
        Number of components.
    n_frequencies : int
        Number of frequencies.

    Methods
    ----------
    fourierAnalysis:
        Compute the Fourier analysis on the time series.

    Examples
    --------
    >>> # collection of signals, from signals
    >>> signals = np.empty((50, 49))
    >>> for i in range(49):
    >>>     signals[:, i] = np.cos(2 * np.pi * np.linspace(0, 2, 50) * i)
    >>> signals += 0.1 * np.random.randn(50, 49)

    >>> coll_signals = StaticSignals()
    >>> coll_signals.from_signals(signals)
    >>> print(coll_signals)

    >>> coll_signals.draw_all_signals()
    >>> coll_signals.draw_signal(1)
    >>> coll_signals.draw_signal(2, same_figure=True)
    >>> coll_signals.draw_signal(3, same_figure=True)
    >>> coll_signals.draw_magnitude_signal(1)
    >>> coll_signals.draw_map_component_frequency(log=1)

    >>> plt.show()
    >>> plt.close('all')
    """

    def __init__(self, signals):

        self.signals = np.array(signals)

        # properties
        self.n_nodes = self.signals.shape[0]
        self.n_components = self.signals.shape[1]

        # compute fourier transform
        frequencies = np.arange(self.n_nodes)[range(int(self.n_nodes / 2 + 1))]
        self.n_frequencies = len(frequencies)

        self.fft = np.sqrt(2) * np.fft.rfft(self.signals, axis=0)
        self.fft[np.abs(self.fft) < EPSILON] = 0j

        # correction if `n_nodes` is even
        if self.n_nodes % 2 == 0:
            self.fft[-1, :] /= np.sqrt(2)

        self.magnitude = np.absolute(self.fft)
        self.magnitude[np.abs(self.magnitude) < EPSILON] = 0

        self.energy = np.absolute(self.fft)**2
        self.energy[np.abs(self.energy) < EPSILON] = 0

        self.phase = np.angle(self.fft)
        self.phase[np.abs(self.phase) < EPSILON] = 0

    @staticmethod
    def from_fft(fft):
        """Create a collection of static signals from FFT.

        Parameters
        ----------
        fft: numpy array
            Matrix of spectra.
        """
        # properties
        n_components = fft.shape[1]
        n_nodes = n_components + 1

        # correction if `n_nodes` is even
        if n_nodes % 2 == 0:
            fft[-1, :] *= np.sqrt(2)

        # Signals
        signals = np.fft.irfft(
            fft / np.sqrt(2), n=n_nodes, axis=0)

        return StaticSignals(signals)

    def plot_component(self, component, draw_marker=False, cmap=None):
        """Plot the selected component.

        Parameters
        ----------
        component : int or list
            Component to draw.
        """
        if cmap is None:
            from matplotlib.cm import gray_r
            cmap = gray_r

        x = np.arange(self.n_nodes)
        y = self.signals[:, (component - 1)]

        if draw_marker:
            colors = np.arange(self.n_nodes)

            min_color = np.amin(colors)
            max_color = np.amax(colors)
            scaled_colors = (colors - min_color) / (max_color - min_color)
            _ = plt.scatter(x, y, s=3, c=scaled_colors, cmap=cmap)
        else:
            _ = plt.plot(x, y)

        _ = plt.xlabel('Vertex')
        _ = plt.ylabel('Coordinate')
        plt.xticks([0, self.n_nodes // 2, self.n_nodes])
        plt.grid(True)
        plt.title('Component {:d}'.format(component))

    def plot_multiple_components(self, components):
        """Plot the selected components.

        Parameters
        ----------
        component : int or list
            Component to draw.
        """
        if not isinstance(components, list):
            components = list(components)

        for idc, component in enumerate(components, start=1):
            plt.subplot(*SUBPLOTS_LAYOUT[len(components)], idc)

            plt.plot(self.signals[:, (component - 1)],
                     label='Component %d' % component)
            plt.xlabel('Vertex')
            #  plt.yticks([])
            plt.grid()
            plt.title('Component {:d}'.format(component))
        plt.tight_layout()

    def draw_magnitude_signal(self, component):
        """Draw a component and its magnitude.

        Parameters
        ----------
        component: int
            Component to plot.
        """

        plt.figure()

        ax1 = plt.axes()
        plt.setp(ax1, yticks=[])
        plt.plot(self.magnitude[:, component], linewidth=3)
        plt.xlabel("Frequencies")

        ax2 = plt.axes([.55, .55, .3, .3], axisbg=[0.95, 0.95, 0.95])
        plt.setp(ax2, yticks=[])
        plt.plot(self.signals[:, component], linewidth=2)
        plt.xlabel("Vertices")

    def draw_map_component_frequency(self, log=False, max_component=None, max_frequency=None, cmap=None):
        """Energy in respect to component and frequency.

        Parameters
        ----------
        log: boolean
            Display in log10.
        max_component: int
            Number max of component to display.
        max_frequency: int
            Number max of frequency to display.
        """
        if max_frequency is None:
            max_frequency = self.n_frequencies
        if max_component is None:
            max_component = self.n_components

        if log:
            plt.imshow(np.log10(self.magnitude[1:max_frequency, 0:max_component]),
                       interpolation='none', origin='lower', aspect='auto')
        else:
            plt.imshow(self.magnitude[1:max_frequency, 0:max_component],
                       interpolation='none', origin='lower', aspect='auto', cmap=cmap)
#        plt.colorbar()

        plt.xlabel('Component')
        plt.ylabel('Frequency')

    def draw_points(self, axes=(0, 1), nodes=None):

        if nodes is None:
            nodes = np.arange(self.n_nodes)

        plt.scatter(self.signals[nodes, axes[0]], self.signals[nodes, axes[1]])
        for n in nodes:
            plt.text(self.signals[n, axes[0]],
                     self.signals[n, axes[1]], str(n))

        plt.xlabel('Dimension {}'.format(axes[0] + 1))
        plt.ylabel('Dimension {}'.format(axes[1] + 1))

    def get_distances(self, alpha=0):
        """Return the distances between points.

        Parameters
        ----------
        alpha : scalar
            Power coefficient to components.

        Returns
        -------
        array-like of size n_nodes * (n_nodes - 1)
            Distances between each pair of points, as a vector.
        """
        distances = np.zeros(self.n_nodes * (self.n_nodes-1)//2)
        energies = np.sum(self.signals**2, 0)**alpha
        energies /= np.sum(energies)
        energies *= (self.n_components)

        for component in range(self.n_components):

            # distances
            distances += energies[component] * \
                pdist(self.signals[:, component:(component+1)])**2

        return np.sqrt(distances)

    def to_graph(self, alpha=0, n_edges=None):
        """Transformation from a collection of signals to a static graph.

        Parameters
        ----------
        labeling : dict
            Dict associating to each vertex a label.
        alpha : int
            Value of alpha in the weighted computation of distances.

        Returns
        -------
        StaticGraph
        """
        from .graph import StaticGraph

        # compute distances
        distances = np.round(self.get_distances(alpha), 7)

        if n_edges is None:
            tau, _ = gm_thresholding(distances)
        else:
            tau = np.sort(distances)[n_edges]

        adj_mat = squareform(np.array(distances < tau, dtype=int))

        sg = StaticGraph(nx.from_numpy_matrix(adj_mat))

        return sg

    def __str__(self):
        string = ['==================',
                  'Collection of signals',
                  '==================',
                  'Number of nodes: {:d}'.format(self.n_nodes),
                  'Number of components: {:d},'.format(self.n_components),
                  'Number of frequencies: {:d}'.format(self.n_frequencies)]

        return '\n'.join(string)
