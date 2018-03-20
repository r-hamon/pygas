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
"""Definition of a static graph.

This implementation lies on the Networkx module, and more specifically on the
class Graph.

Author: Ronan Hamon
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.spatial.distance import squareform


EPSILON = np.finfo(np.float32).eps


def _double_centering(A):
    n = A.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5*np.dot(J, np.dot(A, J))


class StaticGraph(nx.Graph):
    """Representation of a undirected static graph.

    Parameters
    ----------
    data : graph object or None, optional
        Data to initialize graph. If None (default), an empty graph is created.
        The data can be an edge list, or any NetworkX graph object. The data
        can also be a NumPy matrix or 2d ndarray, a SciPy sparse matrix, or a
        PyGraphviz graph.
    """

    def __init__(self, data=None):

        super().__init__(data)

        self.background = None

        # add `id` and `label` attributes
        self.set_node_attributes({node: id_node for id_node, node in
                                  enumerate(self.nodes())}, 'id')
        self.set_node_attributes({node: id_node for id_node, node in
                                  enumerate(self.nodes())}, 'label')

        # add a weight to all edges
        weights = self.get_edge_attributes('weight')
        missing_weights = set(self.edges()).difference(weights.keys())
        if missing_weights:
            weights.update({edge: 1 for edge in missing_weights})
            self.set_edge_attributes(weights, 'weight')

    @property
    def n_nodes(self):
        """Number of nodes."""
        return self.number_of_nodes()

    @property
    def n_edges(self):
        """Number of edges."""
        return self.number_of_edges()

    @property
    def adjacency_matrix(self):
        """Adjacency matrix of the graph."""
        return nx.to_numpy_matrix(self, sorted(self.nodes())).astype(np.bool)

    def add_node(self, u, **attr):
        """Add a node to the graph."""
        super().add_node(u, id=self.n_nodes, **attr)

    def add_edge(self, u, v, **attr):
        """Add a node to the graph."""
        if 'weight' not in attr:
            attr['weight'] = 1
        super().add_edge(u, v, **attr)

    def set_node_attributes(self, mapping, name_attr):
        """Set an attribute to all nodes.

        Parameters
        ----------
        name_attr : str
            Name of the attribute.
        mapping : dict
            Dictionary of attribute values keyed by node. If values is not a
            dictionary, then it is treated as a single attribute value that is
            then applied to every node of the graph.
        """
        nx.set_node_attributes(self, name=name_attr,  values=mapping)

    def get_node_attributes(self, name_attr):
        """Get the value of an attribute for all nodes.

        Parameters
        ----------
        name_attr : str
            Name of the attribute to get.

        Returns
        -------
        dict
            Dictionnary with the nodes as keys and the values of the attribute
            as values.
        """
        return nx.get_node_attributes(self, name=name_attr)

    def set_edge_attributes(self, mapping, name_attr):
        """Set an attribute to edges.

        Parameters
        ----------
        name_attr: str
            Name of the attribute.
        mapping: dict
            Dictionary of attribute values keyed by edge (tuple). The keys must
            be tuples of the form (u, v). If values is not a dictionary, then
            it is treated as a single attribute value that is then applied to
            every edge in G.
        """
        nx.set_edge_attributes(self, name=name_attr, values=mapping)

    def get_edge_attributes(self, name_attr):
        """Get the value of an attribute for all edges.

        Parameters
        ----------
        name_attr : str
            Name of the attribute.

        Return
        ------
        dict
            Dictionnary with the edges as keys and the values of the attribute
            as values.
         """
        return nx.get_edge_attributes(self, name_attr)

    def get_layout(self):
        """Get the layout."""
        return self.get_node_attributes('xy')

    def set_layout(self, layout, name_attr=None):
        """Set a layout for the vertices.

        Parameters
        ----------
        layout : dict
            Mapping from 'key' to 2D coordinates.
        name_attr : str, optional
            Attribute used to key vertices.

        Raises
        ------
        KeyError
            If the key is not an attribute of the nodes.
        """
        if name_attr:
            # define a mapping from key in layout to key `id`
            mapping = {value: key for key,
                       value in self.get_node_attributes(name_attr).items()}
        else:
            mapping = dict(zip(self.nodes(), self.nodes()))

        # get the mapped keys of the layout
        try:
            keys = [mapping[key] for key in layout.keys()]
        except KeyError as e:
            error = 'Key {} is not a valid identifier of node for attribute {}'
            raise KeyError(error.format(e, name_attr))

        # get the coordinate associated to each key
        if self.background:
            coords = [self.background.get_coordinates(value[0], value[1])
                      for value in layout.values()]
        else:
            coords = list(layout.values())

        mapping = dict(zip(keys, coords))

        self.set_node_attributes(mapping, 'xy')

    def relabel_nodes(self, mapping):
        """Relabel vertices of the graph

        Parameters
        ----------
        mapping : dict
            A dictionary whose keys are old labels and values new labels.
        """
        g = nx.relabel_nodes(self, mapping)
        self._adj = g._adj
        self._node = g._node
        self.set_node_attributes(
            dict(zip(self.nodes(), self.nodes())), 'label')

    def get_distance_matrix(self, method):

        if method == 'shimada':
            d_matrix = np.zeros(self.n_nodes * (self.n_nodes - 1) // 2)
            weight = np.sqrt(self.n_nodes / (self.n_nodes - 1))
            curseur = 0

            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    d_matrix[curseur] = 1 * int(self.has_edge(i, j))
                    d_matrix[curseur] += weight * \
                        int(not self.has_edge(i, j))
                    curseur += 1

            return squareform(d_matrix)
        elif method == 'isomap':

            d_matrix = list()
            dijkstra_distances = nx.algorithms.all_pairs_dijkstra_path_length(
                self)
            for node_u in range(self.n_nodes):
                for node_v in range(node_u + 1, self.n_nodes):
                    if node_v in dijkstra_distances[node_u]:
                        d_matrix.append(dijkstra_distances[node_u][node_v])
                    else:
                        d_matrix.append(np.inf)
        elif method == 'eigenmap':
            return nx.laplacian_matrix(self).todense()

    def to_signals(self):
        """Transformation from a static graph to a collection of signals."""
        from .signals import StaticSignals
        n_nodes = self.n_nodes

        if n_nodes > 0:

            # computation of the distance matrix
            D = self.get_distance_matrix(method='shimada')

            # centering matrix
            B = _double_centering(D**2)

            # diagonalization of B
            eival, eivec = np.linalg.eigh(B)

            # sorting by descending order the eigenvalues
            order = np.argsort(eival)[::-1]
            eival = eival[order]
            eivec = eivec[:, order]

            # round off small values
            eival[np.abs(eival) < EPSILON] = 0
            eivec[np.abs(eivec) < EPSILON] = 0

            # deletion of negative values
            is_positive = eival > 0
            eival = eival[is_positive]
            eivec = eivec[:, is_positive]

            # CMDS coordinates
            x_m = np.dot(eivec, np.diag(np.sqrt(eival)))

        else:
            x_m = np.zeros((0, 0))

        return StaticSignals(x_m)

    def draw(self, label='id', size=50, color='#A0CBE2', cmap=None,
             legend=False, draw_nodes=True, draw_node_labels=True,
             draw_edges=True, voronoi=False):
        """Draw the graph.

        Parameters
        -----------
        label : str, optional
            Attribute used to label vertices. If None, no label is displayed.
        size : str or int, optional
            Attribute used to size vertices. If str, the size is scaled from
            MIN_SIZE to MAX_SIZE according to the attribute. If int, the size
            if set to the value of the parameter for all vertices.
        color : str or tuple, optional
            Attribute used to color vertices. If str, if 'color' is an node
            attribute, then the color is scaled using cmap according to the
            attribute, else the paramater is the color for all vertices.
        legend : bool, optional
            Indicates if the legend is displayed.

        Notes
        -----
        - If the graph is weighted, the width of the link is proportional to\
        its weight.

        - The constants give minimal and maximal values for the width of edge\
        and the size of nodes.
        """
        # The constants give minimal and maximal values for the width of edge
        # and the size of nodes.  width of edges
        MAX_WIDTH = 2
        MIN_WIDTH = 0.002
        ALPHA_WIDTH = 3

        # size of nodes
        MAX_SIZE = 800
        MIN_SIZE = 30
        ALPHA_SIZE = 2

        if self.background is not None:
            self.background.plot_map()

        if cmap is None:
            from matplotlib.cm import gray_r
            cmap = gray_r

        # set the layout if not
        layout = self.get_node_attributes('xy')
        if not layout:
            layout = nx.spring_layout(self)

        # set node color
        colors = self.get_node_attributes(color)

        if colors:

            # get colors
            colors = list(colors.values())

            # scales colors from 0 to 1
            min_color = np.amin(colors)
            max_color = np.amax(colors)

            if min_color == max_color:
                node_color = '#A0CBE2'
            else:
                # set flag
                scaled_colors = (colors - min_color) / (max_color - min_color)
                # get colors
                node_color = cmap(scaled_colors)
        else:
            node_color = color

        # set node size
        sizes = self.get_node_attributes(size)
        if sizes:

            # get sizes
            sizes = list(sizes.values())

            # scale sizes from MIN_SIZE to MAX_SIZE
            min_size = np.amin(sizes)
            max_size = np.amax(sizes)

            if min_size == max_size:
                node_size = MIN_SIZE
            else:
                node_size = (sizes - min_size) / (max_size - min_size)
                node_size **= ALPHA_SIZE
                node_size *= (MAX_SIZE - MIN_SIZE)
                node_size += MIN_SIZE
        else:
            node_size = size

        # set edge width
        weights = list(self.get_edge_attributes('weight').values())

        # scale sizes from MIN_SIZE to MAX_SIZE
        min_weight = np.amin(weights)
        max_weight = np.amax(weights)

        if min_weight == max_weight:
            width = min_weight
        else:
            width = (weights - min_weight) / (max_weight - min_weight)
            width **= ALPHA_WIDTH
            width *= (MAX_WIDTH - MIN_WIDTH)
            width += MIN_WIDTH

        # draw nodes
        if draw_nodes:
            nodes = nx.draw_networkx_nodes(self,
                                           pos=layout,
                                           node_size=node_size,
                                           node_color=node_color,
                                           cmap=plt.cm.hsv)

        # draw nodes labels
        if draw_node_labels:
            node_labels = self.get_node_attributes(label)
            nx.draw_networkx_labels(self,
                                    pos=layout,
                                    labels=node_labels, font_size=10)

        # draw edges
        if draw_edges:
            edges = nx.draw_networkx_edges(self,
                                           pos=layout,
                                           width=width,
                                           alpha=0.6)

        if voronoi:

            from matplotlib.path import Path
            import matplotlib.patches as patches
            from scipy.spatial import Voronoi
            points = np.array(list(layout.values()))

            vor = Voronoi(points)
            if [] in vor.regions:
                vor.regions.remove([])

            assert len(vor.regions) == len(points)
            ax = self.map.fig.get_axes()[0]

            for idd, region in enumerate(vor.regions):

                if -1 in region:
                    region.remove(-1)
                path = np.zeros((len(region), 2))

                path = vor.vertices[region + [region[0]]]

                path = Path(path)
                patch = patches.PathPatch(path,
                                          facecolor=node_color[vor.point_region[idd] - 1],
                                          alpha=0.4, lw=1)
                ax.add_patch(patch)

        # set the original limits
        if self.background is not None:
            plt.xlim(self.map.xlim)
            plt.ylim(self.map.ylim)

        # remove axis
        plt.xticks([])
        plt.yticks([])

        # plot legend
        #  if legend:
#
        #  lgd_artists = []
        #  lgd_labels = []
        #  lgd_title = []
#
        #  # if color, set colorbar
        #  if color in self.list_node_attributes:
        #  if min_color != max_color:
        #  norm = plt.Normalize(vmin=min_color, vmax=max_color)
        #  scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        #  scalar_map.set_array([])
        #  plt.colorbar(scalar_map, fraction=0.2,
        #  pad=0.01, aspect=50, shrink=0.55)
#
        #  lgd_title += ["Color: {:s}".format(
        #  color.replace('_', ' '))]
#
        #  # if size, add dots
        #  if size in self.list_node_attributes and False:
        #  sizes = list(self.get_node_attributes(size).values())
        #  if min_size != max_size:
        #  # get max, min and median values
        #  imax = np.argmax(sizes)
        #  imedian = np.abs(sizes - np.median(sizes)).argmin()
        #  imin = np.argmin(sizes)
#
        #  # create dot with the different sizes
        #  max_dot = copy.copy(nodes)
        #  max_dot.set_sizes([node_size[imax]])
        #  median_dot = copy.copy(nodes)
        #  median_dot.set_sizes([node_size[imedian]])
        #  min_dot = copy.copy(nodes)
        #  min_dot.set_sizes([node_size[imin]])
#
        #  # add dots and corresponding values in the legend
        #  lgd_artists += [max_dot, median_dot, min_dot]
        #  lgd_labels += ["{:.2e}".format(sizes[imax]),
        #  "{:.2e}".format(sizes[imedian]),
        #  "{:.2e}".format(sizes[imin])]
#
        #  lgd_title += [" Size: {:s}".format(size.replace('_', ' '))]
#
        #  # A TESTER
        #  if "weight" in self.list_edge_attributes and False:
#
        #  weights = self.get_edge_attribute('weight')
        #  # get max, min and median values
        #  imax = np.argmax(weights)
        #  imedian = np.abs(weights - np.median(weights)).argmin()
        #  imin = np.argmin(weights)
#
        #  # create lines with the different sizes
        #  max_line = copy.copy(nodes)
        #  max_line.set_sizes([node_size[imax]])
        #  median_dot = copy.copy(nodes)
        #  median_dot.set_sizes([node_size[imedian]])
        #  min_dot = copy.copy(nodes)
        #  min_dot.set_sizes([node_size[imin]])
#
        #  # add dots and corresponding values in the legend
        #  lgd_artists += [max_line]
        #  lgd_labels += [edges[imax]]
        #  lgd_title += ["Width: weights"]
#
        #  # if size in self.list_node_attributes and \
        #  #    "weight" in self.list_edge_attributes:
        #  #     plt.legend(lgd_artists, lgd_labels, title='\n '.join(lgd_title), loc=0,
        #  #                scatterpoints=1, fancybox=True, fontsize="x-large",
        #  #                borderpad=1, labelspacing=2)

    def similarity(self, static_graph, method="accuracy_edges"):
        """Return a similarity index between two StaticGraph.

        Parameters
        ----------
        static_graph : StaticGraph
            Static graph used for comparison.
        methold : ['jaccard', 'norm'

        Returns
        -------
        scalar
        """
        if not isinstance(static_graph, StaticGraph):
            errmsg = 'Graph is not a StaticGraph: {}'
            raise TypeError(errmsg.format(type(static_graph)))

        if method == 'jaccard':

            # compute the set of edges for each graph
            edges_self = set([tuple(sorted(item)) for item in self.edges()])
            edges_other = set([tuple(sorted(item))
                               for item in static_graph.edges()])

            common_edges = edges_self.intersection(edges_other)

            return len(common_edges)/len(edges_self.union(edges_other))

        elif method == 'norm':

            adj_self = squareform(self.adjacency_matrix)
            adj_other = squareform(static_graph.adjacency_matrix)

            return (1 - np.linalg.norm(adj_self - adj_other) /
                    (self.n_nodes*(self.n_nodes-1)))

        elif method == 'accuracy':
            adj_self = squareform(self.adjacency_matrix)
            adj_other = squareform(static_graph.adjacency_matrix)
            return (1 - np.sum(np.abs(adj_self - adj_other)) /
                    (self.n_nodes*(self.n_nodes-1)))

        elif method == 'accuracy_edges':
            adj_self = squareform(self.adjacency_matrix)
            adj_other = squareform(static_graph.adjacency_matrix)
            return np.sum(adj_other[adj_self]) / self.n_edges
        else:
            errmsg = 'Unknown similarity method: {}'
            raise ValueError(errmsg.format(method))

    def __str__(self):

        string = ['====================',
                  'Static Graph',
                  '====================',
                  'Number of nodes: {:d}'.format(self.n_nodes),
                  'Number of edges: {:d}'.format(self.n_edges),
                  '====================',
                  'Average degree: {:.2f}'.format(
                      np.mean(list(self.degree()))),
                  '====================']

        return '\n'.join(string)
