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
"""Definition of static graph models.

Author : Ronan Hamon
"""
import numpy as np

from .graph import StaticGraph


def stochastic_block_model(n_nodes=None, n_communities=None, partition=None,
                           probabilities=None):
    """Generate an instance of the stochastic block model.

    Parameters
    ----------
    n_nodes : int or None, optional
        Desired number of nodes. If None, `partition` should be given.
    n_communities : int or None, optional
        Desired number of nodes. If None, `partition`, `communities` or
        `probabilities` should be given.
    partition : dict or list or None, optional
        Dictionary indicating the community of each node. Names of nodes and
        communities can be int or str but should be distinct. If None, the
        allocation of nodes in communities is randomly drawn.
    probabilities : array-like
        Array giving the probability of edge between nodes according to their
        community. Indexing is made using the sorting order on names of the
        community. If None, random probabilities are drawn.

    Returns
    -------
    StaticGraph

    Raises
    ------
    ValueError

    """
    if partition:
        if isinstance(partition, dict):
            nodes = list(partition.keys())
            communities = set(partition.values())
        else:
            nodes = list(range(len(partition)))
            communities = sorted(set(partition))
            partition = dict(zip(nodes, partition))

        if n_nodes:
            if n_nodes != len(nodes):
                errmsg = '`n_nodes` ({}) and the length of the partition ({}) \
should be the same.'
                raise ValueError(errmsg.format(n_nodes, len(partition)))
        else:
            n_nodes = len(nodes)

        if n_communities:
            if n_communities != len(communities):
                errmsg = '`n_communities` ({}) and the number of partitions \
({}) should be the same.'
                raise ValueError(errmsg.format(
                    n_communities, len(communities)))
        else:
            n_communities = len(communities)

    else:
        if not n_nodes:
            errmsg = 'Either `n_nodes` or `partition` should be given.'
            raise ValueError(errmsg)
        nodes = list(range(n_nodes))

        if not n_communities:
            errmsg = 'Either `n_communities` or `partition` should be given.'
            raise ValueError(errmsg)
        communities = list(range(n_communities))

        partition = dict(zip(nodes, np.random.choice(
            range(n_communities), n_nodes, replace=True)))

    if probabilities is None:
        ind_diag = np.eye(n_communities, dtype=np.bool)
        ind_tri = np.triu(
            np.ones((n_communities, n_communities), dtype=np.bool), 1)
        probabilities = np.zeros((n_communities, n_communities))
        probabilities[ind_diag] = np.random.rand(np.sum(ind_diag)) * 0.2 + 0.8
        probabilities[ind_tri] = np.random.rand(np.sum(ind_tri)) * 0.3
        probabilities[ind_tri.T] = probabilities[ind_tri]
    else:
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)

        if (probabilities.shape[0] != n_communities or
                probabilities.shape[1] != n_communities):
            errmsg = 'The number of communities ({}) and the shape of \
`probabilities` ({}) are not consistent .'
            raise ValueError(errmsg.format(n_communities, probabilities.shape))

    # Construction of the graph
    graph = StaticGraph()

    for node, community in partition.items():
        graph.add_node(node, community=community)

    for u in range(n_nodes):
        for v in range(u+1, n_nodes):
            p_uv = probabilities[partition[nodes[u]], partition[nodes[v]]]
            if np.random.rand() < p_uv:
                graph.add_edge(u, v)

    return graph
