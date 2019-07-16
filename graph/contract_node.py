#!/usr/bin/env python
__author__ = 'solivr'

from typing import List


def contract_nodes(G, nodes: List[int], new_node: int, attr_dict: dict=None, **attr):
    """
    Contracts the edges of the nodes in the set "nodes". Keeps the relations of edges but the nodes'
    attributes need to be computed again (since their value changes)

    :param G: Graph to be updated
    :param nodes: list of nodes to contract in a single one
    :param new_node: label of new node
    :param attr_dict: attributes
    :param attr:
    :return: the updated graph (optional since it is already updated in G)
    """
    # Calculate how many 'original'/'initial' superpixels have been merged
    # How many were already merged
    previous_nsp = [G.node[t]['n_superpix'] for t in nodes if 'n_superpix' in G.node[t]]
    # Add both
    n_superpix = len(nodes) - len(previous_nsp) + sum(previous_nsp)

    # Save 'original' id of merged nodes
    id_sp = [G.node[n]['id_sp'] for n in nodes if 'id_sp' in G.node[n]]
    id_sp.extend(nodes)

    # Attributes of nodes disappear but value of edge stays
    # Add the node with its attributes
    G.add_node(new_node, attr_dict, **attr)
    G.node[new_node]['n_superpix'] = n_superpix
    G.node[new_node]['id_sp'] = id_sp
    # Create the set of the edges that are to be contracted
    cntr_edge_set = G.edges(nodes, data=True)
    # Add edges from new_node to all target nodes in the set of edges that are to be contracted
    # Possibly also checking that edge attributes are preserved and not overwritten,
    # especially in the case of an undirected G (Most lazy approach here would be to return a
    # multigraph so that multiple edges and their data are preserved)
    G.add_edges_from(map(lambda x: (new_node, x[1], x[2]), cntr_edge_set))
    # Remove the edges contained in the set of edges that are to be contracted, concluding the contraction operation
    G.remove_edges_from(cntr_edge_set)
    # Remove the nodes as well
    G.remove_nodes_from(nodes)
    # Return the graph
    return G