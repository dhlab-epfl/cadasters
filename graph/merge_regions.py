#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
import networkx as nx
from graph import assign_feature_to_node, generate_vertices_and_edges, measure_similarity, contract_nodes
from helpers import merge_segments


def edge_cut_with_mst(graph, list_segments, dict_features, similarity_method, stop_criterion=5.0e-5):
    if stop_criterion == None:
        stop_criterion = 5.0e-5
    # Construct vertices and neighboring edges
    vertices, edges = generate_vertices_and_edges(list_segments)
    # Add nodes and edges
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    if 'centers' in dict_features.keys():
        # Compute region centers
        gridx, gridy = np.mgrid[:list_segments.shape[0], :list_segments.shape[1]]
        vertices = np.unique(list_segments)
        centers = {v : [gridy[list_segments == v].mean(), gridx[list_segments == v].mean()] for v in vertices}
        dict_features['centers'] = centers

    # Feature assignment to each node
    for segLbl in np.unique(list_segments):
        assign_feature_to_node(graph, segLbl, list_segments, dict_features)

    tmp = measure_similarity(graph, similarity_method, exponential= None)

    graph = nx.minimum_spanning_tree(graph, weight='sim')

    # Sort edges by similarity
    edges_sorted = sorted(graph.edges(data=True), key=lambda t: t[2].get('sim',1), reverse=True)

    halfG = round(len(graph.edges())/2)
    prev_max_sim = 0
    for e in edges_sorted:
        # Find subgraphs
        subG = list(nx.connected_component_subgraphs(graph))
        subG = sorted(subG, key=len, reverse=True)

        max_sim = 0
        list_sum = []
        for sg in subG:
            if len(sg.nodes()) > 1:
                sum_sim = 0
                # Compute sum_similarity in each subgraph
                for (e1,e2) in sg.edges():
                    current_sim = sg[e1][e2]['sim']
                    sum_sim = sum_sim + current_sim
                list_sum.append(sum_sim/len(sg.edges()))
            else :
                break

        new_max_sim = np.mean(list_sum)
        diff_max_sim = np.abs(new_max_sim - prev_max_sim)
        prev_max_sim = new_max_sim
        if diff_max_sim < stop_criterion:
            print('Diff :', diff_max_sim)
            break
        if round(len(graph.edges())) < halfG:
            print('Half edges removed')
            break

        # Remove edge with maximum dissimilarity
        graph.remove_edge(e[0],e[1])

    # MERGING
    # -------------
    subG = list(nx.connected_component_subgraphs(graph))
    subG = sorted(subG, key=len, reverse=True)
    for sg in subG:
        if len(sg.nodes()) > 1:
            # Merge segements
            merge_segments(list_segments, sg.nodes(), min(np.unique(list_segments))-1)
            # Contract nodes
            contract_nodes(graph, sg.nodes(), min(np.unique(list_segments)))
        else:
            break

    # UPDATE ATTRIBUTE NODES
    # ----------------------
    if 'centers' in dict_features.keys():
        gridx, gridy = np.mgrid[:list_segments.shape[0], :list_segments.shape[1]]
        vertices = np.unique(list_segments)
        centers = {v : [gridy[list_segments == v].mean(), gridx[list_segments == v].mean()] for v in vertices}
        dict_features['centers'] = centers

    # Feature assignment to each node
    for segLbl in np.unique(list_segments):
        assign_feature_to_node(graph, segLbl, list_segments, dict_features)

    return graph
# ---------------------------------------------------------------------------------------------------


def edge_cut_minimize_std(graph: nx.Graph, slic_segments: np.array, dict_features: dict,
                          similarity_method: str, mst: bool=True, stop_std_val: float=5.0e-1) -> nx.Graph:
    """
    Segments image by removing edges in the graph representation of the image. At each step, edges with high
    weight are removed and 'subgraphs' are formed (new connected components). In each subgraph the standard
    deviation of the weights is computed and if it is greater than a certain threshold (stop_std_val),
    the highest weight is removed. This is done until all high weights have been removed and the regions
    (vertices) have been merged

    :param graph: Graph of the image to be segmented
    :param slic_segments: map of the segments' labels
    :param dict_features: dictionnary containing all the features (to compute nodes characteristics
                            and edges similarities)
    :param similarity_method: similarity method to use in measure_similarity function (see function's doc)
    :param mst: boolean to use minimum spanning tree during processing. The use of MST speeds up the process,
                but may change the results a bit (less accurate probably -> this has not been verified)
    :param stop_std_val: maximum intraregion dissimilarity variation. A value greater than stop_std_val
                        indicates that there are edges linking dissimilar vertices. When value < stop_std_val,
                        the elements of the subgraph can be merged together to form an homogeneous region.
    :return: The graph of the segmented image.
    """

    # Construct vertices and neighboring edges
    vertices, edges = generate_vertices_and_edges(slic_segments)
    # Add nodes and edges
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)

    # Feature assignment to each node
    for segments_lbl in np.unique(slic_segments):
        assign_feature_to_node(graph, segments_lbl, slic_segments, dict_features)

    tmp = measure_similarity(graph, similarity_method)

    if mst:
        graph = nx.minimum_spanning_tree(graph, weight='sim')

    # Minimise std inside subgraphs
    while True:
        # Compute subgraphs and sort them by length
        subG = list(nx.connected_component_subgraphs(graph))
        subG = sorted(subG, key=len, reverse=True)

        # When all subgraphs with n_nodes >=3 have been treated, finish with 2 nodes and end
        if len(subG[0].edges()) == 1:
            # Merge the 2-nodes subgraphs still remaining
            for sg in subG:
                if len(sg.nodes()) > 1:  # not isolates
                    merge_segments(slic_segments, sg.nodes(), min(np.unique(slic_segments)) - 1)
                    contract_nodes(graph, sg.nodes(), min(np.unique(slic_segments)))
                else:
                    break
            break

        for sg in subG:
            if len(sg.edges()) > 1:  # 2 edges min
                # Sort edges by similarity (most dissimilar first)
                sg_edges_sorted = sorted(sg.edges(data=True), key=lambda t: t[2].get('sim', 1), reverse=True)
                # Compute standard deviation of similarities of the subgraph
                std_sg = np.std([sg[e1][e2]['sim'] for (e1, e2) in sg.edges()])

                # If there are dissimilar vertices connected disconnect it, otherwise merge
                if std_sg > stop_std_val:
                    graph.remove_edge(*sg_edges_sorted[0][:2])
                else:
                    merge_segments(slic_segments, sg.nodes(), min(np.unique(slic_segments)) - 1)
                    contract_nodes(graph, sg.nodes(), min(np.unique(slic_segments)))
                    # print('Merging {} nodes'.format(len(sg.nodes())))
            else:
                break

        if len(graph.nodes()) < 1:
            print('Error, all nodes deleted...')
            return

    # UPDATE ATTRIBUTE NODES
    # ----------------------

    # Feature assignment to each node
    for segments_lbl in np.unique(slic_segments):
        assign_feature_to_node(graph, segments_lbl, slic_segments, dict_features)

    return graph
# -----------------------------------------------------------
