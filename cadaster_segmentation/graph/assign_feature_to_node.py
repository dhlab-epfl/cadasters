import numpy as np


def assign_feature_to_node(graph, node_lbl, list_segments, dict_features):
    """
    To each node of the graph, assigns its characteristics (features) that have been previously
    calculated for all images and stored in dict_features
    :param graph: Graph of superpixels/regions
    :param node_lbl: label of the node
    :param list_segments: map of labelled segments
    :param dict_features: dictionary containing all the computed features
    """

    # Pixels that belong to the node
    pixLbl = list_segments == node_lbl

    # Features
    # Lab
    if 'Lab' in dict_features.keys():
        l_py = dict_features['Lab'][0:, 0:, 0]
        a_py = dict_features['Lab'][0:, 0:, 1]
        b_py = dict_features['Lab'][0:, 0:, 2]

        graph.node[node_lbl]['meanL'] = np.mean(l_py[pixLbl])
        graph.node[node_lbl]['meanA'] = np.mean(a_py[pixLbl])
        graph.node[node_lbl]['meanB'] = np.mean(b_py[pixLbl])

    # Centers
    if 'centers' in dict_features.keys():
        graph.node[node_lbl]['x'] = dict_features['centers'][node_lbl][1]
        graph.node[node_lbl]['y'] = dict_features['centers'][node_lbl][0]

    # Laplacian
    if 'laplacian' in dict_features.keys():
        graph.node[node_lbl]['lapL'] = np.mean(dict_features['laplacian'][pixLbl])
    # Sobel
    if 'sobelx' in dict_features.keys():
        graph.node[node_lbl]['sobx'] = np.mean(dict_features['sobelx'][pixLbl])
    if 'sobely' in dict_features.keys():
        graph.node[node_lbl]['soby'] = np.mean(dict_features['sobely'][pixLbl])

    # Combined directions Gabor
    if 'gabor' in dict_features.keys():
        graph.node[node_lbl]['gabor_mean'] = np.mean(dict_features['gabor'][pixLbl])
        graph.node[node_lbl]['gabor_std'] = np.std(dict_features['gabor'][pixLbl])

    # Frangi Filter (Ridges)
    if 'frangi' in dict_features.keys():
        graph.node[node_lbl]['frangi_mean'] = np.mean(dict_features['frangi'][pixLbl])
        graph.node[node_lbl]['frangi_std'] = np.std(dict_features['frangi'][pixLbl])
