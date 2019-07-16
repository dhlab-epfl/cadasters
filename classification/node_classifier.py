import numpy as np
from classification import construct_feature_vector_classification
from typing import List


def node_classifier(graph, dict_feature: dict, list_features: List[str], map_segments: np.array,
                    classifier, normalize_method: bool=None, graph_nodes=-1, new_attribute_name: str='class'):
    """
    Classifies the graph's nodes

    :param graph: entire graph where some/all nodes are classified
    :param graph_nodes: list of nodes to be classified (-1 for all nodes)
    :param dict_feature: dictionnary containing the features for all the image
    :param list_features: list of features to be used (Be attentive to the fact that these features need
                            to be the same as the ones used for training) The possible features are in
                            dict_features : 'Lab' laplacian' 'frangi' 'gabor' 'sobel'
    :param map_segments: map of segments/regions/superpixels
    :param classifier: fitted classifier (the same used for training!)
    :param normalize_method: to normalize data (should be the same as during training)
    :param new_attribute_name: name of the attribute of the node where the class label is stored
    """

    # >>>> Attention to the order of the features, needs to be the same as in training data!!!
    if graph_nodes < 0:
        graph_nodes = graph.nodes()

    # CREATE FEATURE VECTOR
    data_features = []
    for nod in graph_nodes:
        pix = map_segments == nod
        feat_vector = construct_feature_vector_classification(pix, dict_feature, list_features)

        # Add this to data_features
        data_features.append(feat_vector)

    data_features = np.array(data_features)

    # CLASSIFYING REGIONS
    # -------------------
    # Normalize
    if normalize_method:
        data_features = normalize_method().fit_transform(data_features)

    predicted_lbl = classifier.predict(data_features)

    # ASSIGN LABEL TO NODES
    # ---------------------
    for ilbl, s in enumerate(graph_nodes):
        graph.node[s][new_attribute_name] = predicted_lbl[ilbl]