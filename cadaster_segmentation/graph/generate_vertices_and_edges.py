import numpy as np


def generate_vertices_and_edges(map_superpixels):
    """
    Generate connected graph with vertices and edges connecting neighbouring vertices
    :param map_superpixels: superpiexls resulting from superpixels segmentation (map of labels)
    :return: list of vertices, and list of tuples indicating edges
    """
    # get unique labels
    vertices = np.unique(map_superpixels)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices, np.arange(len(vertices))))
    map_superpixels = np.array([reverse_dict[x] for x in map_superpixels.flat]).reshape(map_superpixels.shape)

    # create edges
    down = np.c_[map_superpixels[:-1, :].ravel(), map_superpixels[1:, :].ravel()]
    right = np.c_[map_superpixels[:, :-1].ravel(), map_superpixels[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges, axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:, 0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    # edges = [[vertices[x%num_vertices],
    #           vertices[x/num_vertices]] for x in edges]
    #             #  These last 2 lines may produce an error in the future
    # Replace by :
    edges = [[vertices[np.int(x % num_vertices)],
              vertices[np.int(x / num_vertices)]] for x in edges]

    return vertices, edges
