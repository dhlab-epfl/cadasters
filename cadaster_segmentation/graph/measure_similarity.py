import numpy as np
import sys
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000 as cie2000


def measure_similarity(graph, method='cie2000', exponential=None):
    """
    Measures similarity (or "distance") between nodes connected by an edge. This value is the "weight" of the edge.
    :param graph: Graph to calculate the edges
    :param method: 'cie2000' : (recommended) Distance based on cie2000 LAB distance in LAB color space
                                (see reference report/paper)
                    'edges' : Distance based on the edges (edge detection) of the image.
                                Vesselness measure (ridge) and Laplacian are used
                    'coloredge' : A mixture between cie2000 and edges measure
    :param exponential : apply exponential to the calculated distance (-1 or 1 to choose negative or positive exponential)
                        (under development, use at your own risk)
    :return: mean value of similarity in the graph (may be useful to get an idea)
    """
    list_sim = []
    if method == 'cie2000':
        for (e1, e2) in graph.edges():
            color1 = LabColor(graph.node[e1]['meanL'], graph.node[e1]['meanA'], graph.node[e1]['meanB'])
            color2 = LabColor(graph.node[e2]['meanL'], graph.node[e2]['meanA'], graph.node[e2]['meanB'])
            distance = cie2000(color1, color2)

            if exponential:  # -1 or 1
                exp_distance = np.exp(exponential * distance)
                graph.edge[e1][e2]['exp_sim'] = exp_distance

            # Add similarity measure to edge
            graph.edge[e1][e2]['sim'] = distance
            # Add distance to list
            list_sim.append(distance)
    # -------------------------

    elif method == 'edges':
        for (e1, e2) in graph.edges():
            distance = np.absolute(graph.node[e1]['frangi_mean'] - graph.node[e2]['frangi_mean']) + \
                       np.absolute(graph.node[e1]['lapL'] - graph.node[e2]['lapL'])

            if exponential:  # -1 or 1
                exp_distance = np.exp(exponential * distance)
                graph.edge[e1][e2]['exp_sim'] = exp_distance

            # Add similarity measure to edge
            graph.edge[e1][e2]['sim'] = distance
            # Add distance to list
            list_sim.append(distance)
    # -------------------------

    elif method == 'coloredge':
        for (e1, e2) in graph.edges():
            color1 = LabColor(graph.node[e1]['meanL'], graph.node[e1]['meanA'], graph.node[e1]['meanB'])
            color2 = LabColor(graph.node[e2]['meanL'], graph.node[e2]['meanA'], graph.node[e2]['meanB'])

            distance = 0.2 * np.absolute(graph.node[e1]['frangi_mean'] - graph.node[e2]['frangi_mean']) + \
                       0.1 * np.absolute(graph.node[e1]['frangi_std'] - graph.node[e2]['frangi_std']) + \
                       0.7 * cie2000(color1, color2)

            if exponential:  # -1 or 1
                exp_distance = np.exp(exponential * distance)
                graph.edge[e1][e2]['exp_sim'] = exp_distance

            # Add similarity measure to edge
            graph.edge[e1][e2]['sim'] = distance
            # Add distance to list
            list_sim.append(distance)
    # -------------------------
    else:
        sys.exit('The chosen similarity method {} is not implemented'.format(method))

    val_sim = np.mean(list_sim)
    return val_sim
