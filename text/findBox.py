import numpy as np
import networkx as nx
import cv2
from text.Box import Box
from helpers import most_common_label
from graph import contract_nodes
from helpers import merge_segments


def find_text_boxes(graph, original_segments):
    """
    Uses the graph representing text elements to bound text by boxes and return a list of
    Box objects

    :param graph: graph with text elements
    :param original_segments: original map of segments/superpixels
    :return: list of Box objects
    """

    listBox = list()

    # Find subgraphs
    subT = list(nx.connected_component_subgraphs(graph))
    subT = sorted(subT, key=len, reverse=True)

    # For each subgraph, make binary image of the elements using segments
    for st in subT:
        # Make binary image and list of polygon labels
        binary_image_text = np.zeros(original_segments.shape, 'uint8')
        list_lbl_poly = list()
        for n in st.nodes():
            pix = original_segments == n
            binary_image_text[pix] = 1
            if 'lbl_poly' in graph.node[n]:
                list_lbl_poly.append(graph.node[n]['lbl_poly'])

        # Find in which polygon the text is
        if list_lbl_poly:
            # take the most common, different from zero
            lbl_poly = most_common_label(list_lbl_poly)
        else :
            lbl_poly = None

        # Find contours of text group
        i, contours, h = cv2.findContours(binary_image_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Rotated rectangle associated with it
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        box_obj = Box(contours[0], box, lbl_poly)
        # Add box and contour information in a dictionary
        listBox.append(box_obj)

# >>>>>>>>>>>>>>>>>>>>>>>>  Nécessaire ?
#         txtsegments = original_segments.copy()
#         # Merge and contract
#         if len(st.nodes()) > 1:
#             merge_segments(txtsegments, st.nodes(), min(np.unique(txtsegments))-1) ### <<<<<< THIS IS NOT USEFUL...
#             contract_nodes(graph, st.nodes(), min(np.unique(txtsegments)))  # This neither...
#             new_node_lbl = min(np.unique(txtsegments))
#         else:
#             new_node_lbl = st.nodes()[0]
#
#         graph.node[new_node_lbl]['contours'] = contours[0]
#         graph.node[new_node_lbl]['boxPoints'] = box
#         graph.node[new_node_lbl]['boxID'] = box_obj.box_id

    return listBox
# --------------------------------------------------------------------------


def find_false_box(list_box_to_process, reference_boxes):
    """
    Finds the possible false positive boxes among all boxes in a list according to some criteria such as
    area and ratios (these criteria are been updated with more)

    :param list_box_to_process: list of boxes
    :param reference_boxes:
    :return: list of possible false positive boxes
    """

    false_positives = list()

    # Box areas
    box_areas = np.array([box.box_area for box in reference_boxes])
    # find small boxes
    false_positives.append([b for b in list_box_to_process if b.box_area < 0.3 * np.mean(box_areas)])
    # find big boxes
    false_positives.append([b for b in list_box_to_process if b.box_area > np.mean(box_areas) + 3 * np.std(box_areas)])

    # Box ratio
    list_bratio = np.array([box.box_ratio for box in reference_boxes])
    false_positives.append([b for b in list_box_to_process if b.box_ratio > np.mean(list_bratio) + 3 * np.std(list_bratio)])

    # Possible false negatives
    boxes_false = [item for inner_list in false_positives for item in inner_list]

    # counts = [[x, boxes_false.count(x)] for x in set(boxes_false)]
    # # False negative if it appears at least once (this can be changed) in the list
    # boxes_false = [t[0] for t in counts if t[1] > 0]

    return boxes_false


# Additional criterion for further use ... ?

# # Solidity
# list_solid = np.array(box.solidity for box in listBox])
# false_positives.append([b for b in listBox if b.solidity > np.mean(list_solid) + 0.5*np.std(list_solid) ]) # 0.8

# # Defect
# # list_defects = np.array([dic_box[key].defectmax for key in dic_box.keys()])
# # list_mindimension = np.array([np.min(dic_box[key].dimensions) for key in dic_box.keys()])
# # mask = list_defects > 2*list_mindimension # 2*np.min(dimensions)
# # false_positives.append(list(keys[mask]))
# # # Extent
# # list_extent = np.array([dic_box[key].extent for key in dic_box.keys()])
# # mask = list_extent > np.mean(list_extent) + np.std(list_extent) # 0.7
# # false_positives.append(list(keys[mask]))
# # Surface Ratio
# # list_sratio = np.array([dic_box[key].surf_ratio for key in dic_box.keys()])
# # mask = list_sratio > np.mean(list_sratio) + np.std(list_sratio) # 0.8
# # false_positives.append(list(keys[mask]))
