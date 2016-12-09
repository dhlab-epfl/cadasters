import cv2
import sys
import os
import numpy as np
import pickle
import copy
import argparse
import networkx as nx
import json
import time
from preprocessing import image_filtering, features_extraction
from segmentation import compute_slic
from helpers import show_superpixels, show_polygons, show_class, show_boxes, \
    most_common_label, add_list_to_list, remove_duplicates, bgr2rgb, \
    padding, rotate_image, write_log_file
from graph import edge_cut_minimize_std, assign_feature_to_node, generate_vertices_and_edges
from classification import node_classifier
from polygons import find_parcels, savePolygons, crop_polygon, clean_image_ridge
from text import find_text_boxes, find_false_box, \
    group_box_with_lbl, group_box_with_isolates, crop_box, find_orientation, crop_object
from ocr import recognize_number


parser = argparse.ArgumentParser(description='Cadaster segmentation process')
parser.add_argument('-im', '--cadaster_img', help="filename of the cadaster image", type=str)
parser.add_argument('-out', '--output_path', help='Output directory for results and plots', type=str,
                    default='../outputs')
parser.add_argument('-c', '--classifier', type=str, help='filename of fitted classifier',
                    default='data/svm_classifier.pkl')
parser.add_argument('-p', '--plot', type=bool, help='Show plots (boolean)', default=True)
parser.add_argument('-sp', '--sp_percent', type=float, help='The number of superpixels for '
                    'SLIC algorithm using a percentage of the total number of pixels. '
                    'Give a percentage between 0 and 1', default=0.01)

args = parser.parse_args()

filename_cadaster_img = args.cadaster_img
output_path = args.output_path
if not os.path.exists(output_path):
    os.makedirs(output_path)
filename_classifier = args.classifier
show_plots = args.plot

# ----- PARAMETERS THAT MAY BE CHANGED (recommended not to change much) ----
similarity_method = 'cie2000'
stop_criterion = 0.3
params_slic = {'percent': args.sp_percent, 'numCompact': 25, 'sigma': 2,
               'mode': 'RGB'}
# ---------------------------

# Time process
t0 = time.time()

# Load cadaster image
img = cv2.imread(filename_cadaster_img)

try:
    s = img.shape
except AttributeError:
    sys.exit("Image not loaded correctly or not found")

# FILTERING
print('** FILTERING')
# ---------
img_filt = image_filtering(img)

# SLIC
print('** SLIC')
# -----
original_segments = compute_slic(bgr2rgb(img_filt), params_slic)

# FEATURE EXTRACTION
print('** FEATURE EXTRACTION')
# ------------------
list_dict_features = ['Lab', 'laplacian', 'frangi']
dict_features = features_extraction(img_filt, list_dict_features)

# GRAPHS AND MERGING
print('** GRAPHS AND MERGING')
# ------------------
# Create G graph
G = nx.Graph()
nsegments = original_segments.copy()

G = edge_cut_minimize_std(G, nsegments, dict_features, similarity_method, mst=True, stop_std_val=stop_criterion)

# Keep track of correspondences between 'original superpiels' and merged ones
dic_corresp_label = {sp: np.int(np.unique(nsegments[original_segments == sp]))
                     for sp in np.unique(original_segments)}

if show_plots:
    namefile = os.path.join(output_path, 'sp_merged.jpg')
    show_superpixels(img_filt, nsegments, namefile)

# CLASSIFICATION
print('** CLASSIFICATION')
# ---------------
# Labels : 0 = Background, 1 = Text, 2 = Contours

# Loading fitted classifier
with open(filename_classifier, 'rb') as handle:
    classifications_infos = pickle.load(handle)

clf_params = classifications_infos['parameters']
clf = classifications_infos['classifier']

# Classify new data
list_features_learning = clf_params['features']
node_classifier(G, dict_features, list_features_learning, nsegments,
                clf, normalize_method=clf_params['normalize_method'])


# PARCELS AND POLYGONS
print('** PARCELS AND POLYGONS')
# --------------------
min_size_region = 3  # Regions should be formed at least of min_size_region merged original superpixels
bgclass = 0  # Label of 'background' class

# Find nodes that are classified as background class and that are bigger than min_size_region
bg_nodes_class = [tn for tn in G.nodes() if 'class' in G.node[tn] and G.node[tn]['class'] == bgclass]
bg_nodes_nsp = [tn for tn in bg_nodes_class if 'n_superpix' in G.node[tn]
                and G.node[tn]['n_superpix'] > min_size_region]

# Find parcels and export polygons in geoJSON format
ksize_flooding = 2

listFeatPolygon, dic_polygon = find_parcels(bg_nodes_nsp, nsegments, dict_features['frangi'],
                                                ksize_flooding)

# >>>>>>> HERE POLYGONS SHOULD BE SORTED BY AREA (BIGGER FIRST)
#       so that when they are exported to VTM-Canvas, if a small parcel is situated within a bigger parcel,
#       the small one is on top and is accessible by clicking

# Export geoJSON
filename_geoJson = os.path.join(output_path, 'parcels_polygons.geojson')
savePolygons(listFeatPolygon, filename_geoJson)

if show_plots:
    # Show ridge image for flooding
    ridge2flood = clean_image_ridge(dict_features['frangi'], ksize_flooding)
    filename_ridge = os.path.join(output_path, 'ridges_to_flood.jpg')
    cv2.imwrite(filename_ridge, ridge2flood)

    # Show Polygons
    filename_polygons = os.path.join(output_path, 'polygons.jpg')
    show_polygons(img_filt, dic_polygon, (6, 6, 133), filename_polygons)

# Create directory for cropped polygons
crop_poly_dirpath = os.path.join(output_path, 'cropped_polygons')
if not os.path.exists(crop_poly_dirpath):
    os.makedirs(crop_poly_dirpath)

# Give one unique ID for each polygon, crop each one and save it independently
# Image with labeled polygons
polygons_labels = np.zeros(img_filt.shape[:2])

for bg_nodes, list_tuple in dic_polygon.items():
    for uid, poly in list_tuple:
        mask_polygon_to_fill = np.zeros(polygons_labels.shape, 'uint8')
        cv2.fillPoly(mask_polygon_to_fill, [poly], 255)
        polygons_labels[mask_polygon_to_fill > 0] = bg_nodes
        # >>>>>>>> HERE USE UUID AS LABEL
        # polygons_labels[mask_polygon_to_fill > 0] = uid.int # then do uuid.UUID(int=label) to find uuid

        if show_plots:
            # Crop polygon image and save it
            cropped_polygon_image = crop_polygon(img_filt, [poly])
            filename_cropped_polygon = os.path.join(crop_poly_dirpath, '{}.jpg'.format(uid))
            cv2.imwrite(filename_cropped_polygon, cropped_polygon_image)


# TEXT PIXELS
print('** TEXT PIXELS')
# ------------

# Erode binary version of polygons to avoid touching boundaries to have a 'map'
# of polygons defined by their label. This will be useful to group text elements.
mask_to_erode = cv2.erode(np.uint8(1*(polygons_labels != 0)), np.ones((4, 4), np.uint8))
polygons_labels *= (mask_to_erode > 0)

# This should give text candidates since text is usually inside the polygons
text = (dict_features['frangi'] > 0.4)*polygons_labels
mask_text_erode = np.uint8(255*(text != 0))
# Opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_CLOSE, kernel)
mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_OPEN, kernel)
text *= (mask_text_erode > 0)


# CORRECTION OF TEXT SP USING POLYGONS INFORMATION
# ------------------------------------------------
# We use the fact that a text zone (label or street name) is generally inside a parcel. So we enlarge the
# number of possible text candidates by adding the ridges present inside polygons.

# Get id node of text segment
text_node_id = np.unique(original_segments*(text != 0))
# remove BG id
if 0 in text_node_id:
    text_node_id = list(text_node_id)
    text_node_id.remove(0)

# set attribute class of text node to be 1
attr_text = {tn: 1 for tn in text_node_id}
G.add_nodes_from([tn for tn in text_node_id if dic_corresp_label[tn] < 0])  # add only nodes that have been removed
nx.set_node_attributes(G, 'class', attr_text)

# Correct nodes class by removing the text node from previous eventual merged group of nodes
node_to_correct = {tn: dic_corresp_label[tn] for tn in text_node_id if dic_corresp_label[tn] < 0}

for k, v in node_to_correct.items():
    G.node[v]['id_sp'].remove(k)
    pix = original_segments == k
    nsegments[pix] = k
    assign_feature_to_node(G, k, nsegments, dict_features)

# Update correspondancies label nodes
dic_corresp_label = {sp: np.int(np.unique(nsegments[original_segments == sp]))
                     for sp in np.unique(original_segments)}

# For each text node, look which is its corresponding polygon and label the text with the polygon label
# For that we look at which label is the most present (in the case of multiple labels appearing)
attr_lbl_poly = {tn: most_common_label(polygons_labels[original_segments == tn]) for tn in text_node_id}
nx.set_node_attributes(G, 'lbl_poly', attr_lbl_poly)


# CLASSIFICATION OF LEFT UNCLASSIFIED NODES
# -----------------------------------------
# Get class attribute from every node
attribute_node_class = {n: G.node[dic_corresp_label[n]]['class'] for n in np.unique(original_segments)
                        if 'class' in G.node[dic_corresp_label[n]]}
attribute_node_lbl_poly = {n: G.node[dic_corresp_label[n]]['lbl_poly'] for n in np.unique(original_segments)
                           if 'lbl_poly' in G.node[dic_corresp_label[n]]}

# New graph with original superpixels
O = nx.Graph()

# Construct vertices and neighboring edges
vertices, edges = generate_vertices_and_edges(original_segments)
# Add nodes
O.add_nodes_from(vertices)
nx.set_node_attributes(O, 'class', attribute_node_class)
nx.set_node_attributes(O, 'lbl_poly', attribute_node_lbl_poly)
O.add_edges_from(edges)

# Show class
if show_plots:
    namefile_class = os.path.join(output_path, 'predicted3class.jpg')
    show_class(nsegments, G, namefile_class)


# BOUNDING BOXES
print('** BOUNDING BOXES')
# ---------------

# Build text graphs
# -----------------
Tgraph = nx.Graph()

text_nodes = {tn: O.node[tn]['class'] for tn in O.nodes() if O.node[tn]['class'] == 1}
attr_lbl_poly = {tn: O.node[tn]['lbl_poly'] for tn in O.nodes() if 'lbl_poly' in O.node[tn]}

Tgraph.add_nodes_from(text_nodes.keys())
nx.set_node_attributes(Tgraph, 'class', text_nodes)
Tgraph.add_nodes_from(attr_lbl_poly.keys())
nx.set_node_attributes(Tgraph, 'lbl_poly', attr_lbl_poly)

# Add edges only between text nodes
for tn in text_nodes:
    adjacent_nodes = [an[0] for an in O[tn].items() if not an[1]]  # check only nodes that are not linked yet
    for an in adjacent_nodes:
        if O.node[tn]['class'] == O.node[an]['class']:  # add edge if link between two text superpixels
            Tgraph.add_edge(tn, an)

# Find boxes
# ----------
box_id = 0
listBox = find_text_boxes(Tgraph, original_segments)


reference_boxes = copy.deepcopy(listBox)
# TRY TO ELIMINATE FALSE POSITIVES
# --------------------------------
boxes_false = find_false_box(listBox, reference_boxes)
boxes_with_lbl = [b for b in listBox if b.lbl_polygon is not None]
true_non_labeled = [b for b in listBox if b not in boxes_false and b not in boxes_with_lbl]

# GROUP BOXES that have labels
# ----------------------------
maximum_distance = 15
groupedBox = group_box_with_lbl(boxes_with_lbl, boxes_false, maximum_distance)

# Maybe check also too big boxes that may appear with groupbox
# and add it to the previous list
add_list_to_list(boxes_false, find_false_box(boxes_with_lbl, reference_boxes))
# Consider only unique elements
boxes_false = remove_duplicates(boxes_false)

# True boxes
boxes_true = true_non_labeled.copy()
add_list_to_list(boxes_true, [b for b in boxes_with_lbl if b not in boxes_false])
# Flatten list and consider only unique elements
boxes_true = remove_duplicates(boxes_true)

# GROUP BOXES that are true with small false elements
# ----------------------------
final_boxes = boxes_true.copy()
maximum_distance = 7
groupedBox = group_box_with_isolates(final_boxes, boxes_false, maximum_distance)

# Check for false box one last time
add_list_to_list(boxes_false, find_false_box(final_boxes, reference_boxes))
boxes_false = remove_duplicates(boxes_false)

final_boxes = [b for b in final_boxes if b not in boxes_false]


# SHOW BOXES
if show_plots:
    # All
    img_allBox = img_filt.copy()
    show_boxes(img_allBox, final_boxes, (0, 255, 0))
    show_boxes(img_allBox, boxes_false, (0, 0, 255))
    cv2.imwrite(os.path.join(output_path, 'allBox.jpg'), img_allBox)

    # Final boxes
    filename_finalBox = os.path.join(output_path, 'finalBox.jpg')
    show_boxes(img_filt.copy(), final_boxes, (0, 255, 0), filename_finalBox)


# PROCESS BOX (ROTATE, ...) AND SAVE IT
print('** PROCESS BOX')
# -------------------------------------
# Create directory to save digits image
path_digits = os.path.join(output_path, 'digits')
if not os.path.exists(path_digits):
    os.makedirs(path_digits)

for box in final_boxes:
    # Expand box
    box.expand_box(padding=2)

    # Crop
    crop_imgL, (xmin, xmax, ymin, ymax) = crop_box(box, dict_features['Lab'][:, :, 0])
    cropped_number = img[ymin:ymax + 1, xmin:xmax + 1].copy()

    # Binarize to have the general shape so that we can dilate it as a
    # blob and find the orientation of the blob

    # Binarization
    blur = cv2.GaussianBlur(crop_imgL, (3, 3), 0)
    ret, binary_crop = cv2.threshold(blur, 0, np.max(crop_imgL), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_crop = np.uint8(255 * (binary_crop < 1))
    # Morphology _ dilation
    dilated = cv2.dilate(inv_crop, np.ones((5, 5), np.uint8))
    # Find orientation with PCA
    center, eigvect, angle = find_orientation(dilated)
    # Plot orientation
    #     img2draw = inv_crop.copy()
    #     diagonal = eigvect[0]
    #     linex = int(center[0] + diagonal[0]*20)
    #     liney = int(center[0] + diagonal[1]*20)
    #     cv2.circle(img2draw, (int(center[0]), int(center[1])), 1, 128, -1)
    #     cv2.line(img2draw, (int(center[0]), int(center[1])), (linex, liney), 128, 1)

    # Rotate image to align it horizontally
    img_pad = padding(inv_crop, 0)
    rotated_img = rotate_image(img_pad, angle)

    # Recrop image
    final_digit_bin, (y, x, h, w) = crop_object(rotated_img)
    final_digit_bin = final_digit_bin > 1  # >>>>> NOT SURE USEFUL

    # Format image to be saved into 3 channel uint8
    final_digit_bin = np.uint8(255 * final_digit_bin)
    formated_digit_img = np.dstack([final_digit_bin] * 3)

    # Rotate original image
    rotated_number = rotate_image(padding(cropped_number, 255), angle)
    rotated_number = rotated_number[x:x + w, y:y + h, :]

    # Save cropped image
    if show_plots:
        cv2.imwrite(os.path.join(path_digits, '{}.jpg'.format(box.box_id)), formated_digit_img)
        cv2.imwrite(os.path.join(path_digits, '{}_original.jpg'.format(box.box_id)), rotated_number)

    # RECOGNIZING NUMBERS
    # --------------------
    # Number of digits per number  <<<<<< find a way to estimate before recognition
    # final_skelton = skeletonize(final_digit_bin)
    # projx = np.sum(final_skelton > 0, axis=0)
    # More than 2 pixels per colums non zero, digits is composed of at least 4 columns
    # number_of_digits = find_pattern(projx > 2, [True] * 4)
    number_of_digits = 4
    prediction, proba = recognize_number(rotated_number, number_of_digits=number_of_digits)

    # Save in JSON file
    data = {'number': prediction, 'confidence': proba}
    filename_json = os.path.join(path_digits, '{}_json.txt'.format(box.box_id))
    with open(filename_json, 'w') as fjson:
        json.dump(data, fjson)

# LOG FILE
print('** LOG FILE')
# ---------
# Write Log file
elapsed_time = time.time() - t0
log_filename = os.path.join(output_path, 'log.txt')
write_log_file(log_filename, elapsed_time=elapsed_time, cadaster_filename=filename_cadaster_img,
               classifier_filename=filename_classifier, size_image=img_filt.shape,
               params_slic=params_slic, list_dict_features=list_dict_features,
               similarity_method=similarity_method, stop_criterion=stop_criterion)