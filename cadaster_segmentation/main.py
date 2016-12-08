import cv2
import sys
import networkx as nx
import numpy as np
import time
import datetime
import copy
import os
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from preprocessing import image_filtering, features_extraction
from segmentation import compute_slic, skeletonize
from classification import make_labeled_image, prepare_classifier, node_classifier, classification_errors, \
    classification_with_min_std, classification_with_neighbors
from graph import merge_regions_using_threshold, edge_cut_minimize_std, edge_cut_with_mst, assign_feature_to_node, generate_vertices_and_edges
from helpers import show, most_common_label, lists, padding, rotate_image, find_pattern
from polygons import find_parcels, savePolygons, crop_polygon, error_polygon, compute_images_for_flooding
from text import findTextBoxes, find_false_box, group_box_with_isolates, group_box_with_lbl, fill_outside_box, crop_box, find_orientation, crop_object, error_box
from ocr import recognize_number
from helpers.channel_ordering import bgr2rgb

def launch_cadaster_segmentation(path_image, cad, filename_img, use_lbl_img, path_lbl_img, params_slic, similarity_method, \
                                 stop_criterion, path_cropped_polygon, path_logfile, img_truth_polygons, img_truth_digits, name_directory):

    path_result_img = '../data_images/result_img/' + name_directory
    if not os.path.exists(path_result_img):
        os.makedirs(path_result_img)

    merging_method = 'std'
    classification_with_corrections = False
    isol_sim_method = 'cie2000'

    classification_method = 'svm'
    normalize_method = MinMaxScaler

    # To compute execution time
    t0 = time.time()
    d0 = datetime.datetime.now()


    # Load cadaster image
    img = cv2.imread(path_image + filename_img)
    if use_lbl_img:
        img_label = cv2.imread(path_lbl_img)

    try:
        s = img.shape
    except:
        sys.exit("Image not loaded correctly or not found")



    # FILTERING
    # ---------
    filters = ['nlmeans']
    img_filt = image_filtering(img, filters)

    # SLIC
    # -----
    segments = compute_slic(bgr2rgb(img_filt), params_slic)

    # LABELED SUPERPIXELS
    # -------------------
    if use_lbl_img:
        dic_true_class = make_labeled_image(img_label, segments, '')

    # FEATURE EXTRACTION
    # ------------------
    list_dict_features = ['Lab', 'laplacian', 'frangi']
    dict_features = features_extraction(img_filt, list_dict_features)



    # GRAPHS AND MERGING
    # ------------------
    # Create G graph
    G = nx.Graph()

    nsegments = segments.copy()

    # ----------------------------

    if merging_method == 'thresh':
        G, sim_mean = merge_regions_using_threshold(G, nsegments, dict_features, similarity_method, plot = False,
                                                       imcolor=img_filt, path='../data_images/test_merging/', ind_namefile=1)
        updt_nnodes = len(G.nodes())
        nnodes = updt_nnodes + 1 # must be grater than updt_nnodes
        i = 2
        while (nnodes - updt_nnodes) != 0:
            nnodes = updt_nnodes
            G, tmp = merge_regions_using_threshold(G, nsegments, dict_features, similarity_method, val_similarity=sim_mean,
                                                   plot = True, imcolor=img_filt, path='../data_images/test_merging/', ind_namefile=i)
            updt_nnodes = len(G.nodes())
            i += 1

    elif merging_method == 'mst':
        G = edge_cut_with_mst(G, nsegments, dict_features, similarity_method, stop_criterion=stop_criterion)

    elif merging_method == 'std':
        G = edge_cut_minimize_std(G, nsegments, dict_features, similarity_method, mst=True, stop_std_val=stop_criterion)


    namefile = path_result_img + str(d0.month) + str(d0.day) + '_' + \
               str(d0.hour) + str(d0.minute) + '_merged.jpg'
    show.show_superpixels(img_filt, nsegments, namefile)

    # LOADING TRAINING SET
    # --------------------
    feat1 = np.load('../data_images/data_training/train_features1.npy')
    class_lbl1 = np.load('../data_images/data_training/class_labels1.npy')
    feat2 = np.load('../data_images/data_training/train_features2.npy')
    class_lbl2 = np.load('../data_images/data_training/class_labels2.npy')
    feat4 = np.load('../data_images/data_training/train_features4.npy')
    class_lbl4 = np.load('../data_images/data_training/class_labels4.npy')

    feat = np.vstack((np.vstack((feat1,feat2)), feat4))
    class_lbl = np.hstack((np.hstack((class_lbl1,class_lbl2)), class_lbl4))

    # feat = np.vstack((feat2,feat1))
    # class_lbl = np.hstack((class_lbl2,class_lbl1))
    # feat = np.vstack((feat4,feat1))
    # class_lbl = np.hstack((class_lbl4,class_lbl1))
    # feat = np.vstack((feat4,feat2))
    # class_lbl = np.hstack((class_lbl4,class_lbl2))

    # Chose equal number of elements per class
    # data_training, lbl_training = equi_number_samples(feat, class_lbl)
    data_training = feat
    lbl_training = class_lbl

    clf = prepare_classifier(data_training, lbl_training, \
                             normalize_method, classification_method)



    # CLASSIFY NEW DATA
    # --------------------------
    list_features_learning = ['Lab', 'laplacian', 'frangi']

    if not classification_with_corrections:

        if normalize_method:
            node_classifier(G, G.nodes(), dict_features, list_features_learning, nsegments, 'class', clf, normalize_method())
        else:
            node_classifier(G, G.nodes(), dict_features, list_features_learning, nsegments, 'class', clf, normalize_method)

        dic_corresp_label = {sp : np.int(np.unique(nsegments[segments == sp])) \
                         for sp in np.unique(segments)}
    else:
        min_size_region = 1
        nodes2classify = [nod for nod in G.nodes() if 'n_superpix' in G.node[nod] and G.node[nod]['n_superpix'] > min_size_region]

        if normalize_method:
            node_classifier(G, nodes2classify, dict_features, list_features_learning, nsegments, 'class', clf, normalize_method())
        else:
            node_classifier(G, nodes2classify, dict_features, list_features_learning, nsegments, 'class', clf, normalize_method)

        dic_corresp_label = {sp : np.int(np.unique(nsegments[segments == sp])) \
                             for sp in np.unique(segments)}


    # CLASSIFICATION ERRORS
    # -----------------------------
    if use_lbl_img :
        fail_rate, ratio_classified, fpr = classification_errors(G, dic_true_class, dic_corresp_label)



    # BACKGROUND SP
    # --------------
    min_size_region = 3
    bgclass = 0

    bg_nodes_class = [tn for tn in G.nodes() if 'class' in G.node[tn] and G.node[tn]['class'] == bgclass]
    bg_nodes_nsp = [tn for tn in bg_nodes_class if 'n_superpix' in G.node[tn]
                    and G.node[tn]['n_superpix'] > min_size_region]

    # Find parcels and export polygons in geoJSON format
    # --------------------------------------------------
    ksize_flooding = 2
    listFeatPolygon, dic_polygon = find_parcels(bg_nodes_nsp, nsegments, dict_features, ksize_flooding)
    # Export geoJSON
    filename_geoJson = '../data_images/export/' + cad + '_final_polygons.geojson'
    savePolygons(listFeatPolygon, filename_geoJson)

    # Show ridge image for flooding
    ridge2flood, tmp = compute_images_for_flooding(dict_features['frangi'], ksize_flooding)
    filename_ridge = path_result_img + str(d0.month) + str(d0.day) + '_' +  \
                    str(d0.hour) + str(d0.minute) + '_ridges.jpg'
    cv2.imwrite(filename_ridge, ridge2flood)

    # Show Polygons
    filename_polygons = path_result_img + str(d0.month) + str(d0.day) + '_' +  \
                    str(d0.hour) + str(d0.minute) + '_poly.jpg'
    show.show_polygons(img_filt, dic_polygon, (0,255,0), filename_polygons)

    # Give one unique ID for each polygon
    # ----------------------------------------------
    # Image with labeled polygons
    polygons_labels = np.zeros(img_filt.shape[:2])
    for parcels in dic_polygon.keys():
        for poly in dic_polygon[parcels]:
            mask_polygon_to_fill = np.zeros(polygons_labels.shape, 'uint8')
            cv2.fillPoly(mask_polygon_to_fill, [poly], 255)
            polygons_labels[mask_polygon_to_fill > 0] = parcels
            # Crop polygon image and save it
            cropped_polygon_image = crop_polygon(img_filt, [poly])
            filename_crooped_polygon = path_cropped_polygon + cad + str(parcels) + '_' + str(poly[0][0][0]) + '.jpg'
            cv2.imwrite(filename_crooped_polygon, cropped_polygon_image)

    # Erode to avoid touching boundaries
    mask_to_erode = 1*(polygons_labels != 0)
    mask_to_erode = cv2.erode(np.uint8(mask_to_erode), np.ones((4,4),np.uint8))
    # Label each polygon with different labels
    polygons_labels = np.int0(polygons_labels*(mask_to_erode > 0))


    # Compute error on polygons
    if use_lbl_img:
        ntrue_parcels, ntotal_parcels, ntruepositive_parcels = error_polygon(polygons_labels, img_truth_polygons)


    # Get text pixels
    text = np.int0((dict_features['frangi'] > 0.4)*polygons_labels)
    mask_text_erode = np.uint8(255*(text != 0))
    # Opening
    # text = cv2.morphologyEx(np.uint8(text), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    # text = cv2.morphologyEx(text, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)))
    text = text*(mask_text_erode > 0)

    # CORRECTION OF TEXT SP USING POLYGONS INFORMATION
    # ------------------------------------------------
    # Set node attribute of text nodes
    # mask_text = text > 0
    # Get id node of text segment
    # text_node_id = np.unique(segments*(text > 0))
    text_node_id = np.unique(segments*(text != 0))
    # remove BG id
    if 0 in text_node_id:
        text_node_id = list(text_node_id)
        text_node_id.remove(0)

    # set attribute class of text node to be 1
    attr_text = {tn : 1 for tn in text_node_id}
    G.add_nodes_from([tn for tn in text_node_id if dic_corresp_label[tn] < 0]) # add only nodes that have been removed
    nx.set_node_attributes(G, 'class', attr_text)

    # Correct nodes class by removing the text node from previous eventual merged group of nodes
    node_to_correct = {tn : dic_corresp_label[tn] for tn in text_node_id if dic_corresp_label[tn] < 0}

    for nc in node_to_correct.keys():
        G.node[node_to_correct[nc]]['id_sp'].remove(nc)
        pix = segments == nc
        nsegments[pix] = nc
        assign_feature_to_node(G, nc, nsegments, dict_features)

    # Update correspondanceies label nodes
    dic_corresp_label = {sp : np.int(np.unique(nsegments[segments == sp])) \
                             for sp in np.unique(segments)}

    # For each text node, look which is its corresponding polygon and label the text with the polygon label
    # attr_lbl_poly = {tn : np.max(np.unique(polygons_labels[segments == tn])) for tn in text_node_id}
    attr_lbl_poly = {tn : most_common_label(polygons_labels[segments == tn]) for tn in text_node_id}
    nx.set_node_attributes(G, 'lbl_poly', attr_lbl_poly)


    # CLASSIFICATION OF LEFT UNCLASSIFIED NODES
    # -----------------------------------------

    if classification_with_corrections:

        # Get class attribute from every node
        attribute_node_class = {n : G.node[dic_corresp_label[n]]['class'] for n in np.unique(segments) if 'class' in G.node[dic_corresp_label[n]]}
        attribute_node_lbl_poly = {n : G.node[dic_corresp_label[n]]['lbl_poly'] for n in np.unique(segments) if 'lbl_poly' in G.node[dic_corresp_label[n]]}
        O = nx.Graph()
        # Construct vertices and neighboring edges
        vertices, edges = generate_vertices_and_edges(segments)
        # Add nodes
        O.add_nodes_from(vertices)
        nx.set_node_attributes(O, 'class', attribute_node_class)
        nx.set_node_attributes(O, 'lbl_poly', attribute_node_lbl_poly)
        O.add_edges_from(edges)

        # Classification
        params = {'normalize_method' : normalize_method, 'clf' : clf, 'feat_learning' : list_features_learning, \
                  'corresp_label' : dic_corresp_label, 'similarity_method' : isol_sim_method}
        O2 = classification_with_neighbors(O, G, segments, dict_features, params)
        classification_with_min_std(O2, O, G, segments, dict_features, isol_sim_method)

        unclassnodes = [nod for nod in G.nodes() if 'class' not in G.node[nod]]

        if normalize_method:
            node_classifier(G, unclassnodes, dict_features, list_features_learning, nsegments, 'class', clf, normalize_method())
        else:
            node_classifier(G, unclassnodes, dict_features, list_features_learning, nsegments, 'class', clf, normalize_method)


        # -----------------------------
        dic_corresp_label = {sp : np.int(np.unique(nsegments[segments == sp])) \
                             for sp in np.unique(segments)}

    else:
        # Get class attribute from every node
        attribute_node_class = {n : G.node[dic_corresp_label[n]]['class'] for n in np.unique(segments) if 'class' in G.node[dic_corresp_label[n]]}
        attribute_node_lbl_poly = {n : G.node[dic_corresp_label[n]]['lbl_poly'] for n in np.unique(segments) if 'lbl_poly' in G.node[dic_corresp_label[n]]}

        O = nx.Graph()

        # Construct vertices and neighboring edges
        vertices, edges = generate_vertices_and_edges(segments)
        # Add nodes
        O.add_nodes_from(vertices)
        nx.set_node_attributes(O, 'class', attribute_node_class)
        nx.set_node_attributes(O, 'lbl_poly', attribute_node_lbl_poly)
        O.add_edges_from(edges)


    # CLASSIFICATION ERRORS
    # -----------------------------
    if use_lbl_img :
        fail_rate, ratio_classified, fpr = classification_errors(G, dic_true_class, dic_corresp_label)



    # SHOW CLASS
    # -----------
    namefile = path_result_img + str(d0.month) + str(d0.day) + '_' + \
                 str(d0.hour) + str(d0.minute) + '_predicted3class.jpg'
    show.show_class(segments, nsegments, G, namefile)




    # BOUNDING BOXES
    # ---------------

    # Build text graphs
    # -----------------
    Tgraph = nx.Graph()

    text_nodes = {tn : O.node[tn]['class'] for tn in O.nodes() if O.node[tn]['class'] == 1}
    attr_lbl_poly = {tn : O.node[tn]['lbl_poly'] for tn in O.nodes() if 'lbl_poly' in O.node[tn]}

    Tgraph.add_nodes_from(text_nodes.keys())
    nx.set_node_attributes(Tgraph, 'class', text_nodes)
    Tgraph.add_nodes_from(attr_lbl_poly.keys())
    nx.set_node_attributes(Tgraph, 'lbl_poly', attr_lbl_poly)

    # Add edges only between text nodes
    for tn in text_nodes:
        adjacent_nodes = [an[0] for an in O[tn].items() if not an[1]] # check only nodes that are not linked yet
        for an in adjacent_nodes:
            if O.node[tn]['class'] == O.node[an]['class']: # add edge if link between two text superpixels
                Tgraph.add_edge(tn, an)


    # Find boxes
    # ----------
    box_id = 0
    listBox = findTextBoxes(Tgraph, segments, box_id)

    # Show all bounding boxes
    # -------------------
    img_box = img_filt.copy()
    filename_box = path_result_img + str(d0.month) + str(d0.day) + '_' + \
                str(d0.hour) + str(d0.minute) + '_box_img.jpg'
    show.show_boxes(img_box, listBox, (255,0,0), filename_box)



    boxes_for_stats = copy.deepcopy(listBox)

    # TRY TO ELIMINATE FALSE POSITIVES
    # --------------------------------
    boxes_false = find_false_box(listBox, boxes_for_stats)
    boxes_with_lbl = [b for b in listBox if b.lbl_polygon is not None]
    true_non_labeled = [b for b in listBox if b not in boxes_false \
                  and b not in boxes_with_lbl]

    # GROUP BOXES That have labels
    # ----------------------------
    maximum_distance = 15
    groupedBox = group_box_with_lbl(boxes_with_lbl, boxes_false, maximum_distance, box_id)

    # Maybe check also too big boxes that may appear with groupbox
    # and add it to the previous list
    lists.add_list_to_list(boxes_false, find_false_box(boxes_with_lbl, boxes_for_stats))
    # Consider only unique elements
    boxes_false = lists.remove_duplicates(boxes_false)

    # True boxes
    boxes_true = true_non_labeled.copy()
    lists.add_list_to_list(boxes_true, [b for b in boxes_with_lbl if b not in boxes_false])
    # Flatten list and consider only unique elements
    boxes_true = lists.remove_duplicates(boxes_true)

    # GROUP BOXES taht are true with small false elements
    # ----------------------------
    final_boxes = boxes_true.copy()
    maximum_distance = 7
    groupedBox = group_box_with_isolates(final_boxes, boxes_false, maximum_distance, box_id)

    # Check for false box one last time
    lists.add_list_to_list(boxes_false, find_false_box(final_boxes, boxes_for_stats))
    boxes_false = lists.remove_duplicates(boxes_false)

    final_boxes = [b for b in final_boxes if b not in boxes_false]

    # Compute error of detection
    if use_lbl_img:
        ntrue_digits, ntotal_digits, ntruepositive_digits = error_box(final_boxes, img_truth_digits)



    # SHOW BOXES
    # ----------
    # All
    img_allBox = img_filt.copy()
    show.show_boxes(img_allBox, final_boxes, (0,255,0))
    show.show_boxes(img_allBox, boxes_false, (0,0,255))
    cv2.imwrite(path_result_img + str(d0.month) + str(d0.day) + '_' + \
                str(d0.hour) + str(d0.minute) + '_allBox.jpg', img_allBox)

    # Final boxes
    filename_finalBox = path_result_img + str(d0.month) + str(d0.day) + '_' + \
                        str(d0.hour) + str(d0.minute) + '_finalBox.jpg'
    show.show_boxes(img_filt.copy(), final_boxes, (0,255,0), filename_finalBox)
    # # False boxes
    # filename_falseBox = path_result_img + str(d0.month) + str(d0.day) + '_' + \
    #                     str(d0.hour) + str(d0.minute) + '_falseBox.jpg'
    # show.show_boxes(img_filt.copy(), boxes_false, (0,0,255), filename_falseBox)





    # PROCESS BOX (ROTATE, ...) AND SAVE IT

    # Create directory to save digits image
    path_digits = '../data_images/digits/' + cad + str(d0.month) + str(d0.day) + '_' + \
                    str(d0.hour) + str(d0.minute) + '/'
    if not os.path.exists(path_digits):
        os.makedirs(path_digits)

    for box in final_boxes:
        # Expand box
        box.expand_box(padding=2)

        # Crop
        crop_imgL, (xmin,xmax,ymin,ymax) = crop_box(box, dict_features['Lab'][:,:,0])
        cropped_number = img[ymin:ymax+1, xmin:xmax+1].copy()

        # Fill pixels outside the box
        crop_imgL, maxVal = fill_outside_box(crop_imgL, box)

        # Binarization
    #     maxVal = np.max(crop_img)
        blur = cv2.GaussianBlur(crop_imgL, (3, 3), 0)
        ret, binary_crop = cv2.threshold(blur, 0, maxVal, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        inv_crop = np.uint8(255*(binary_crop < 1))

        # Skeletonization
        skel = skeletonize(inv_crop)

        # Morphology
        # dilated
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(inv_crop, kernel)

        # Find orientation with PCA
        center, eigvect, angle = find_orientation(dilated)
        # Plot orientation
    #     img2draw = inv_crop.copy()
    #     diagonal = eigvect[0]
    #     linex = int(center[0] + diagonal[0]*20)
    #     liney = int(center[0] + diagonal[1]*20)
    #     cv2.circle(img2draw, (int(center[0]), int(center[1])), 1, 128, -1)
    #     cv2.line(img2draw, (int(center[0]), int(center[1])), (linex, liney), 128, 1)

        # Rotate image to align it horizontaly
        img_pad = padding(inv_crop, 0)
        rotated_img = rotate_image(img_pad, angle)

        # Recrop image
        final_digit_bin, (y,x,h,w) = crop_object(rotated_img)
        final_digit_bin = final_digit_bin > 1

        # Format image to be saved into 3 channel uint8
        final_digit_bin = np.uint8(255*final_digit_bin)
        formated_digit_img = np.dstack([final_digit_bin]*3)

        # Rotate original image
        rotated_number = rotate_image(padding(cropped_number, 255), angle)
        rotated_number = rotated_number[x:x+w, y:y+h, :]

        # Save croped image
        if box.lbl_polygon:
            cv2.imwrite(path_digits + str(box.lbl_polygon) + '_' + str(box.box_id) + '.jpg', formated_digit_img)
            cv2.imwrite(path_digits + str(box.lbl_polygon) + '_' + str(box.box_id) + '_orginal' + '.jpg', rotated_number)
        else:
            cv2.imwrite(path_digits + str(box.box_id) + '.jpg', formated_digit_img)
            cv2.imwrite(path_digits + str(box.box_id) + '_orginal' + '.jpg', rotated_number)


        # RECOGNIZING NUMBERS
        # --------------------
        # Number of digits per number  <<<<<< find a way to estimate before recognition
        final_skelton = skeletonize(final_digit_bin)
        projx = np.sum(final_skelton > 0, axis=0)
        # More than 2 pixels per colums non zero, digits is composed of at least 4 columns
        number_of_digits = find_pattern(projx > 2, [True]*4)
        number_of_digits = 4
        prediction, proba = recognize_number(rotated_number, number_of_digits=number_of_digits)

        # Save in JSON file
        data = {'number': prediction, 'confidence': proba}
        if box.lbl_polygon:
            filename_json = path_digits + str(box.lbl_polygon) + '_' + str(box.box_id) + '_json.txt'
        else:
            filename_json = path_digits + str(box.box_id) + '_json.txt'
        with open(filename_json, 'w') as fp:
            json.dump(data, fp)




    elapsed_time = time.time() - t0
    m, sec = divmod(np.float32(elapsed_time), 60)
    h, m = divmod(m, 60)


  # SAVING INFO IN FILE
    name_logfile = str(d0.month) + str(d0.day) + '_' + str(d0.hour) + str(d0.minute) + str(d0.second)
    # Open file (or create it)
    log_file = open(path_logfile + name_logfile + '.txt', 'w+')

    log_file.write('* Image  : filename : %s' % filename_img)
    log_file.write(', size : %s\n' % str(np.product(img_filt.shape[:2])))
    log_file.write('* Filters : %s\n ' % str(filters))
    log_file.write(' ---------------- \n')
    log_file.write('* Superpixels : %s\n' % str((params_slic)))
    log_file.write(' ---------------- \n')
    log_file.write('* Features : %s\n' % str((list_dict_features)))
    log_file.write(' ---------------- \n')
    log_file.write('* Merging method : %s\n' % merging_method)
    log_file.write('* Similarity method : %s\n' % similarity_method)
    log_file.write('* Stop criterion : %s\n' % stop_criterion)
    log_file.write(' ---------------- \n')
    # log_file.write('* Learning features : %s\n' % str((list_features_learning)))
    # log_file.write('* Normalization method : %s\n' % normalize_method)
    # log_file.write('* Classification method : %s\n' % classification_method)
    log_file.write(' ---------------- \n')
    log_file.write('* Classification with correction : %s\n' % classification_with_corrections)
    log_file.write(' ---------------- \n')
    if use_lbl_img:
        log_file.write('* Failing : total : %s' % str(fail_rate[3]))
        log_file.write(', text : %s' % str(fail_rate[1]))
        log_file.write(', contours : %s' % str(fail_rate[2]))
        log_file.write(', background : %s\n' % str(fail_rate[0]))
        log_file.write(' ---------------- \n')
        log_file.write('* False positive rate : total : %s' % str(fpr[3]))
        log_file.write(', text : %s' % str(fpr[1]))
        log_file.write(', contours : %s' % str(fpr[2]))
        log_file.write(', background : %s\n' % str(fpr[0]))
        log_file.write('* Parcels : %02d / %02d found (%02f)'  % (ntruepositive_parcels, \
                    ntrue_parcels, ntruepositive_parcels/ntrue_parcels))
        log_file.write(' , false positive: %02d / %02d (%02f) \n' % (ntotal_parcels - ntruepositive_parcels, \
                        ntotal_parcels, (ntotal_parcels - ntruepositive_parcels)/ntotal_parcels))
        log_file.write('* Digits : %02d / %02d found (%02f)'  % (ntruepositive_digits, \
                       ntrue_digits, ntruepositive_digits/ntrue_digits))
        log_file.write(' , false positive: %02d / %02d (%02f) \n' % (ntotal_digits - ntruepositive_digits, \
                        ntotal_digits, (ntotal_digits - ntruepositive_digits)/ntotal_digits))
        log_file.write('* Time elapsed : %d:%02d:%02d\n' % (h, m, sec))
    log_file.write('-------')
    log_file.close()

    if use_lbl_img:
        return { 'classification' : [fail_rate, fpr], 'parcels' : [ntruepositive_parcels, ntrue_parcels, ntotal_parcels - ntruepositive_parcels, ntotal_parcels], \
                'digits' : [ntruepositive_digits, ntrue_digits, ntotal_digits - ntruepositive_digits, ntotal_digits] }
    else:
        return