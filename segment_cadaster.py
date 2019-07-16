#!/usr/bin/env python
__author__ = 'solivr'

import cv2, sys, os, pickle, copy, json, time, argparse
from osgeo import gdal
import numpy as np
import networkx as nx
import tensorflow as tf
from collections import OrderedDict
from preprocessing import image_filtering, features_extraction
from segmentation import compute_slic
import helpers, polygons, graph
from helpers import Params
from classification import node_classifier
import text as txt
try:
    import better_exceptions
except ImportError:
    pass


def segment_cadaster(filename_cadaster_img: str, tf_model_dir: str, params: Params) -> None:
    """
    Launches the segmentation of the cadaster image and outputs

    :param filename_cadaster_img: cadaster image to be processed
    :param tf_model_dir : Path of tensorflow model to be used for digit recognition.
    :param params : All the parameters needed (see helpers.config.Params)
    """

    # Time process
    t0 = time.time()

    # Load cadaster image
    img = cv2.imread(filename_cadaster_img)

    try:
        img.shape
    except AttributeError:
        sys.exit("Image not loaded correctly or not found")

    #
    # FILTERING
    #
    print('-- FILTERING --')
    img_filt = image_filtering(img)

    #
    # SLIC
    #
    print('-- SLIC --')
    original_segments = compute_slic(helpers.bgr2rgb(img_filt), parameters)

    #
    # FEATURE EXTRACTION
    #
    print('-- FEATURE EXTRACTION --')
    try:

        with open(params.saving_filename_feats, 'rb') as handle:
            dict_features = pickle.load(handle)
        print('\t Debug Mode : {} file loaded'.format(os.path.split(params.saving_filename_feats)[-1]))
    except FileNotFoundError:
        dict_features = features_extraction(img_filt, params.list_features)

        if params.debug_flag:
            with open(params.saving_filename_feats, 'wb') as handle:
                pickle.dump(dict_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t Debug Mode : {} file saved'.format(os.path.split(params.saving_filename_feats)[-1]))

    #
    # GRAPHS AND MERGING
    #
    print('-- GRAPHS AND MERGING --')
    try:
        with open(params.saving_filename_graph, 'rb') as handle:
            G = pickle.load(handle)
        with open(params.saving_filename_nsegments, 'rb') as handle:
            nsegments = pickle.load(handle)
        print('\t Debug Mode : {} and {} files loaded'.format(os.path.split(params.saving_filename_graph)[-1],
                                                              os.path.split(params.saving_filename_nsegments)[-1]))
    except FileNotFoundError:
        # Create G graph
        G = nx.Graph()
        nsegments = original_segments.copy()

        G = graph.edge_cut_minimize_std(G, nsegments, dict_features, params.merging_similarity_method,
                                        mst=True, stop_std_val=params.merging_stop_criterion)

        if params.debug_flag:
            with open(params.saving_filename_graph, 'wb') as handle:
                pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(params.saving_filename_nsegments, 'wb') as handle:
                pickle.dump(nsegments, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t Debug Mode : {} and {} files saved'.format(os.path.split(params.saving_filename_graph)[-1],
                                                                 os.path.split(params.saving_filename_nsegments)[-1]))

    # Keep track of correspondences between 'original superpiels' and merged ones
    dic_corresp_label = {sp: np.int(np.unique(nsegments[original_segments == sp]))
                         for sp in np.unique(original_segments)}

    if params.show_plots:
        helpers.show_superpixels(img_filt, nsegments, params.plots_filename_superpixels)

    #
    # CLASSIFICATION
    #
    print('-- CLASSIFICATION --')
    # Labels : 0 = Background, 1 = Text, 2 = Contours

    # Loading fitted classifier
    with open(params.classifier_filename, 'rb') as handle:
        classifications_infos = pickle.load(handle)

    clf_params = classifications_infos['parameters']
    clf = classifications_infos['classifier']

    # Classify new data
    list_features_learning = clf_params['features']
    node_classifier(G, dict_features, list_features_learning, nsegments,
                    clf, normalize_method=clf_params['normalize_method'])

    #
    # PARCELS AND POLYGONS
    #
    print('-- PARCELS AND POLYGONS --')

    min_size_region = 3  # Regions should be formed at least of min_size_region merged original superpixels
    # bgclass = 0  # Label of 'background' class

    # Find nodes that are classified as background class and that are bigger than min_size_region
    bg_nodes = [tn for tn in G.nodes() if 'class' in G.node[tn] and G.node[tn]['class'] == params.label_background_class]
    background_class_nodes = [tn for tn in bg_nodes if 'n_superpix' in G.node[tn]
                              and G.node[tn]['n_superpix'] > min_size_region]

    # Find parcels and export polygons in geoJSON format
    ksize_flooding = 2

    try:
        with open(params.saving_filename_listpoly, 'rb') as handle:
            listFeatPolygon = pickle.load(handle)
        with open(params.saving_filename_dicpoly, 'rb') as handle:
            dic_polygon = pickle.load(handle)
        print('\t Debug Mode : {} and {} files loaded'.format(os.path.split(params.saving_filename_listpoly)[-1],
                                                              os.path.split(params.saving_filename_dicpoly)[-1]))

    except FileNotFoundError:
        # Get geometric transform
        ds = gdal.Open(filename_cadaster_img)
        geo_transform = ds.GetGeoTransform()

        listFeatPolygon, dic_polygon = \
            polygons.find_parcels(background_class_nodes, nsegments, dict_features['frangi'],
                                  ksize_flooding, geo_transform=geo_transform,
                                  approximation_epsilon=params.polygon_approx_epsilon)
        if params.debug_flag:
            with open(params.saving_filename_listpoly, 'wb') as handle:
                pickle.dump(listFeatPolygon, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(params.saving_filename_dicpoly, 'wb') as handle:
                pickle.dump(dic_polygon, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t Debug Mode : {} and {} files saved'.format(os.path.split(params.saving_filename_listpoly)[-1],
                                                                 os.path.split(params.saving_filename_dicpoly)[-1]))

    #
    # >>>>>>> HERE POLYGONS SHOULD BE SORTED BY AREA (BIGGER FIRST)
    #       so that when they are exported to VTM-Canvas, if a small parcel is situated within a bigger parcel,
    #       the small one is on top and is accessible by clicking

    if params.evaluate:
        # Evaluate
        results_evaluation_parcels = \
            polygons.global_evaluation_parcels(dic_polygon, params.groundtruth_parcels_filename,
                                               iou_thresh_parcels=params.iou_threshold_parcels)

    #
    # Export geoJSON
    polygons.savePolygons(listFeatPolygon, params.filename_geojson, filename_cadaster_img)

    if params.show_plots:
        # Show ridge image for flooding
        ridge2flood = polygons.clean_image_ridge(dict_features['frangi'], ksize_flooding)
        cv2.imwrite(params.plots_filename_ridge, ridge2flood)
        helpers.show_polygons(img_filt, dic_polygon, color=(6, 6, 133), filename=params.plots_filename_polygons)

    # Give one unique ID for each polygon, crop each one and save it independently
    # Image with labeled polygons
    polygons_labels = np.zeros(img_filt.shape[:2])

    for bg_nodes, list_tuple in dic_polygon.items():
        for uid, poly in list_tuple:
            mask_polygon_to_fill = np.zeros(polygons_labels.shape, 'uint8')
            cv2.fillPoly(mask_polygon_to_fill, poly, 255)
            polygons_labels[mask_polygon_to_fill > 0] = bg_nodes
            # >>>>>>>> HERE USE UUID AS LABEL
            # polygons_labels[mask_polygon_to_fill > 0] = uid.int # then do uuid.UUID(int=label) to find uuid

            if params.show_plots:
                # Crop polygon image and save it
                cropped_polygon_image = polygons.crop_polygon(img_filt, poly)
                filename_cropped_polygon = os.path.join(params.dir_cropped_polygons, '{}.jpg'.format(uid))
                cv2.imwrite(filename_cropped_polygon, cropped_polygon_image)

    #
    # TEXT PIXELS
    #
    print('__TEXT PIXELS__')

    # Saving for debug
    try:

        with open(params.saving_filename_boxes, 'rb') as handle:
            final_boxes = pickle.load(handle)
        print('\t Debug Mode : {} file loaded'.format(os.path.split(params.saving_filename_boxes)[-1]))

    except FileNotFoundError:
        # Erode binary version of polygons to avoid touching boundaries to have a 'map'
        # of polygons defined by their label. This will be useful to group text elements.
        mask_to_erode = cv2.erode(np.uint8(1 * (polygons_labels != 0)), np.ones((4, 4), np.uint8))
        polygons_labels *= (mask_to_erode > 0)

        # This should give text candidates since text is usually inside the polygons
        text = (dict_features['frangi'] > 0.4) * polygons_labels
        mask_text_erode = np.uint8(255 * (text != 0))
        # Opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_CLOSE, kernel)
        mask_text_erode = cv2.morphologyEx(mask_text_erode, cv2.MORPH_OPEN, kernel)
        text *= (mask_text_erode > 0)

        #
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
            graph.assign_feature_to_node(G, k, nsegments, dict_features)

        # Update correspondancies label nodes
        dic_corresp_label = {sp: np.int(np.unique(nsegments[original_segments == sp]))
                             for sp in np.unique(original_segments)}

        # For each text node, look which is its corresponding polygon and label the text with the polygon label
        # For that we look at which label is the most present (in the case of multiple labels appearing)
        attr_lbl_poly = {tn: helpers.most_common_label(polygons_labels[original_segments == tn]) for tn in text_node_id}
        nx.set_node_attributes(G, 'lbl_poly', attr_lbl_poly)

        #
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
        vertices, edges = graph.generate_vertices_and_edges(original_segments)
        # Add nodes
        O.add_nodes_from(vertices)
        nx.set_node_attributes(O, 'class', attribute_node_class)
        nx.set_node_attributes(O, 'lbl_poly', attribute_node_lbl_poly)
        O.add_edges_from(edges)

        # Show class
        if params.show_plots:
            helpers.show_class(nsegments, G, params.plots_filename_class)

        #
        #
        # BOUNDING BOXES
        #
        print('__BOUNDING BOXES__')

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
        # box_id = 0
        listBox = txt.find_text_boxes(Tgraph, original_segments)

        reference_boxes = copy.deepcopy(listBox)
        # TRY TO ELIMINATE FALSE POSITIVES
        # --------------------------------
        boxes_false = txt.find_false_box(listBox, reference_boxes)
        boxes_with_lbl = [b for b in listBox if b.lbl_polygon is not None]
        true_non_labeled = [b for b in listBox if b not in boxes_false and b not in boxes_with_lbl]

        # GROUP BOXES that have labels
        # ----------------------------
        maximum_distance = 15
        groupedBox = txt.group_box_with_lbl(boxes_with_lbl, boxes_false, maximum_distance)

        # Maybe check also too big boxes that may appear with groupbox
        # and add it to the previous list
        helpers.add_list_to_list(boxes_false, txt.find_false_box(boxes_with_lbl, reference_boxes))
        # Consider only unique elements
        boxes_false = helpers.remove_duplicates(boxes_false)

        # True boxes
        boxes_true = true_non_labeled.copy()
        helpers.add_list_to_list(boxes_true, [b for b in boxes_with_lbl if b not in boxes_false])
        # Flatten list and consider only unique elements
        boxes_true = helpers.remove_duplicates(boxes_true)

        # GROUP BOXES that are true with small false elements
        # ----------------------------
        final_boxes = boxes_true.copy()
        maximum_distance = 7
        groupedBox = txt.group_box_with_isolates(final_boxes, boxes_false, maximum_distance)

        # Check for false box one last time
        helpers.add_list_to_list(boxes_false, txt.find_false_box(final_boxes, reference_boxes))
        boxes_false = helpers.remove_duplicates(boxes_false)

        final_boxes = [b for b in final_boxes if b not in boxes_false]

        # SHOW BOXES
        if params.show_plots:
            # All
            img_allBox = img_filt.copy()
            helpers.show_boxes(img_allBox, final_boxes, (0, 255, 0))
            helpers.show_boxes(img_allBox, boxes_false, (0, 0, 255))
            cv2.imwrite(params.plots_filename_allbox, img_allBox)

            # Final boxes
            helpers.show_boxes(img_filt.copy(), final_boxes, (0, 255, 0), params.plots_filename_finalbox)

        #
        #
        # PROCESS BOX (ROTATE, ...), PREDICT NUMBER AND SAVE IT
        #
        print('__PROCESS BOX (ID RECOGNITION)__')
        # Restore model and then predict one by one the images -> not optimal (batch instead)!!
        sess = tf.Session()

        loaded_model = tf.saved_model.loader.load(sess, ['serve'], tf_model_dir)
        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def['predictions'])

        for box in final_boxes:
            # Find orientation of the text (center, eigenvector, angle)
            _, _, angle = txt.find_text_orientation_from_box(box, img_filt)

            # Get original image with margin and rotate it
            bounding_rect_coords = txt.custom_bounding_rect(box.original_box_pts)
            bounding_rect_coords = txt.add_margin_to_rectangle(bounding_rect_coords, margin=3)
            bounding_rect_coords = txt.check_validity_points(bounding_rect_coords, img_filt.shape)
            x, y, w, h = cv2.boundingRect(bounding_rect_coords)
            crop_img = img_filt[y:y+h, x:x+w].copy()
            rotated_crop, rot_mat = helpers.rotate_image_with_mat(crop_img.copy(), angle)

            # Get the box points with the new rotated coordinates
            box_pts_offset_crop = box.original_box_pts.copy()
            box_pts_offset_crop[:, 0] = box.original_box_pts[:, 0] - x
            box_pts_offset_crop[:, 1] = box.original_box_pts[:, 1] - y
            rotated_coords = cv2.transform(np.array([box_pts_offset_crop]), rot_mat)
            rotated_coords = txt.check_validity_points(rotated_coords, rotated_crop.shape)
            # Crop rotated element
            x_rot, y_rot, w_rot, h_rot = cv2.boundingRect(rotated_coords[0])
            crop_number = rotated_crop[y_rot:y_rot+h_rot, x_rot:x_rot+w_rot]

            if crop_number.size == 0:
                ind = final_boxes.index(box)
                final_boxes[ind] = []
                # final_boxes.remove(box)
                continue
            ratio_size = crop_number.shape[1]/crop_number.shape[0]
            if ratio_size < 0.8:
                ind = final_boxes.index(box)
                final_boxes[ind] = []
                # final_boxes.remove(box)
                continue

            # RECOGNIZING NUMBERS
            # --------------------
            # Format cropped number for crnn network
            crop_number_formatted = cv2.cvtColor(crop_number, cv2.COLOR_BGR2GRAY)  # grayscale
            crop_number_formatted = crop_number_formatted[:, :, None]

            # Get 'words' and 'difference_logprob'
            predictions = sess.run(output_dict, feed_dict={input_dict['images']: crop_number_formatted})

            number_predicted = predictions['words'][0].decode('utf8')
            confidence = predictions['score'][0]

            try:
                box.prediction_number = tuple([number_predicted,
                                               float('{:.02f}'.format(confidence))])
            except TypeError:  # Delete box
                ind = final_boxes.index(box)
                final_boxes[ind] = []
                # final_boxes.remove(box)
                continue

            # Save in JSON file
            data = OrderedDict([('number', number_predicted), ('confidence', str(confidence))])
            filename_json = os.path.join(params.dir_digits, '{}_{}_json.txt'.format(box.prediction_number, box.box_id))
            with open(filename_json, 'w') as fjson:
                json.dump(data, fjson)

            # Save cropped image
            if params.show_plots:
                cv2.imwrite(os.path.join(params.dir_digits, '{}_{}_original.jpg'.format(box.prediction_number, box.box_id)),
                            crop_number)

        # Don't forget to close session
        sess.close()

        # Remove empty items from list
        final_boxes = [b for b in final_boxes if b]

        if params.debug_flag:
            with open(params.saving_filename_boxes, 'wb') as handle:
                pickle.dump(final_boxes, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('\t Debug Mode : {} file saved'.format(os.path.split(params.saving_filename_boxes)[-1]))

    #
    # Evaluation of predicted digits
    if params.evaluate:
        # -- IOU evaluation
        print('-- Evaluation IoU ({})--'.format(params.iou_threshold_digits))
        results_localization_iou, results_recognition_iou, \
        boxes_predict_list_correct_located_iou, boxes_predict_list_incorrect_located_iou \
            = txt.global_digit_evaluation(final_boxes, params.groundtruth_labels_digits_filename,
                                          thresh=params.iou_threshold_digits, use_iou=True, printing=True)
        # -- inter evaluation
        print('-- Evaluation INTER -- ({})'.format(params.inter_threshold_digits))
        results_localization_inter, results_recognition_inter, \
            boxes_predict_list_correct_located_inter, boxes_predict_list_incorrect_located_inter \
            = txt.global_digit_evaluation(final_boxes, params.groundtruth_labels_digits_filename,
                                          thresh=params.inter_threshold_digits, use_iou=False, printing=True)

        results_digits_eval = {'iou': (results_localization_iou, results_recognition_iou),
                               'inter': (results_localization_inter, results_recognition_inter)}
    #
    # LOG FILE
    # ---------
    print('-- LOG FILE --')
    # Write Log file
    elapsed_time = time.time() - t0

    if params.evaluate:
        helpers.write_log_file(params, elapsed_time=elapsed_time,
                               results_parcels_eval=results_evaluation_parcels,
                               results_digits_eval=results_digits_eval)
    else:
        helpers.write_log_file(params, elapsed_time=elapsed_time)

    print('Cadaster image processed with success!')
# ----------------------------------------------------------------------------------------


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}


if __name__ == '__main__':
    # Parsing
    parser = argparse.ArgumentParser(description='Cadaster segmentation process')
    parser.add_argument('-im', '--cadaster_img', help="Filename of the cadaster image", type=str)
    parser.add_argument('-out', '--output_path', help='Output directory for results and plots. Default : outputs',
                        type=str, default='outputs')
    parser.add_argument('-c', '--classifier', type=str, help='Filename of fitted classifier. '
                                                             'Default : data/svm_classifier.pkl',
                        default='data/svm_classifier.pkl')
    parser.add_argument('-tf', '--tensorflow_model', help='Path of the tensorflow model for digit recognition',
                        default='data/models/crnn_numbers')
    parser.add_argument('-p', '--plot', type=bool, help='Show plots (boolean). Default : 1', default=True)
    parser.add_argument('-ev', '--evaluation', type=bool, help='To enable evaluation of parcels extraction and digit '
                                                               'segmentation (only possible if a ground-truth is '
                                                               'available in folder data/data_evaluation). Default : 0',
                        default=False)
    parser.add_argument('-d', '--debug', type=bool, help='Debug flag. 1 to activate. Default : 0', default=False)
    parser.add_argument('-g', '--gpu', type=str, help='GPU device, ('' For CPU)', default='')

    args = vars(parser.parse_args())

    parameters = Params(input_filenames=args.get('cadaster_img'),
                        output_dir=args.get('output_path'),
                        tf_model_dir=args.get('tensorflow_model'),
                        show_plots=args.get('plot'),
                        evaluate=args.get('evaluation'),
                        debug=args.get('debug'),
                        gpu=args.get('gpu'),
                        classifier_filename='data/svm_classifier.pkl',
                        list_features=['Lab', 'laplacian', 'frangi', 'RGB'],
                        iou_threshold_parcels=0.7,
                        iou_threshold_digits=0.5,
                        inter_threshold_digits=0.8)

    # Session config for tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Launch segmentation
    segment_cadaster(parameters.input_filenames, parameters.tf_model_dir, parameters)
