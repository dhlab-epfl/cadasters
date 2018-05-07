***REMOVED***
***REMOVED***

import argparse
***REMOVED***
import cv2
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imsave
from skimage.morphology import h_minima, watershed, label
***REMOVED***
import pickle
import click
from src.utils import MyPolygon, crop_with_margin, find_orientation_blob, rotate_image_and_crop, get_rotation_matrix, \
    export_geojson
from src.evaluation import get_labelled_parcels_matrix, get_labelled_digits_matrix, evaluate, evaluation_json_file

try:
    import better_exceptions
except ImportError:
    pass

from dh_segment import loader
from tf_crnn.loader import PredictionModel


def process_cadaster(filename_img: str, denoising: bool, segmentation_model_dir: str,
                     transcription_model_dir: str, output_dir, gpu='1', plot=False, evaluation=False):

    if plot:
        plotting_dir = os.path.join(output_dir, 'plots_***REMOVED******REMOVED***'.format(os.path.split(filename_img)[1].split('.')[0]))
        os.makedirs(plotting_dir, exist_ok=True)
    if evaluation:
        path_eval_split = os.path.split(filename_img)
        parcels_groundtruth_filename = os.path.join(path_eval_split[0],
                                                    #'***REMOVED******REMOVED***_labelled_parcels_gt.jpg'.format(path_eval_split[1].split('.')[0]))
                                                    '***REMOVED******REMOVED***_parcels_gt.jpg'.format(path_eval_split[1].split('.')[0]))
        parcel_groundtruth_matrix = get_labelled_parcels_matrix(parcels_groundtruth_filename)
        numbers_groundtruth_filename = os.path.join(path_eval_split[0],
                                                    '***REMOVED******REMOVED***_digits_label_gt.png'.format(path_eval_split[1].split('.')[0]))
        numbers_groundtruth_matrix = get_labelled_digits_matrix(numbers_groundtruth_filename)
        pickle_filename = os.path.join(output_dir, '***REMOVED******REMOVED***_polygons_data.pkl'.format(os.path.split(filename_img)[1].split('.')[0]))
        log_filename = os.path.join(output_dir, '***REMOVED******REMOVED***_evaluation_results.json'.format(os.path.split(filename_img)[1].split('.')[0]))

    # Load cadaster image
    cadaster_original_image = imread(filename_img)
    cadaster_grayscale = cv2.cvtColor(cadaster_original_image, cv2.COLOR_RGB2GRAY)
    try:
        cadaster_original_image.shape
    except AttributeError:
        raise AttributeError("Image not loaded correctly or not found")

    #
    # FILTERING
    print('-- FILTERING --')
    if denoising:
        k = 5
        cadaster_image = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(cv2.cvtColor(cadaster_original_image,
                                                                                   cv2.COLOR_RGB2BGR),
                                                                      h=k, hColor=k), cv2.COLOR_BGR2RGB)
    else:
        cadaster_image = cadaster_original_image.copy()

    #
    # SEGMENTATION
    print('-- PIXEL-WISE SEGMENTATION --')
    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = gpu
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=session_config):
        segmentation_model = loader.LoadedModel(segmentation_model_dir)
        # prediction = segmentation_model.predict_with_tiles(cadaster_image[None, :, :, :])  # returns ***REMOVED***'probs', 'labels'***REMOVED***
        prediction = segmentation_model.predict_with_tiles(filename_img)  # returns ***REMOVED***'probs', 'labels'***REMOVED***

    # TODO : Try to use hysteresis thresholding
    contours_segmented_probs = prediction['probs'][0, :, :, 1]  # first class is contours
    text_segmented_probs = prediction['probs'][0, :, :, 0]  # second class is text

    if plot:
        imsave(os.path.join(plotting_dir, '__contours.jpg'), contours_segmented_probs)
        imsave(os.path.join(plotting_dir, '__text.jpg'), text_segmented_probs)


    #
    # WATERSHED
    print('-- WATERSHED --')
    h_level = 0.1
    minimas = label(h_minima(contours_segmented_probs, h_level))
    watershed_parcels = watershed((255 * contours_segmented_probs).astype('int'), minimas)

    # Tensorflow : loading transcription model
    tf.reset_default_graph()
    transcription_session = tf.Session(config=session_config)
    loaded_model = PredictionModel(transcription_model_dir, transcription_session)

    polygons_list = list()
    for marker_labels in tqdm(np.unique(watershed_parcels), total=len(np.unique(watershed_parcels))):

        # PARCEL EXTRACTION
        mask_parcels = watershed_parcels == marker_labels
        _, contours, _ = cv2.findContours(mask_parcels.astype('uint8').copy(), cv2.RETR_CCOMP,
                                          cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            current_polygon = MyPolygon(contours)
        else:
            continue

        # Binarize probs with Otsu's threshold
        _, binary_text_segmented = cv2.threshold(cv2.GaussianBlur((text_segmented_probs * 255).astype('uint8'),
                                                                  (3, 3), 0),
                                                 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_text_segmented = cv2.morphologyEx(binary_text_segmented, cv2.MORPH_OPEN,
                                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # LABEL EXTRACTION
        # Crop to have smaller image
        margin_text_label = 7
        text_binary_crop, coordinates_crop_parcel = crop_with_margin(binary_text_segmented,
                                                   cv2.boundingRect(current_polygon.approximate_coordinates(epsilon=1)),
                                                   margin=margin_text_label, return_coords=True)
        (x_crop_parcel, y_crop_parcel, w_crop_parcel, h_crop_parcel) = coordinates_crop_parcel
        binary_parcel_number = (255 * text_binary_crop * crop_with_margin(mask_parcels, coordinates_crop_parcel,
                                                                          margin=0)).astype('uint8')
        parcel_number = (255 * crop_with_margin(text_segmented_probs, coordinates_crop_parcel, margin=0)
                         * crop_with_margin(mask_parcels, coordinates_crop_parcel, margin=0)).astype('uint8')

        # Cleaning : Morphological opening
        binary_parcel_number = cv2.morphologyEx(binary_parcel_number, cv2.MORPH_OPEN,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Find parcel number but do not consider small elements (open)
        parcel_number_blob = cv2.dilate(binary_parcel_number, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        opening_kernel = (9, 9)
        parcel_number_blob = cv2.morphologyEx(parcel_number_blob, cv2.MORPH_OPEN,
                                              cv2.getStructuringElement(cv2.MORPH_RECT, opening_kernel))
        _, contours_blob_list, _ = cv2.findContours(parcel_number_blob.copy(), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        if contours_blob_list is None:
            continue

        if plot:
            imsave(os.path.join(plotting_dir, '***REMOVED******REMOVED***_number_binarization.jpg'.format(current_polygon.uuid)), binary_parcel_number)
            imsave(os.path.join(plotting_dir, '***REMOVED******REMOVED***_parcel_blob.jpg'.format(current_polygon.uuid)), parcel_number_blob)
            imsave(os.path.join(plotting_dir, '***REMOVED******REMOVED***_text_probs.jpg'.format(current_polygon.uuid)), parcel_number)

        number_predicted_list = list()
        scores_list = list()
        label_contour_list = list()
        for i, contour_blob in enumerate(contours_blob_list):

            if len(contour_blob) < 5:  # There should be a least 5 points to fit the ellipse
                continue

            # Compute rotation matrix and padding
            _, _, angle = find_orientation_blob(contour_blob)
            rotation_matrix, (x_pad, y_pad) = get_rotation_matrix(binary_parcel_number.shape[:2], angle - 90)

            # Crop on grayscale image
            image_parcel_number = cadaster_grayscale[y_crop_parcel:y_crop_parcel + h_crop_parcel,
                                  x_crop_parcel:x_crop_parcel + w_crop_parcel]

            image_parcel_number_rotated, rotated_contours = rotate_image_and_crop(image_parcel_number, rotation_matrix,
                                                                                  (x_pad, y_pad),
                                                                                  contour_blob[:, 0, :],
                                                                                  border_value=128)

            x_box, y_box, w_box, h_box = cv2.boundingRect(rotated_contours)
            margin_box = 0
            grayscale_number_crop = image_parcel_number_rotated[y_box + margin_box:y_box + h_box - margin_box,
                                    x_box + margin_box:x_box + w_box - margin_box]

            if grayscale_number_crop.size < 100:
                continue

            if plot:
                imsave(os.path.join(plotting_dir, '***REMOVED******REMOVED***_label_crop***REMOVED******REMOVED***.jpg'.format(current_polygon.uuid, i)),
                       image_parcel_number)
                imsave(os.path.join(plotting_dir, '***REMOVED******REMOVED***_label_rotated***REMOVED******REMOVED***.jpg'.format(current_polygon.uuid, i)),
                       grayscale_number_crop)

            # TRANSCRIPTION
            try:
                predictions = loaded_model.predict(grayscale_number_crop[:, :, None])
                number_predicted_list.append(predictions['words'][0].decode('utf8'))
                scores_list.append(predictions['score'][0])
                label_contour_list.append((contour_blob[:, 0, :] + [x_crop_parcel, y_crop_parcel])[:, None, :])
            except:
                pass

        # Add transcription and score to Polygon object
        current_polygon.assign_transcription(number_predicted_list, scores_list, label_contour_list)

        polygons_list.append(current_polygon)

    # Export GEOJSON file
    export_filename = os.path.join(output_dir, 'parcels_***REMOVED******REMOVED***.geojson'.format(os.path.split(filename_img)[1].split('.')[0]))
    export_geojson(polygons_list, export_filename, filename_img)

    # EVALUATION
    if evaluation:
        print('-- EVALUATION --')
        result_parcel_localisation, \
        result_label_localisation, \
        result_transcription, _ = evaluate(polygons_list, parcel_groundtruth_matrix, numbers_groundtruth_matrix,
                                        threshold_parcels=0.8, threshold_labels=0.8)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(polygons_list, f)

        evaluation_json_file(log_filename, results_parcels=result_parcel_localisation,
                             result_numbers=(result_label_localisation, result_transcription))

    print('Cadaster image processed!')


***REMOVED***
    parser = argparse.ArgumentParser(description='Cadaster segmentation process')
    parser.add_argument('-im', '--cadaster_img', help="Filename of the cadaster image", type=str, nargs='+')
    parser.add_argument('-out', '--output_dir', help='Output directory for results and plots.', type=str)
    parser.add_argument('-sm', '--segmentation_tf_model', type=str, help='Path of the tensorflow segmentation model '
                                                                          'for pixel-wise segmentation')
    parser.add_argument('-tm', '--transcription_tf_model', type=str, help='Path of the tensorflow segmentation model '
                                                                          'for digit transcription')
    parser.add_argument('-g', '--gpu', type=str, help='GPU device, ('' For CPU)', default='')
    parser.add_argument('-d', '--debug', type=bool, help='Plot intermediate resutls to facilitate debug', default=False)
    parser.add_argument('-ev', '--evaluate', type=bool, help='Evaluation of the results', default=False)
    args = vars(parser.parse_args())

    os.makedirs(args.get('output_dir'), exist_ok=True)

    if not isinstance(args.get('cadaster_img'), list):
        cadaster_images_filenames = [args.get('cadaster_img')]
    else:
        cadaster_images_filenames = args.get('cadaster_img')

    for cadaster_image_filename in tqdm(cadaster_images_filenames, desc='Processing_file'):
        process_cadaster(cadaster_image_filename,
                         denoising=False,
                         segmentation_model_dir=args.get('segmentation_tf_model'),
                         transcription_model_dir=args.get('transcription_tf_model'),
                         output_dir=args.get('output_dir'),
                         gpu=args.get('gpu'),
                         plot=args.get('debug'),
                         evaluation=bool(args.get('evaluate')))
