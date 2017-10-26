***REMOVED***
***REMOVED***

import cv2
***REMOVED***
import tensorflow as tf
import sys
import argparse
import numpy as np
from scipy.misc import imread
***REMOVED***
from skimage.morphology import h_minima, watershed, label
from utils import Polygon, crop_with_margin, find_orientation_blob, rotate_image_and_crop, get_rotation_matrix, \
    export_geojson
try:
    import better_exceptions
except ImportError:
    pass

sys.path.insert(0, '/home/soliveir/DocumentSegmentation/')
#sys.path.insert(0, '/Users/soliveir/Documents/DHLAB/DocumentSegmentation/')
from doc_seg import loader


def process_cadaster(filename_img: str, denoising: bool, segmentation_model_dir: str,
                     transcription_model_dir: str, output_dir, gpu='1'):

    # Load cadaster image
    cadaster_original_image = imread(filename_img)
    cadaster_cv2_image = cv2.imread(filename_img)

    try:
        cadaster_original_image.shape
    except AttributeError:
        raise AttributeError("Image not loaded correctly or not found")

    #
    # FILTERING
    print('-- FILTERING --')
    if denoising:
        k = 5
        cadaster_image = cv2.fastNlMeansDenoisingColored(cadaster_original_image, h=k, hColor=k)
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
        prediction = segmentation_model.predict(cadaster_image[None, :, :, :])  # returns ***REMOVED***'probs', 'labels'***REMOVED***
        # prediction = segmentation_model.predict_with_tiles(cadaster_image[None, :, :, :])  # returns ***REMOVED***'probs', 'labels'***REMOVED***

    contours_segmented_probs = prediction['probs'][0, :, :, 0]  # first class is contours
    text_segmented_probs = prediction['probs'][0, :, :, 1]  # second class is text

    #
    # WATERSHED
    print('-- WATERSHED --')
    h_level = 0.05
    minimas = label(h_minima(contours_segmented_probs, h_level))
    watershed_parcels = watershed((255 * contours_segmented_probs).astype('int'), minimas)

    # Tensorflow : loading transcription model
    transcription_session = tf.Session()
    tf.reset_default_graph()
    loaded_model = tf.saved_model.loader.load(transcription_session, ['serve'], transcription_model_dir)
    input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def['predictions'])

    approximation_epsilon = 1
    polygons_list = list()
    for marker_labels in tqdm(np.unique(watershed_parcels), total=len(np.unique(watershed_parcels))):
        mask_parcels = watershed_parcels == marker_labels
        _, contours, hierarchy = cv2.findContours(mask_parcels.astype('uint8').copy(), cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        # PARCEL EXTRACTION
        current_polygon = Polygon(contours[0])

        # LABEL EXTRACTION
        # Crop to have smaller image
        text_probs_crop, \
        (x, y, w, h) = crop_with_margin(text_segmented_probs,
                                        cv2.boundingRect(current_polygon.approximate_coordinates(approximation_epsilon)),
                                        margin=3, return_coords=True)
        parcel_number = (255 * text_probs_crop * crop_with_margin(mask_parcels, (x, y, w, h), margin=0)).astype('uint8')
        # parcel_number = (255 * text_segmented_probs[y:y+h, x:x+w] * mask_parcels[y:y+h, x:x+w]).astype('uint8')

        # Cleaning : Otsu's thresholding after Gaussian filtering and morphological opening
        _, binary_parcel_number = cv2.threshold(cv2.GaussianBlur(parcel_number, (3, 3), 0),
                                                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_parcel_number = cv2.morphologyEx(binary_parcel_number, cv2.MORPH_OPEN,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

        # # Inverse colormap to have white background
        # parcel_number_clean = (-(parcel_number.astype('float32')*binary_parcel_number.astype('bool')
        #                          - 255)).astype('uint8')

        # Find parcel number and rotate it
        parcel_number_blob = cv2.dilate(binary_parcel_number, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        _, contours_blob_list, hierarchy = cv2.findContours(parcel_number_blob.copy(), cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
        number_predicted_list = list()
        scores_list = list()
        for contour_blob in contours_blob_list:
            # Crop and keep trak of coordinates of blob
            margin_crop = 2
            crop_binary_parcel_number, (x, y, w, h) = crop_with_margin(binary_parcel_number,
                                                                       cv2.boundingRect(contour_blob),
                                                                       margin=margin_crop,
                                                                       return_coords=True)
            contours_blob_offset = contour_blob[:, 0, :] - [x, y]

            # Compute rotation matrix and padding
            _, _, angle = find_orientation_blob(contours_blob_offset)
            rotation_matrix, (x_pad, y_pad) = get_rotation_matrix(crop_binary_parcel_number.shape[:2], angle)

            # Rotate horizontally
            # Crop number
            parcel_number_crop = crop_with_margin(parcel_number, (x, y, w, h), margin=0)
            parcel_number_rotated, rotated_contours = rotate_image_and_crop(parcel_number_crop, rotation_matrix,
                                                                            (x_pad, y_pad),
                                                                            contours_blob_offset, border_value=0)

            # Inverse colormap to have white background
            parcel_number_clean = (-(parcel_number_rotated.astype('float32') - 255)).astype('uint8')

            # TRANSCRIPTION
            predictions = transcription_session.run(output_dict, feed_dict=***REMOVED***input_dict['images']: parcel_number_clean[:, :, None]***REMOVED***)

            number_predicted_list.append(predictions['words'][0].decode('utf8'))
            scores_list.append(predictions['score'][0])

        # Add transcription and score to Polygon object
        current_polygon.assign_transcription(number_predicted_list, scores_list)

        polygons_list.append(current_polygon)

    transcription_session.close()

    # Export GEOJSON file
    export_filename = os.path.join(output_dir, 'parcels.geojson')
    export_geojson(polygons_list, export_filename, filename_img)

    print('Cadaster image processed!')


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return ***REMOVED***k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()***REMOVED***, \
           ***REMOVED***k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()***REMOVED***


***REMOVED***
    parser = argparse.ArgumentParser(description='Cadaster segmentation process')
    parser.add_argument('-im', '--cadaster_img', help="Filename of the cadaster image", type=str)
    parser.add_argument('-out', '--output_dir', help='Output directory for results and plots.', type=str)
    parser.add_argument('-sm', '--segmentation_tf_model', type= str, help='Path of the tensorflow segmentation model '
                                                                          'for pixel-wise segmentation')
    parser.add_argument('-tm', '--transcription_tf_model', type=str, help='Path of the tensorflow segmentation model '
                                                                          'for digit transcription')
    parser.add_argument('-g', '--gpu', type=str, help='GPU device, ('' For CPU)', default='')
    args = vars(parser.parse_args())

    os.makedirs(args.get('output_dir'), exist_ok=True)

    process_cadaster(args.get('cadaster_img'),
                     denoising=False,
                     segmentation_model_dir=args.get('segmentation_tf_model'),
                     transcription_model_dir=args.get('transcription_tf_model'),
                     output_dir=args.get('output_dir'),
                     gpu=args.get('gpu'))