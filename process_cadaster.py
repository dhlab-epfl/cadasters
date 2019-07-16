#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from skimage.morphology import h_minima, watershed, label
from tqdm import tqdm
import pickle
from src.utils import export_geojson
from src.evaluation import get_labelled_parcels_matrix, get_labelled_digits_matrix, evaluate, evaluation_json_file
from src.process import process_watershed_parcel

try:
    import better_exceptions
except ImportError:
    pass

from dh_segment.inference import loader
from tf_crnn.loader import PredictionModel


def process_cadaster(filename_img: str,
                     segmentation_model_dir: str,
                     transcription_model_dir: str,
                     output_dir: str,
                     plot=False,
                     evaluation=False):

    if plot:
        plotting_dir = os.path.join(output_dir, 'plots_{}'.format(os.path.split(filename_img)[1].split('.')[0]))
        os.makedirs(plotting_dir, exist_ok=True)
    if evaluation:
        dirname, filename = os.path.split(filename_img)
        filename = filename.split('.')[0]
        parcels_groundtruth_filename = os.path.join(dirname, '{}_parcels_gt.jpg'.format(filename))
        parcel_groundtruth_matrix = get_labelled_parcels_matrix(parcels_groundtruth_filename)
        numbers_groundtruth_filename = os.path.join(dirname, '{}_digits_label_gt.png'.format(filename))
        numbers_groundtruth_matrix = get_labelled_digits_matrix(numbers_groundtruth_filename)
        pickle_filename = os.path.join(output_dir, '{}_polygons_data.pkl'.format(filename))
        log_filename = os.path.join(output_dir, '{}_evaluation_results.json'.format(filename))

    # Load cadaster image
    cadaster_original_image = imread(filename_img)
    cadaster_grayscale = cv2.cvtColor(cadaster_original_image, cv2.COLOR_RGB2GRAY)
    try:
        cadaster_original_image.shape
    except AttributeError:
        raise AttributeError("Image not loaded correctly or not found")

    #
    # SEGMENTATION
    print('-- PIXEL-WISE SEGMENTATION --')
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    tf.reset_default_graph()
    with tf.Session(config=session_config):
        segmentation_model = loader.LoadedModel(segmentation_model_dir)
        # prediction = segmentation_model.predict_with_tiles(cadaster_image[None, :, :, :])  # returns {'probs', 'labels'}
        prediction = segmentation_model.predict_with_tiles(filename_img)  # returns {'probs', 'labels'}

    # TODO : Try to use hysteresis thresholding
    contours_segmented_probs = prediction['probs'][0, :, :, 1]  # second class is contours
    text_segmented_probs = prediction['probs'][0, :, :, 0]  # first class is text

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
    # transcription_session = tf.Session(config=session_config)
    with tf.Session() as session:
        transcription_model = PredictionModel(transcription_model_dir, session)

        polygons_list = list()
        n_unique_watershed_parcels = np.unique(watershed_parcels)
        for marker_labels in tqdm(n_unique_watershed_parcels, total=len(n_unique_watershed_parcels)):

            # PARCEL EXTRACTION AND TRANSCRIPTION
            mask_parcels = watershed_parcels == marker_labels

            current_polygon = process_watershed_parcel(mask_parcels, text_segmented_probs, cadaster_grayscale,
                                                       transcription_model, plotting_dir=plotting_dir)

            polygons_list.append(current_polygon)

    # Export GEOJSON file
    export_filename = os.path.join(output_dir, 'parcels_{}.geojson'.format(os.path.split(filename)))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cadaster segmentation process')
    parser.add_argument('-im', '--cadaster_img', help="Filename of the cadaster image", type=str, nargs='+')
    parser.add_argument('-out', '--output_dir', help='Output directory for results and plots.', type=str)
    parser.add_argument('-sm', '--segmentation_tf_model', type=str,
                        help='Path of the tensorflow segmentation model for pixel-wise segmentation')
    parser.add_argument('-tm', '--transcription_tf_model', type=str,
                        help='Path of the tensorflow segmentation model for digit transcription')
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
                         segmentation_model_dir=args.get('segmentation_tf_model'),
                         transcription_model_dir=args.get('transcription_tf_model'),
                         output_dir=args.get('output_dir'),
                         plot=args.get('debug'),
                         evaluation=bool(args.get('evaluate')))
