#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import cv2
import os
from typing import Tuple
from math import sqrt
from .utils import MyPolygon
import numpy as np
from imageio import imsave


def binarize_text_probs(text_prob_map: np.array):
    """

    :param text_prob_map:
    :return:
    """
    # Binarize probs with Otsu's threshold
    blurred = cv2.GaussianBlur((text_prob_map * 255).astype('uint8'), (3, 3), 0)
    _, binary_text_segmented = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_binary_map = cv2.morphologyEx(binary_text_segmented, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    return text_binary_map


def crop_with_margin(full_img: np.array, crop_coords: np.array, margin: int=0, return_coords: bool=False):
    """

    :param box_coords: (x, y, w, h)
    :param full_img:
    :param margin:
    :return:
    """

    x, y, w, h = crop_coords
    # Update coordinated with margin
    shape = full_img.shape
    x, y = (np.maximum(0, x - margin), np.maximum(0, y - margin))
    w, h = (np.minimum(shape[1] - (x+w + margin), 0) + w + margin, np.minimum(shape[0] - (y+h + margin), 0) + h + margin)

    if len(shape) > 2:
        crop = full_img[y:y+h+1, x:x+w+1, :]
    else:
        crop = full_img[y:y+h+1, x:x+w+1]

    if return_coords:
        return crop, (x, y, w, h)
    else:
        return crop


def get_rotation_matrix(image_shape: tuple, angle: float) -> (np.array, np.array):

    # Maximum shape after rotation is diagonal of image (Pythagore)
    maximum_dimension = 0.5 * sqrt(image_shape[0]**2 + image_shape[1]**2)
    y_pad = np.ceil(0.5 * (2*maximum_dimension - image_shape[0])).astype('int')
    x_pad = np.ceil(0.5 * (2*maximum_dimension - image_shape[1])).astype('int')

    # Rotate
    new_shape = tuple(np.array(image_shape)[:2] + [2*y_pad, 2*x_pad])
    image_center = (new_shape[1]/2, new_shape[0]/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return rotation_matrix, (x_pad, y_pad)


def rotate_image_and_crop(image: np.array, rotation_matrix: np.array, padding_dimensions: tuple,
                          contour_blob: np.array, border_value: int=0):
    """
    Rotates image with the specified angle
    :param image: image to be rotated
    :param angle: angle of rotation. if angle < 0 : clockwise rotation, if angle > 0 : counter-clockwise rotation
    :return: rotated image and rotation matrix
    """
    # Inspired by : https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/22042434#22042434
    x_pad, y_pad = padding_dimensions
    image_padded = cv2.copyMakeBorder(image, y_pad, y_pad, x_pad, x_pad,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=border_value
                                      )
    # Rotate
    rotated_image = cv2.warpAffine(image_padded, rotation_matrix, image_padded.shape[:2], flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
                                   )

    # Crop
    contour_blob = contour_blob + [x_pad, y_pad]
    rotated_contours = cv2.transform(contour_blob[None], rotation_matrix)
    rotated_image_cropped, (x_crop, y_crop, w, h) = crop_with_margin(rotated_image, cv2.boundingRect(rotated_contours),
                                                                     margin=2, return_coords=True)
    rotated_contours = rotated_contours - [x_crop, y_crop]

    return rotated_image_cropped, rotated_contours


def find_orientation_blob(blob_contour: np.array) -> (Tuple, np.array, float):
    """
    Uses PCA to find the orientation of blob
    :param blob_contour: contour of blob
    :return: center point, eignevectors, angle of orientation (in degrees)
    """

    # # Find orientation with PCA
    # blob_contours = blob_contours.reshape(blob_contours.shape[0], 2)  # must be of size nx2
    # mean, eigenvector = cv2.PCACompute(np.float32(blob_contours), mean=np.array([]))
    #
    # center = mean[0]
    # diagonal = eigenvector[0]
    # angle = np.arctan2(diagonal[1], diagonal[0])  # orientation in radians
    # angle *= (180/np.pi)
    #
    # # TODO : This needs to be checked...!
    # if diagonal[0] > 0:  # 1st and 4th cadran
    #     angle = angle
    # elif diagonal[0] < 0:  # 2nd and 3rd cadran
    #     angle = 90 - angle
    #
    # return center, eigenvector, angle

    center, diags, angle = cv2.fitEllipse(blob_contour)

    return center, diags, angle


def process_watershed_parcel(mask_parcel: np.array,
                             text_segmented_probs: np.array,
                             cadaster_imgae_gray: np.array,
                             transcription_model,
                             plotting_dir: str=None):
    """

    :param mask_parcel:
    :param text_segmented_probs:
    :param cadaster_imgae_gray:
    :param transcription_model:
    :param plotting_dir:
    :return:
    """
    # PARCEL EXTRACTION
    _, contours, _ = cv2.findContours(mask_parcel.astype('uint8').copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        current_polygon = MyPolygon(contours)
    else:
        return

    # Binarize probs
    text_binary_map = binarize_text_probs(text_segmented_probs)

    # LABEL EXTRACTION
    # Crop to have smaller image
    margin_text_label = 7
    bounding_rect = cv2.boundingRect(current_polygon.approximate_coordinates(epsilon=1))
    text_binary_crop, coordinates_crop_parcel = crop_with_margin(text_binary_map, bounding_rect,
                                                                 margin=margin_text_label, return_coords=True)
    x_crop_parcel, y_crop_parcel, w_crop_parcel, h_crop_parcel = coordinates_crop_parcel

    binary_parcel_number = crop_with_margin(mask_parcel, coordinates_crop_parcel, margin=0)
    binary_parcel_number = (255 * text_binary_crop * binary_parcel_number).astype('uint8')

    # Cleaning : Morphological opening
    binary_parcel_number = cv2.morphologyEx(binary_parcel_number, cv2.MORPH_OPEN,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Find parcel number but do not consider small elements (open)
    parcel_number_blob = cv2.dilate(binary_parcel_number, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    opening_kernel = (9, 9)
    parcel_number_blob = cv2.morphologyEx(parcel_number_blob, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, opening_kernel))
    # Get contours
    _, contours_blob_list, _ = cv2.findContours(parcel_number_blob.copy(), cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

    if contours_blob_list is None:
        return

    if plotting_dir is not None:
        parcel_number_probs = (255 * crop_with_margin(text_segmented_probs, coordinates_crop_parcel, margin=0)
                               * crop_with_margin(mask_parcel, coordinates_crop_parcel, margin=0)).astype('uint8')

        imsave(os.path.join(plotting_dir, '{}_number_binarization.jpg'.format(current_polygon.uuid)),
               binary_parcel_number)
        imsave(os.path.join(plotting_dir, '{}_parcel_blob.jpg'.format(current_polygon.uuid)), parcel_number_blob)
        imsave(os.path.join(plotting_dir, '{}_text_probs.jpg'.format(current_polygon.uuid)), parcel_number_probs)

    number_predicted_list = list()
    scores_list = list()
    label_contour_list = list()
    for i, contour_blob in enumerate(contours_blob_list):

        if len(contour_blob) < 5:  # There should be a least 5 points to fit the ellipse
            continue

        # Compute rotation matrix and padding
        _, _, angle = find_orientation_blob(contour_blob)
        rotation_matrix, x_pad, y_pad = get_rotation_matrix(binary_parcel_number.shape[:2], angle - 90)

        # Crop on grayscale image and rotate
        image_parcel_number = cadaster_imgae_gray[y_crop_parcel:y_crop_parcel + h_crop_parcel,
                              x_crop_parcel:x_crop_parcel + w_crop_parcel]
        image_parcel_number_rotated, rotated_contours = rotate_image_and_crop(image_parcel_number, rotation_matrix,
                                                                              (x_pad, y_pad),
                                                                              contour_blob[:, 0, :],
                                                                              border_value=128)

        x_box, y_box, w_box, h_box = cv2.boundingRect(rotated_contours)
        margin_box = 0
        grayscale_number_crop = image_parcel_number_rotated[y_box + margin_box:y_box + h_box - margin_box,
                                x_box + margin_box:x_box + w_box - margin_box]

        if grayscale_number_crop.size < 100: # element shouldn't be too small
            continue

        if plotting_dir is not None:
            imsave(os.path.join(plotting_dir, '{}_label_crop{}.jpg'.format(current_polygon.uuid, i)),
                   image_parcel_number)
            imsave(os.path.join(plotting_dir, '{}_label_rotated{}.jpg'.format(current_polygon.uuid, i)),
                   grayscale_number_crop)

        # TRANSCRIPTION
        try:
            predictions = transcription_model.predict(grayscale_number_crop[:, :, None])
            number_predicted_list.append(predictions['words'][0].decode('utf8'))
            scores_list.append(predictions['score'][0])
            label_contour_list.append((contour_blob[:, 0, :] + [x_crop_parcel, y_crop_parcel])[:, None, :])
        except:
            print('WARNING -- there was an error when transcribing')
            pass

    # Add transcription and score to Polygon object
    current_polygon.assign_transcription(number_predicted_list, scores_list, label_contour_list)

    return current_polygon
