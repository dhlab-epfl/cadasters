import numpy as np
import cv2
from typing import List


def rgb2gray(img, channel_order= 'RGB'):
    """
    Convert RGB image into grayscale image
    :param img: 3-channel RGB or BGR image to be converted into gray
    :param channel_order: chose 'BGR' or 'RGB' (default)
    :return: 1-channel gray image using the formula : 0.2989*R + 0.5870*G + 0.1140*B
    """
    imgf = np.float64(img)
    if channel_order == 'RGB':
        R = imgf[:,:,0]
        G = imgf[:,:,1]
        B = imgf[:,:,2]
    elif channel_order == 'BGR':
        R = imgf[:,:,2]
        G = imgf[:,:,1]
        B = imgf[:,:,0]

    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray
# --------------------------------------


def bgr2rgb(img_bgr):
    """
    Invert 3rd and first color channel
    :param img_bgr: BGR image
    :return: RGB image
    """
    img_rgb = img_bgr.copy()
    # Change BGR to RGB
    tmp_im = img_rgb[:, :, 0]
    img_rgb[:, :, 0] = img_rgb[:, :, 2]
    img_rgb[:, :, 2] = tmp_im

    return img_rgb
# --------------------------------------


def merge_segments(slic_segments: np.array, seg_to_merge: List[int], label: int):
    """
    Merge segments in seg_to_merge and update the map of segments list_segments

    :param slic_segments: Map of labels of segments/superpixels
    :param seg_to_merge: list of segment's labels to merge
    :param label: new label to assign to the merged segments
    """

    # Alternative
    for s in seg_to_merge:
        # Find where the segment to merge is
        seg_new_label = slic_segments == s
        # Assign a new label to the segments
        slic_segments[seg_new_label] = label
# --------------------------------------


def padding(image, value, npad=None):
    """
    Add a border to image
    :param image: image to be padded
    :param value: value to use for padding
    :param npad: number of zero pixels to add to the border
    :return: padded image
    """

    if npad is None :
        npad = np.int0(np.round(0.5*max(image.shape)))

    # For image with 3 channels
    if len(image.shape) > 2:
        new_size = np.array(image.shape) + 2*np.array([npad, npad, 0])

        img_pad = value*np.ones(new_size, image.dtype)
        img_pad[npad:-npad, npad:-npad, :] = image

    else:
        new_size = np.array(image.shape) + 2*np.array([npad, npad])

        img_pad = value*np.ones(new_size, image.dtype)
        img_pad[npad:-npad, npad:-npad] = image

    return img_pad
# --------------------------------------


def rotate_image(image, angle):
    """
    Rotates image with the specified angle
    :param image: image to be rotated
    :param angle: angle of rotation. if angle < 0 : clockwise rotation, if angle > 0 : counter-clockwise rotation
    :return: rotated image
    """

    if len(image.shape) > 2:
        image_center = tuple(np.array(image.shape[:2])/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        new_shape = np.array([image.shape[1], image.shape[0]])
        rotated = cv2.warpAffine(image, rot_mat, tuple(new_shape), flags=cv2.INTER_LANCZOS4,
                                 # borderMode=cv2.BORDER_REFLECT
                                 )
    else:
        image_center = tuple(np.array(image.shape)/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        new_shape = np.array([image.shape[1], image.shape[0]])
        rotated = cv2.warpAffine(image, rot_mat, tuple(new_shape), flags=cv2.INTER_LANCZOS4,
                                 # borderMode=cv2.BORDER_REFLECT
                                 )

    return rotated


def rotate_image_with_mat(image, angle):
    """
    Rotates image with the specified angle
    :param image: image to be rotated
    :param angle: angle of rotation. if angle < 0 : clockwise rotation, if angle > 0 : counter-clockwise rotation
    :return: rotated image and rotation matrix
    """

    if len(image.shape) > 2:
        image_center = tuple(np.array(image.shape[:2])/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        new_shape = np.array([image.shape[1], image.shape[0]])
        rotated = cv2.warpAffine(image, rot_mat, tuple(new_shape), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT
                                 )
    else:
        image_center = tuple(np.array(image.shape)/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        new_shape = np.array([image.shape[1], image.shape[0]])
        rotated = cv2.warpAffine(image, rot_mat, tuple(new_shape), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT
                                 )

    return rotated, rot_mat
