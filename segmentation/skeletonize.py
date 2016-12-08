import numpy as np
import cv2


def skeletonize(bin_img):
    """
    Skeletonize binary image
    :param bin_img: binary image
    :return: skeleton
    """
    size = np.size(bin_img)
    skel = np.zeros(bin_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(bin_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(bin_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        bin_img = eroded.copy()

        # When there is only a shape of one pixel-width
        zeros = size - cv2.countNonZero(bin_img)
        if zeros == size:
            done = True

    return skel