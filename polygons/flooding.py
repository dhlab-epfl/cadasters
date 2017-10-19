#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
import cv2
import uuid


def clean_image_ridge(img_ridge, ksize):
    """
    Binarizes and removes small elements from the ridge image.

    :param img_ridge: Image of ridge detector (containing the 'vesselness measure')
    :param ksize: size of the kernel for opening nd closing (mathematical morphology)
    :return: img_ridge: binarized version of the input ridge image, also smaller by 2*ksize
                            in each dimension
    """

    # Discard pixels where the vesselness measure is less than 5%
    # And make a binary image of it
    img_ridge = np.uint8(255*(img_ridge > 0.05))

    # Opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opening = cv2.morphologyEx(img_ridge, cv2.MORPH_OPEN, kernel)
    img_ridge = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # Don't take border columns and rows because the opening and closing
    # may have deleted some edge points (and thus floodfill will leak)
    img_ridge = img_ridge[ksize:-ksize,ksize:-ksize]
    # Pading the borders with ones
    img_ridge[:, 0] = 255
    img_ridge[:, -1] = 255
    img_ridge[0, :] = 255
    img_ridge[-1, :] = 255

    return img_ridge
# --------------------------------------------------------------------------


# def compute_images_for_flooding(img_ridge, ksize):
#     """
#
#     :param img_ridge: Image of ridge detector (containing the 'vesselness measure')
#     :param ksize: size of the kernel for opening nd closing (mathematical morphology)
#     :return: img_ridge: binarized version of the input ridge image, also smaller by 2*ksize
#                             in each dimension
#             mask:
#     """
#
#     # Discard pixels where the vesselness measure is less than 5%
#     # And make a binary image of it
#     img_ridge = np.uint8(255*(img_ridge > 0.05))
#
#     # Opening and closing
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
#     opening = cv2.morphologyEx(img_ridge, cv2.MORPH_OPEN, kernel)
#     img_ridge = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#
#     # Don't take border columns and rows because the opening and closing
#     # may have deleted some edge points (and thus floodfill will leak)
#     img_ridge = img_ridge[ksize:-ksize,ksize:-ksize]
#     # Pading the borders with ones
#     img_ridge[:,0] = 255
#     img_ridge[:,-1] = 255
#     img_ridge[0,:] = 255
#     img_ridge[-1,:] = 255
#
#     # # Mask for floodfill algorithm
#     # mask = img_ridge.copy()
#     # # Adding border and padding the border with ones
#     # mask = np.c_[np.ones(mask.shape[0]), mask, np.ones(mask.shape[0])]
#     # mask = np.r_[[np.ones(mask.shape[1])], mask, [np.ones(mask.shape[1])]]
#     # mask = np.uint8(mask)
#
#     return img_ridge, mask
# ---------------------------------------------------------------------------------------


def Polygon2geoJSON(polyCV2, img_frangi: np.array, offset: int) -> (str, list):
    """
    Given a ridge image and the contours of a possible polygon, runs the floodfill
    algorithm taking as seed point a pixel inside the possible parcel.

    :param polyCV2: polygon in openCV format (aproxPolyDP output)
    :param img_frangi: ridge image
    :param offset: offset corresponding to the difference between original ridge image and cropped one.
                This is used to realign coordinates of cropped ridge image with original coordinates
    :return: parcels: list of tuples (uuid, polygon (opencv format))
    """

    # Mask for floodfill algorithm
    mask = img_frangi.copy()
    # Adding border and padding the border with ones
    mask = np.c_[np.ones(mask.shape[0]), mask, np.ones(mask.shape[0])]
    mask = np.r_[[np.ones(mask.shape[1])], mask, [np.ones(mask.shape[1])]]
    mask = np.uint8(mask)

    parcels = list()
    img2flood = img_frangi
    kernel_concav = np.ones((15, 15), np.uint8)
    kernel_holes = np.ones((19, 19), np.uint8)

    # Find size of optimal kernel
    kernel = np.ones((3, 3), np.uint8)
    # kernel = size_kernel_erode(polyCV2)

    # SeedMask -> one point will be used as seed
    seedmask = np.zeros(img_frangi.shape)
    cv2.fillPoly(seedmask, polyCV2, 255)
    # Erosion to make sure seed is strictly inside the region to flood (and not in the borders)
    seedmask = cv2.erode(seedmask, kernel)

    new_seed = True
    while new_seed:
        # Find seed point. Should be != 0 in img_frangi
        found_seed = False
        nonzero_points = np.transpose(np.nonzero(seedmask))  # possible seedpoints candidates
        for pt in nonzero_points:
            # Check if it is a BG region and if it has not already been flooded
            if (img2flood[tuple(pt)] == 0) and (mask[pt[0]+1, pt[1]+1] == 0):  # (remember mask has been padded with 1 additional outside border)
                # Inverse column and row for opencv format
                seed_point = (pt[1], pt[0])
                found_seed = True
                break

        if found_seed:
            # Keep track of the previous image flooded (to compute difference later)
            prev_imgflood = img2flood.copy()
            tmp = cv2.floodFill(img2flood, mask, seed_point, 128)  # mask is not filled, only img2flood

            # Get the flooded zone
            flooded_zone = np.absolute(img2flood - prev_imgflood)

            # Dilate and erode to get a closed polygon (non concave)
            aprox_parcel = cv2.dilate(flooded_zone, kernel_concav, iterations=1)
            aprox_parcel = cv2.erode(aprox_parcel, kernel_concav, iterations=1)

            # Closing to close small holes (smaller than 20x20)
            aprox_parcel = cv2.morphologyEx(aprox_parcel, cv2.MORPH_CLOSE, kernel_holes)

            # Make a polygon of it
            tmp, cnt, tmp = cv2.findContours(aprox_parcel.copy(), cv2.RETR_CCOMP,
                                             cv2.CHAIN_APPROX_SIMPLE)
            aprox_parcel = list()
            for c in cnt:
                poly = cv2.approxPolyDP(c, 5, True)
                poly = poly + [offset, offset]
                aprox_parcel.append(poly)

            # Generate uuid
            uid = str(uuid.uuid4())
            parcels.append((uid, aprox_parcel))

            # Find zones that have not been flooded (in case a node include two distinct zones to flood)
            non_flooded_zones = (1*(seedmask > 0) - 1*(flooded_zone > 0)) > 0
            if sum(non_flooded_zones.flatten()) > 0:
                seedmask = np.array(255*non_flooded_zones, 'uint8')
                # Erosion and opening to make sure seed is strictly inside the region to flood (and not in the borders)
                # kernel = size_kernel_erode(polyCV2)
                seedmask = cv2.erode(seedmask, kernel)
            else:
                new_seed = False
        else:
            new_seed = False

    return parcels
# -------------------------------------------------------------------------------------
