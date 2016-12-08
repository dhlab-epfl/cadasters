import numpy as np
import cv2
from geojson import Feature, Polygon
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


def Polygon2geoJSON(polyCV2, listFeatPolygon, node_graph, img_frangi, offset):
    """
    Given a ridge image and the contours of a possible polygon, runs the floodfill
    algorithm taking as seed point a pixel inside the possible parcel.

    :param polyCV2: polygon in openCV format (aproxPolyDP output)
    :param node_graph: id of graph node corresponding to polygon
    :param img_frangi: ridge image
    :param offset: offset corresponding to the difference between original ridge image and cropped one.
                This is used to realign coordinates of cropped ridge image with original coordinates
    :return: parcels: list of tuples (uuid, polygon (opencv format)), list of FeaturePolygon (geoJSON format) updated
            in listFEatPolygon variable
    """

    # Mask for floodfill algorithm
    mask = img_frangi.copy()
    # Adding border and padding the border with ones
    mask = np.c_[np.ones(mask.shape[0]), mask, np.ones(mask.shape[0])]
    mask = np.r_[[np.ones(mask.shape[1])], mask, [np.ones(mask.shape[1])]]
    mask = np.uint8(mask)


    parcels = list()
    img2flood = img_frangi.copy()
    kernel_concav = np.ones((11, 11), np.uint8)

    # Find size of optimal kernel
    kernel = np.ones((3, 3), np.uint8)
    # kernel = size_kernel_erode(polyCV2)

    # SeedMask -> one point will be used as seed
    seedmask = np.zeros(img_frangi.shape)
    cv2.fillPoly(seedmask, [polyCV2], 255)
    # Erosion to make sure seed is strictly inside the region to flood (and not in the borders)
    seedmask = cv2.erode(seedmask, kernel)

    new_seed = True
    while new_seed:
        # Find seed point. Should be != 0 in img_frangi
        found_seed = False
        nonzero_points = np.transpose(np.nonzero(seedmask)) # possible seedpoints candidates
        for pt in nonzero_points:
            seed_point = tuple(pt)
            # Check if it is a BG region and if it has not already been flooded
            if img2flood[seed_point] == 0:
                if mask[seed_point[0]+1, seed_point[1]+1] == 0: # (remember mask has been padded with 1 additional outside border)
                    # Inverse column and row for opencv format
                    seed_point = (seed_point[1], seed_point[0])
                    found_seed = True
                    break

        if found_seed:
            # Keep track of the previous image flooded (to compute difference later)
            prev_imgflood = img2flood.copy()
            tmp = cv2.floodFill(img2flood, mask, seed_point, 128)

            # Get the flooded zone
            flooded_zone = np.absolute(img2flood - prev_imgflood)

            # Dilate and erode to get a closed polygon (non concave)
            aprox_parcel = cv2.dilate(flooded_zone, kernel_concav, iterations=1)
            aprox_parcel = cv2.erode(aprox_parcel, kernel_concav, iterations=1)

            # Make a polygon of it
            tmp, cnt, tmp = cv2.findContours(aprox_parcel.copy(), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
            aprox_parcel = cv2.approxPolyDP(cnt[0], 5, True)
            aprox_parcel = aprox_parcel + [offset, offset]

            # Generate uuid
            uid = str(uuid.uuid4())
            parcels.append((uid, aprox_parcel))

            # Transform polygon aproxParcel to geoJSON Polygon format
            poly_points = [(float(pt[0, 0]), float(pt[0, 1])) for pt in aprox_parcel]
            myFeaturePoly = Feature(geometry=Polygon([poly_points]),
                                    properties={"uuid": uid, "node": str(node_graph)})
            listFeatPolygon.append(myFeaturePoly)

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


# def DicPolygons2geoJSON(dic_polygons, img_frangi, namesavefile):
#
#     # Raw Polygons from graph
#     # ---------
#     polygons = [dic_polygons[p] for p in dic_polygons.keys()]
#     idnodes = [n for n in dic_polygons.keys()]
#
#     # Prepare ridge image
#     # -------------------
#     kernel_size = 3 # Size of the kernel for opening and closing, and also width of the removed border of the image
#     offset = kernel_size # offset that will be added to align cropped image coordinates with original coordinates
#     img_frangi, mask = compute_images_for_flooding(img_frangi, kernel_size)
#
#     # Zones/polygons to export
#     listPolygon = list()
#
#     # Params
#     kernel = np.ones((6,6),np.uint8)
#
#     # Graph node id corresponding to the processed polygon
#     id = 0
#     for pol in polygons:
#         parcels = Polygon2geoJSON(pol, listPolygon, idnodes[id], img_frangi, mask, kernel, offset)
#         id += 1
#
#     savePolygons(listPolygon, namesavefile)
