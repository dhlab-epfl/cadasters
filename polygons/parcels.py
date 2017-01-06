import numpy as np
import cv2
from .flooding import Polygon2geoJSON
from .flooding import clean_image_ridge


def find_parcels(nodes_bg2flood, merged_segments, ridge_image, ksize_kernel_flooding):
    """
    Given the possible 'background' regions, finds the parcels by flooding the regions
    and returns the found polygons as GEOJSON file and opencv format. Also matches an uuid
    to each generated polygon/parcel.

    :param nodes_bg2flood: graph nodes corresponding to regions to be flooded
    :param merged_segments: map of segments/regions
    :param ridge_image: ridge image (frangi feature)
    :param ksize_kernel_flooding: size of kernel to dilate and erode ridge image (to precporcess it)
    :return: listFeatPolygon: list of Polygon in GeoJSON format
            dic_polygon: dictionary with node graphs as keys and
                        tuple of (uuid, list of polygons in cv2 format) as values
    """

    # List to be updated during loop
    listFeatPolygon = list()
    dic_polygon = {}

    # Get clean ridge image
    ridges = clean_image_ridge(ridge_image, ksize_kernel_flooding)

    for bgn in nodes_bg2flood:
        # Make binary image of zone to flood
        binary_image_bg = np.zeros(merged_segments.shape, 'uint8')
        binary_image_bg[merged_segments == bgn] = 1

        # Find contours
        i, contours, h = cv2.findContours(binary_image_bg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Fit polygon
        epsilon = 3
        poly = list()
        for c in contours:
            poly.append(cv2.approxPolyDP(c, epsilon, closed=True))
        # poly = cv2.approxPolyDP(contours[0], epsilon, closed=True)

        # Flooding of area to get fitted polygon
        parcel = Polygon2geoJSON(poly, listFeatPolygon, bgn, ridges, ksize_kernel_flooding)

        # G.node[bgn]['polygon'] = parcel
        dic_polygon[bgn] = parcel

#       # Convex hull
#       hull = cv2.convexHull(poly)

    return listFeatPolygon, dic_polygon

