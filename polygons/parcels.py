import numpy as np
import cv2
from .flooding import Polygon2geoJSON
from .flooding import clean_image_ridge
from geojson import Feature, Polygon
from osgeo import gdal


def find_parcels(nodes_bg2flood, merged_segments, ridge_image, ksize_kernel_flooding, img_filename):
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

    # Georeferencing for geojson (will have no effect if no geographic metadata is found)
    #     From : http://www.gdal.org/classGDALDataset.html#a5101119705f5fa2bc1344ab26f66fd1d
    #     GeoTransform[0] / * top left x
    #     GeoTransform[1] / * w - e pixel resolution (width)
    #     GeoTransform[2] / * rotation, 0 if image is "north up"
    #     GeoTransform[3] / * top left y */
    #     GeoTransform[4] / * rotation, 0 if image is "north up"
    #     GeoTransform[5] / * n - s pixel resolution (height)
    #     Xp = geo_transform[0] + row*geo_transform[1] + col*geo_transform[2];
    #     Yp = geo_transform[3] + row*geo_transform[4] + col*geo_transform[5];

    ds = gdal.Open(img_filename)
    geo_transform = ds.GetGeoTransform()

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

        # Flooding of area to get fitted polygon
        parcels = Polygon2geoJSON(poly, ridges, ksize_kernel_flooding)

        dic_polygon[bgn] = parcels

        # Transform polygon parcel to geoJSON Polygon format
        for (uuid, poly) in parcels:
            final_polygon = list()

            for cnt in poly:
                poly_contours = list()

                for pt in cnt:
                    ptx = float(pt[0, 0])
                    pty = float(pt[0, 1])
                    geo_ptx = geo_transform[0] + ptx * geo_transform[1] + pty * geo_transform[2]
                    geo_pty = geo_transform[3] + ptx * geo_transform[4] + pty * geo_transform[5]
                    poly_contours.append((geo_ptx, geo_pty))

                poly_contours.append(poly_contours[0])  # close polygon
                final_polygon.append(poly_contours)

            if final_polygon:
                myFeaturePoly = Feature(geometry=Polygon(final_polygon),
                                        properties={"uuid": uuid})
                listFeatPolygon.append(myFeaturePoly)

    return listFeatPolygon, dic_polygon

