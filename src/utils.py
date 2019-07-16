#!/usr/bin/env python
__author__ = 'solivr'

import uuid
import cv2
import numpy as np
import gdal
import re
from typing import List
import geojson


class MyPolygon:

    def __init__(self, contours: np.array):
        """
        :param contours : list of contours. [ [[x1,y1], [x2,y2], ...],
                                              [[x1.y1], [x2,y2], ...],
                                              ... ]
        """
        # TODO change contours to shapely Polygon object
        self.contours = contours  # shape [(N_POINTS, 1, 2)]
        self.uuid = self._generate_uuid()
        self.transcription = None
        self.score = None
        self.georeferenced_contours = None
        self._label_contours = None
        self.best_transcription = None

    def assign_transcription(self, transcription, score, label_contour):
        self.transcription = transcription
        self.score = score
        self._label_contours = label_contour
        self.best_transcription = self.find_best_transcription()

    @staticmethod
    def _generate_uuid():
        return str(uuid.uuid4())

    def find_best_transcription(self):
        if self.score:
            return self.transcription[np.argmax(self.score)]
        else:
            return ''

    def approximate_coordinates(self, epsilon=1, inner=False):
        """

        :param epsilon:
        :param inner: return also inner contours (if there is a hole)
        :return:
        """
        if inner:
            approx_contours = list()
            for c in self.contours:
                approx_contours.append(cv2.approxPolyDP(c, epsilon, closed=True))
            return approx_contours
        else:
            return cv2.approxPolyDP(self.contours[0], epsilon, closed=True)

    @staticmethod
    def georeferenecing(contours: np.array, geotransform: tuple) -> np.array:
        """
        Georeferencing for geojson (will have no effect if no geographic metadata is found)
        From : http://www.gdal.org/classGDALDataset.html#a5101119705f5fa2bc1344ab26f66fd1d
             GeoTransform[0] / * top left x
             GeoTransform[1] / * w - e pixel resolution (width)
             GeoTransform[2] / * rotation, 0 if image is "north up"
             GeoTransform[3] / * top left y */
             GeoTransform[4] / * rotation, 0 if image is "north up"
             GeoTransform[5] / * n - s pixel resolution (height)
             Xp = geo_transform[0] + row*geo_transform[1] + col*geo_transform[2];
             Yp = geo_transform[3] + row*geo_transform[4] + col*geo_transform[5];

        :param contours : list of contours. [ [[x1,y1], [x2,y2], ...],
                                              [[x1.y1], [x2,y2], ...],
                                              ... ]
        """
        georeferenced_contours = list()
        for coordinates in contours:
            georeferenced_coordinates = [(geotransform[0] + pt[0] * geotransform[1] + pt[1] * geotransform[2],
                                          geotransform[3] + pt[0] * geotransform[4] + pt[1] * geotransform[5])
                                         for pt in coordinates[:, 0, :]]
            georeferenced_coordinates.append(georeferenced_coordinates[0])
            georeferenced_contours.append(georeferenced_coordinates)

        return georeferenced_contours

    @property
    def label_contours(self):
        return self._label_contours


class GeoProjection:

    def __init__(self, projection_name: str=None):
        if projection_name == 'Monte Mario':
            self.crs = {"type": "name",
                        "properties": {
                            "name": "urn:ogc:def:crs:EPSG::3004"
                        }
                        }
        elif projection_name == 'WGS84':
            self.crs = {"type": "name",
                        "properties": {
                            "name": "urn:ogc:def:crs:EPSG::4326"
                        }
                        }
        elif projection_name is None:
            self.crs = {"type": "name",
                        "properties": {
                            "name": "urn:ogc:def:crs:EPSG::3004"
                        }
                        }
        else:
            raise NotImplementedError

    @staticmethod
    def get_geoprojection_from_file(filename_gtiff):
        ds = gdal.Open(filename_gtiff)
        try:
            toks = re.search('AUTHORITY\[\".*,.*\"\],', ds.GetProjectionRef()) \
                       .group()[len('AUTHORITY[\"'):-len('\"],')].split('\"')
            crs = {"type": "name",
                   "properties": {
                       "name": "urn:ogc:def:crs:{}::{}".format(toks[0], toks[2])
                   }
                   }
        except AttributeError:
            print('No CRS found in file {}'.format(filename_gtiff))
            crs = None
        return crs


# def export_geojson(list_polygons: List[MyPolygon], export_filename: str, crs) -> None:
#
#     # Object to save
#     collectionPolygons = geojson.FeatureCollection([poly for poly in list_polygons], crs=crs)
#
#     # Save file
#     with open(export_filename, 'w') as outfile:
#         # TODO : check if sortkeys=True is the best (maybe have bigger polygons first)
#         geojson.dump(collectionPolygons, outfile, sort_keys=True)


def export_geojson(list_polygon_objects: List[MyPolygon], export_filename: str, filename_img: str):
    # TODO : Already do some postprocessing like removing polygons with 3 or less coordinates, empty transcriptions, ...

    # Geographic info
    ds = gdal.Open(filename_img)
    geotransform = ds.GetGeoTransform()
    projection = GeoProjection(GeoProjection.get_geoprojection_from_file(filename_img))

    collection_polygons = geojson.FeatureCollection(
        [geojson.Feature(geometry=geojson.Polygon(MyPolygon.georeferenecing(polygon.approximate_coordinates(epsilon=2,
                                                                                                            inner=True),
                                                                            geotransform=geotransform)),
                         properties={'uuid': polygon.uuid,
                                     'transcription': str(polygon.transcription),
                                     'score': str(polygon.score),
                                     'best_transcription' : str(polygon.best_transcription)
                                     }) for polygon in list_polygon_objects],
                                     # 'score': [str(score) for score in polygon.score]}) for polygon in list_polygon_objects],
        crs=projection.crs)

    # Save file
    with open(export_filename, 'w') as outfile:
        geojson.dump(collection_polygons, outfile, sort_keys=True)

