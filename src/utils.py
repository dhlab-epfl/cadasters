***REMOVED***
***REMOVED***

import uuid
import cv2
import numpy as np
import gdal
import re
from typing import List, Tuple
import geojson
from math import sqrt
from shapely.geometry import Polygon


class MyPolygon:

    def __init__(self, contours: np.array):
    ***REMOVED***"
        :param contours : list of contours. [ [[x1,y1], [x2,y2], ...],
                                              [[x1.y1], [x2,y2], ...],
                                              ... ]
    ***REMOVED***"
        # TODO shange contours to shapely Polygon object
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
    ***REMOVED***"

        :param epsilon:
        :param inner: return also inner contours (if there is a hole)
        :return:
    ***REMOVED***"
        if inner:
            approx_contours = list()
            for c in self.contours:
                approx_contours.append(cv2.approxPolyDP(c, epsilon, closed=True))
            return approx_contours
        else:
            return cv2.approxPolyDP(self.contours[0], epsilon, closed=True)

    @staticmethod
    def georeferenecing(contours: np.array, geotransform: tuple) -> np.array:
    ***REMOVED***"
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
    ***REMOVED***"
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
            self.crs = ***REMOVED***"type": "name",
                        "properties": ***REMOVED***
                            "name": "urn:ogc:def:crs:EPSG::3004"
                    ***REMOVED***
                    ***REMOVED***
        elif projection_name == 'WGS84':
            self.crs = ***REMOVED***"type": "name",
                        "properties": ***REMOVED***
                            "name": "urn:ogc:def:crs:EPSG::4326"
                    ***REMOVED***
                    ***REMOVED***
        elif projection_name is None:
            self.crs = ***REMOVED***"type": "name",
                        "properties": ***REMOVED***
                            "name": "urn:ogc:def:crs:EPSG::3004"
                    ***REMOVED***
                    ***REMOVED***
        else:
            raise NotImplementedError

    @staticmethod
    def get_geoprojection_from_file(filename_gtiff):
        ds = gdal.Open(filename_gtiff)
        try:
            toks = re.search('AUTHORITY\[\".*,.*\"\],', ds.GetProjectionRef()) \
                       .group()[len('AUTHORITY[\"'):-len('\"],')].split('\"')
            crs = ***REMOVED***"type": "name",
                   "properties": ***REMOVED***
                       "name": "urn:ogc:def:crs:***REMOVED******REMOVED***::***REMOVED******REMOVED***".format(toks[0], toks[2])
               ***REMOVED***
               ***REMOVED***
        except AttributeError:
            print('No CRS found in file ***REMOVED******REMOVED***'.format(filename_gtiff))
            crs = None
        return crs


def export_geojson(list_polygons: List[MyPolygon], export_filename: str, crs) -> None:

    # Object to save
    collectionPolygons = geojson.FeatureCollection([poly for poly in list_polygons], crs=crs)

    # Save file
    with open(export_filename, 'w') as outfile:
        # TODO : check if sortkeys=True is the best (maybe have bigger polygons first)
        geojson.dump(collectionPolygons, outfile, sort_keys=True)


def find_orientation_blob(blob_contour: np.array) -> (Tuple, np.array, float):
***REMOVED***"
    Uses PCA to find the orientation of blob
    :param blob_contour: contour of blob
    :return: center point, eignevectors, angle of orientation (in degrees)
***REMOVED***"

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


def crop_with_margin(full_img: np.array, crop_coords: np.array, margin: int=0, return_coords: bool=False):
***REMOVED***"

    :param box_coords: (x, y, w, h)
    :param full_img:
    :param margin:
    :return:
***REMOVED***"

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
***REMOVED***"
    Rotates image with the specified angle
    :param image: image to be rotated
    :param angle: angle of rotation. if angle < 0 : clockwise rotation, if angle > 0 : counter-clockwise rotation
    :return: rotated image and rotation matrix
***REMOVED***"
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
                         properties=***REMOVED***'uuid': polygon.uuid,
                                     'transcription': str(polygon.transcription),
                                     'score': str(polygon.score),
                                     'best_transcription' : str(polygon.best_transcription)
                                 ***REMOVED***) for polygon in list_polygon_objects],
                                     # 'score': [str(score) for score in polygon.score]***REMOVED***) for polygon in list_polygon_objects],
        crs=projection.crs)

    # Save file
    with open(export_filename, 'w') as outfile:
        geojson.dump(collection_polygons, outfile, sort_keys=True)

