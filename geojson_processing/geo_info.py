***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from osgeo import gdal, gdalconst
from shapely.ops import transform


def get_georef_data(tif_filename):
***REMOVED***"
    Fetches the coefficients for transforming between pixel/line (P,L) raster space,
    and projection coordinates (Xp,Yp) space from the TIF file header.

    Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2];
    Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5];

    In a north up image, padfTransform[1] is the pixel width, and padfTransform[5] is the pixel height.
    The upper left corner of the upper left pixel is at position (padfTransform[0],padfTransform[3]).

    :param tif_filename: filename of the tif image file
    :return: the geotransform (x0, dx, _, y0, _, dy)
***REMOVED***"

    src_exif_data = gdal.Open(tif_filename, gdalconst.GA_ReadOnly)

    return src_exif_data.GetGeoTransform()  # x0, dx, _, y0, _, dy


def remove_geo_offset(shape, x_offset, dx, y_offset, dy, rounding_digits=5):
    return transform(lambda x, y: (round((x-x_offset)/dx, rounding_digits),
                                   round((y-y_offset)/dy, rounding_digits)), shape)
