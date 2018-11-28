***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from osgeo import gdal, gdalconst
from shapely.ops import transform
from shapely.geometry import Polygon
***REMOVED***
import geopandas as gpd
import re
***REMOVED***
from typing import Tuple


def get_georef_data(tif_filename: str) -> Tuple[float, float, float, float, float, float]:
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


def remove_geo_offset(shape: Polygon, x_offset: float, dx: float, y_offset: float, dy: float, rounding_digits: int=5) \
        -> Tuple[float, float]:
    return transform(lambda x, y: (round((x-x_offset)/dx, rounding_digits),
                                   round((y-y_offset)/dy, rounding_digits)), shape)


def offset_geojson_file(geojson_filename: str, geotif_directory: str, output_dir: str) -> None:
***REMOVED***"
    This function will convert back georeferenced shape to shapes with their corresponding image coordinates.

    :param geojson_filename: filename of the geojson file to convert
    :param geotif_directory: path to the directory where the images (geotif) are stored
    :param output_dir: path of the output directory to export the resulting geojson
***REMOVED***"
    # find corresponding tif image
    _, basename = os.path.split(geojson_filename)
    id_map_sheet = re.search('[0-9]***REMOVED***2***REMOVED***', basename).group()
    tif_filename_list = glob(os.path.abspath(os.path.join(geotif_directory, '****REMOVED******REMOVED****.tif*'.format(id_map_sheet))))

    if len(tif_filename_list) != 1:
        raise ValueError
    else:
        tif_filename = tif_filename_list[0]

    # Get projection parameters
    x0, dx, _, y0, _, dy = get_georef_data(tif_filename)

    # Read filename
    gdf = gpd.read_file(geojson_filename)
    gdf['geometry'] = gdf.geometry.apply(lambda s: remove_geo_offset(s, x0, dx, y0, dy))

***REMOVED***
    filename_output = os.path.join(output_dir, basename)
    gdf.to_file(filename_output, driver='GeoJSON', encoding='utf8')
