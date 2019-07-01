***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

***REMOVED***
import geopandas as gpd
import pandas as pd
***REMOVED***
import re
from typing import List
***REMOVED***
from .geo_info import get_georef_data, remove_geo_offset


def concat_geojson_files(list_geojson_files: List[str],
                         geotif_files_directory: str,
                         remove_geographical_offset: bool=False):
***REMOVED***"
    Given a list of geojson files, concatenates them, and also keeps track of which geometry belongs to which file / image

    :param list_geojson_files: list containing the filenames of the geojson files
    :param geotif_files_directory: directory containign the corresponding (geo)tif images
    :param remove_geographical_offset: either to remove geographical offset or to keep geographical coordinates
    :return:
***REMOVED***"
    geodataframes_to_concatenate = list()
    for filename in tqdm(list_geojson_files):
        gdf = gpd.read_file(filename)

        # add colum with tif filename
        _, basenme = os.path.split(filename)
        id_map_sheet = re.search('[0-9]***REMOVED***2***REMOVED***', basenme).group()
        tif_filename_list = glob(os.path.abspath(os.path.join(geotif_files_directory, '****REMOVED******REMOVED****.tif*'.format(id_map_sheet))))
        if len(tif_filename_list) != 1:
            raise ValueError('There are more than 1 item with id ***REMOVED******REMOVED***'.format(id_map_sheet))
        else:
            tif_filename = tif_filename_list[0]
        gdf = gdf.assign(image_filename=os.path.basename(tif_filename))

        if remove_geographical_offset:
            # Transform georeferenced coordinates to image-relative coordinates
            x0, dx, _, y0, _, dy = get_georef_data(tif_filename)
            gdf['geometry'] = gdf.geometry.apply(lambda s: remove_geo_offset(s, x0, dx, y0, dy))

        # Add GeoDataframe to list of GeoDataframes to concatenate
        geodataframes_to_concatenate.append(gdf)

    if len(geodataframes_to_concatenate) > 1:
        # Make a concatenated GeoDataframe with all the data
        gdf_global = gpd.GeoDataFrame(pd.concat(geodataframes_to_concatenate, ignore_index=True))
    elif len(geodataframes_to_concatenate) == 1:
        gdf_global = geodataframes_to_concatenate[0]
    else:
        raise NotImplementedError

    return gdf_global
