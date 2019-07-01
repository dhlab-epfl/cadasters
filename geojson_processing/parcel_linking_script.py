# ***REMOVED***
# __author__ = "solivr"
# __license__ = "GPL"
#
# import geopandas as gpd
# import pandas as pd
# ***REMOVED***
# import re
# from typing import List, Union
# ***REMOVED***
# from .geo_info import get_georef_data, remove_geo_offset
# from .filtering import remove_empty_transcripts
#
#
# def process_parcels_with_numbers(input_geojson: Union[List[str], str],
#                                  geotif_directory: str,
#                                  remove_geographical_offset: bool=True):
# ***REMOVED***"
#
#     :param input_geojson: input geojson file(s). If several files are provided, it will concatenate their result
#     :param geotif_directory: directory where are situated the geotif image files. \
#     These will be used to get the geographical information
#     :param remove_geographical_offset: either to remove geographical offset or to keep geographical coordinates
#     :return:
# ***REMOVED***"
#
#     if not isinstance(input_geojson, list):
#         input_geojson = [input_geojson]
#
#     geodataframes_to_concatenate = list()
#     for filename in input_geojson:
#         gdf = gpd.read_file(filename)
#         gdf_with_numbers = remove_empty_transcripts(gdf)
#
#         # add column with tif filename
#         # gdf_with_numbers = gdf_with_numbers.assign(filename=filename)
#         _, basename = os.path.split(filename)
#         id_map_sheet = re.search('[0-9]***REMOVED***2***REMOVED***', basename).group()
#         tif_filename_list = glob(os.path.abspath(os.path.join(geotif_directory, '****REMOVED******REMOVED****.tif*'.format(id_map_sheet))))
#         if len(tif_filename_list) != 1:
#             raise ValueError('There are more than 1 item with id ***REMOVED******REMOVED***'.format(id_map_sheet))
#         else:
#             tif_filename = tif_filename_list[0]
#         gdf_with_numbers = gdf_with_numbers.assign(image_filename=os.path.basename(tif_filename))
#
#         if remove_geographical_offset:
#             # Transform georeferenced coordinates to image-relative coordinates
#             x0, dx, _, y0, _, dy = get_georef_data(tif_filename)
#             gdf_with_numbers['geometry'] = gdf_with_numbers.geometry.apply(lambda s: remove_geo_offset(s, x0, dx, y0, dy))
#
#         # Add GeoDataframe to list of GeoDataframes to concatenate
#         geodataframes_to_concatenate.append(gdf_with_numbers)
#
#     if len(geodataframes_to_concatenate) > 1:
#         # Make a concatenated GeoDataframe with all the data
#         return gpd.GeoDataFrame(pd.concat(geodataframes_to_concatenate, ignore_index=True))
#     elif len(geodataframes_to_concatenate) == 1:
#         return geodataframes_to_concatenate[0]
#     else:
#         raise NotImplementedError
#
