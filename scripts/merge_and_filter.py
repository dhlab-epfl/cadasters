***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from typing import List
import numpy as np
import click
from geojson_processing.merging import concat_geojson_files
from geojson_processing.filtering import filter_by_area, remove_empty_transcripts


@click.command
@click.argument('geojson_files', nargs=-1)
@click.option('--geotif_directory', help='Directory containing the geotifs of the images')
@click.option('--exported_merged_geojson_filename', help='Filename (geoJSON) to save the merged and filtered geodataframe')
@click.option('--remove_empty_transcriptions', help="Wether to remove or not rows that have no transcription", default=False)
def merge_and_filter_geojson(geojson_files: List[str],
                             geotif_directory: str,
                             exported_merged_geojson_filename: str,
                             remove_empty_transcriptions: bool=False):
    gdf_global = concat_geojson_files(geojson_files, geotif_directory)

    # remove emtpy transcriptions
    if remove_empty_transcriptions:
        gdf_global = remove_empty_transcripts(gdf_global)

    # convert transcription to int
    gdf_global.best_transcription = gdf_global.best_transcription.apply(lambda t: int(t) if t else np.nan)

    # filter by area
    min_area = 2.
    max_area = 15000.
    gdf_global_filtered = filter_by_area(gdf_global, min_area=min_area, max_area=max_area)

    gdf_global_filtered.crs = ***REMOVED***'init': 'epsg:3004'***REMOVED***  # epsg:3004 : Monte Mario
    gdf_global_filtered.to_file(exported_merged_geojson_filename, driver='GeoJSON', encoding='utf8')
