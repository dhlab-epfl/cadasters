#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from typing import List
import numpy as np
import geopandas as gpd
import pandas as pd
import click
from geojson_processing.merging import concat_geojson_files
from geojson_processing.filtering import filter_by_area, remove_empty_transcripts, clean_manually_annotated_parcels
from geojson_processing.transcriptions import find_transcriptions_outliers
from geojson_processing.evaluation import get_correspondencies, iou_get_recall, iou_get_precision, transcription_get_recall, transcription_get_precision
from geojson_processing.evaluation import transcriptions_without_outliers_get_precision, transcriptions_without_outliers_get_recall

@click.command
@click.argument("auto_geojson_files", nargs=-1, help="Geojson files produced by the automatic extraction")
@click.option("--geotif_directory", help="Directory where the geoTIF images are stored")
@click.option("--manually_annotated_geometries_csv",
              help="csv file of the manually annotated geometries (and associatedtranscriptions)")
@click.option("--output_txt_file", "Filename of the .txt file to output")
@click.option("--iou_threshold", help="IoU threshold for the precision/recall evaluation of geometries")
def evaluate(auto_geojson_files: List[str],
             geotif_directory: str,
             manually_annotated_geometries_csv: str,
             output_txt_file: str,
             iou_threshold: float=0.9):

    # -- Merge and convert float to int transcriptions
    gdf_global = concat_geojson_files(auto_geojson_files, geotif_directory)

    print("Total number of extracted geometries: ", len(gdf_global))

    # # remove emtpy transcriptions
    # if remove_empty_transcriptions:
    #     gdf_global = remove_empty_transcripts(gdf_global)

    # convert transcription to int
    gdf_global.best_transcription = gdf_global.best_transcription.apply(lambda t: int(t) if t else np.nan)

    # -- Filter auto extracted polygons
    # filter by area
    min_area = 2.
    max_area = 15000.
    gdf_global_filtered = filter_by_area(gdf_global, min_area=min_area, max_area=max_area)

    gdf_global_filtered.crs = {'init': 'epsg:3004'}

    print("Total number of filtered geometries: ", len(gdf_global_filtered))

    # -- Neighbor transcription filtering

    # take only those who have a transcription
    gdf_neighbors = gdf_global_filtered[~gdf_global_filtered.best_transcription.isna()].copy()
    gdf_neighbors = find_transcriptions_outliers(gdf_neighbors, n_neighbors=6, tolerance=10, return_median=True)

    print("Percentage of outliers: ", len(gdf_neighbors[gdf_neighbors.is_outlier]) / len(gdf_neighbors))

    # assign outlier status to general gdf_global_filtered
    gdf_global_filtered['is_outlier'] = gpd.Series()
    gdf_global_filtered.loc[gdf_neighbors.index.values, 'is_outlier'] = gdf_neighbors.is_outlier

    # -- Manually annotated data cleaning
    gdf_manually_annotated = pd.read_csv(manually_annotated_geometries_csv)
    gdf_manually_annotated = clean_manually_annotated_parcels(gdf_manually_annotated)

    print("Total annotated geometries: ", len(gdf_manually_annotated))

    # -- Get correspondences between manually annotated and automatically extracted

    gdf_correspondances = get_correspondencies(gdf_global_filtered, gdf_manually_annotated, n_neighbors=4)

    # -- Evaluation geometries
    geometries_precision, correct_geometries_count = iou_get_precision(gdf_correspondances,
                                                                       iou_threshold=iou_threshold, return_number=True)
    geometries_recall = iou_get_recall(gdf_correspondances, gdf_manually_annotated,
                                       iou_threshold=iou_threshold, return_number=False)

    print('-- Geometries evaluation - P: {}, R: {}. {}'.format(geometries_precision, geometries_recall,
                                                               correct_geometries_count))

    # -- Evaluation transcriptions
    transcriptions_precision, correct_transcriptions_count = transcription_get_precision(gdf_correspondances,
                                                                                         return_number=True)
    transcriptions_recall = transcription_get_recall(gdf_correspondances, gdf_manually_annotated, return_number=False)

    print('-- Transcriptions evaluation - P: {}, R: {}. {}'.format(transcriptions_precision, transcriptions_recall,
                                                                   correct_transcriptions_count))

    # # -- Evaluation transcriptions after outlier detection
    # transcriptions_without_outliers_get_precision(gdf_correspondances, return_number=True)
    # transcriptions_without_outliers_get_recall(gdf_correspondances, gdf_manually_annotated, return_number=True)

    with open(output_txt_file, "w") as text_file:
        print("Total number of filtered geometries: ", len(gdf_global_filtered), file=text_file)
        print("Total number of filtered geometries: ", len(gdf_global_filtered), file=text_file)
        print("Total annotated geometries: ", len(gdf_manually_annotated), file=text_file)

        print('-- Geometries evaluation - P: {}, R: {}. {}'.format(geometries_precision, geometries_recall,
                                                                   correct_geometries_count), file=text_file)
        print('-- Transcriptions evaluation - P: {}, R: {}. {}'.format(transcriptions_precision, transcriptions_recall,
                                                                       correct_transcriptions_count), file=text_file)




