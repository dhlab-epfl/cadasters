#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from geojson_processing.transcriptions import find_transcriptions_outliers
import geopandas as gpd
import click


@click.command
@click.argument("geopandas_filename")
@click.option("--export_filename")
@click.option("--n_neighbors", default=6, help="Number of neighbors (n+1)")
@click.option("--diff_tolerance", default=10, help="Difference value between neighboring transcriptions")
def neighbor_transcription_filtering(geopandas_filename: str,
                                     export_filename: str,
                                     n_neighbors: int=6,
                                     diff_tolerance: int=10):
    gdf_filtered = gpd.read_file(geopandas_filename, encoding='utf8')

    # take only elements that have a transcription
    gdf_neighbors = gdf_filtered[~gdf_filtered.best_transcription.isna()].copy()

    gdf_neighbors = find_transcriptions_outliers(gdf_neighbors, n_neighbors=n_neighbors,
                                                 tolerance=diff_tolerance, return_median=True)

    print("Percentage of outliers:", len(gdf_neighbors[gdf_neighbors.is_outlier]) / len(gdf_neighbors))

    # assign outlier status to general gdf_global_filtered
    gdf_filtered['is_outlier'] = gpd.Series()
    gdf_filtered.loc[gdf_neighbors.index.values, 'is_outlier'] = gdf_neighbors.is_outlier

    gdf_filtered.to_file(export_filename, driver='GeoJSON', encoding='utf8')
