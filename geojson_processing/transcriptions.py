***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union
***REMOVED***


def find_transcriptions_outliers(geodataframe_with_transcriptions: Union[pd.DataFrame, gpd.GeoDataFrame],
                                 n_neighbors: int=6, tolerance: int=10, return_median: bool=False):
***REMOVED***"

    :param geodataframe_with_transcriptions: dataframe that contains polygons with transcriptions
    :param n_neighbors: number of neighbors to consider for the nearest neighbor algorithm
    :param tolerance: the polygon must have transcription values within range [x-tol; x+tol] according to the
    median of the nearest neighbors
    :param return_median: add a column ``median_transcription`` to the returned dataframe
    :return: the updated geodataframe with True/False values in ``is_outlier`` column
***REMOVED***"

    geodataframe_with_transcriptions['is_outlier'] = pd.Series()
    if return_median:
        geodataframe_with_transcriptions['median_transcription'] = pd.Series()

    geodataframe_with_transcriptions['centroid'] = \
        geodataframe_with_transcriptions.geometry.apply(lambda s: s.centroid)

    nneigh = NearestNeighbors(n_neighbors=n_neighbors, radius=50.0)

    # create a array with cent_x, cent_y
    centroids = np.stack([geodataframe_with_transcriptions.centroid.x.values,
                          geodataframe_with_transcriptions.centroid.y.values],
                         axis=1)
    nneigh.fit(centroids)

    for i in tqdm(range(len(centroids))):
        centroid_point = centroids[i]
        # the first distance/index is 0/i because it's the `centroid_point`
        distances, indexes_neighboring_centroids = nneigh.kneighbors([centroid_point])

        df_nns = geodataframe_with_transcriptions.loc[geodataframe_with_transcriptions.index[indexes_neighboring_centroids].values[0]]
        # difference between the neighboring transcriptions and the current trnscription
        median_diff = abs(np.median(df_nns[1:].best_transcription - df_nns.iloc[0].best_transcription))

        if return_median:
            median_transcription = np.median(df_nns[1:].best_transcription)
            geodataframe_with_transcriptions.loc[df_nns.iloc[0].name, 'median_transcription'] = median_transcription

        if median_diff <= tolerance:
            geodataframe_with_transcriptions.loc[df_nns.iloc[0].name, 'is_outlier'] = False
        else:
            geodataframe_with_transcriptions.loc[df_nns.iloc[0].name, 'is_outlier'] = True

    return geodataframe_with_transcriptions
