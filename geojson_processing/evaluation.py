***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from typing import Union
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
import numpy as np
***REMOVED***


def iou_get_precision(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                      iou_threshold: float=0.8, return_number: bool=False):

    n_correct = len(dataframe_correspondencies[dataframe_correspondencies.iou > iou_threshold])
    if return_number:
        return n_correct / len(dataframe_correspondencies), n_correct
    else:
        return n_correct / len(dataframe_correspondencies)


def iou_get_recall(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                   dataframe_gt: Union[pd.DataFrame, gpd.GeoDataFrame],
                   iou_threshold: float=0.8, return_number: bool=False):
    n_correct = len(dataframe_correspondencies[dataframe_correspondencies.iou > iou_threshold])
    if return_number:
        return n_correct / len(dataframe_gt), n_correct
    else:
        return n_correct / len(dataframe_gt)


def transcription_get_precision(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                                return_number: bool = False):

    # Count also the nan prediction when there wasn't any transcription
    #correct_nan = sum(dataframe_correspondencies[dataframe_correspondencies.manual_ID == -1].best_transcription.isna())
    #print(correct_nan)
    df_has_transcription = dataframe_correspondencies[~dataframe_correspondencies.best_transcription.isna()]

    correct_transcript = sum(df_has_transcription.best_transcription == df_has_transcription.manual_ID)

    if return_number:
        # return (correct_transcript + correct_nan) / len(df_has_transcription), (correct_transcript + correct_nan)
        return correct_transcript / len(df_has_transcription), correct_transcript
    else:
        # return (correct_transcript + correct_nan) / len(df_has_transcription)
        return correct_transcript/ len(df_has_transcription)


def transcription_get_recall(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                             dataframe_gt: Union[pd.DataFrame, gpd.GeoDataFrame],
                             return_number: bool = False):

    # Count also the nan prediction when there wasn't any transcription
    # correct_nan = sum(dataframe_correspondencies[dataframe_correspondencies.manual_ID == -1].best_transcription.isna())

    correct_transcript = sum(dataframe_correspondencies.manual_ID == dataframe_correspondencies.best_transcription)
    gt_has_transcription = dataframe_gt[~dataframe_gt.ID.isna()]

    if return_number:
        # return (correct_transcript + correct_nan) / len(dataframe_gt), (correct_transcript + correct_nan)
        return correct_transcript / len(gt_has_transcription), correct_transcript
    else:
        # return (correct_transcript + correct_nan) / len(dataframe_gt)
        return correct_transcript / len(gt_has_transcription)


def outlier_get_precision(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame], return_number: bool=False):

    not_outlier_transcription = dataframe_correspondencies[dataframe_correspondencies.is_outlier == False]

    correct_in_insiders = len(not_outlier_transcription[
                                  not_outlier_transcription.manual_ID == not_outlier_transcription.best_transcription
                               ***REMOVED***)
    print('Not outliers: ', len(not_outlier_transcription))
    if return_number:
        return correct_in_insiders / len(not_outlier_transcription), correct_in_insiders
    else:
        return correct_in_insiders / len(not_outlier_transcription)


def outlier_get_recall(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                       return_number: bool=False):

    # Count also the nan prediction when there wasn't any transcription
    has_correct_transcription = \
        dataframe_correspondencies[dataframe_correspondencies.manual_ID == dataframe_correspondencies.best_transcription]

    print('Correct_transcripitons', len(has_correct_transcription))
    if return_number:
        return sum(has_correct_transcription.is_outlier == False) / len(has_correct_transcription), \
               sum(has_correct_transcription.is_outlier == False)
    else:
        return sum(has_correct_transcription.is_outlier == False) / len(has_correct_transcription)


def transcriptions_without_outliers_get_precision(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                                                  return_number: bool = False):

    not_outlier_transcription = dataframe_correspondencies[dataframe_correspondencies.is_outlier == False]

    correct_trans = len(not_outlier_transcription[
                            not_outlier_transcription.manual_ID == not_outlier_transcription.best_transcription
                         ***REMOVED***)

    if return_number:
        return correct_trans / len(not_outlier_transcription), correct_trans
    else:
        return correct_trans / len(not_outlier_transcription)


def transcriptions_without_outliers_get_recall(dataframe_correspondencies: Union[pd.DataFrame, gpd.GeoDataFrame],
                                               dataframe_gt: Union[pd.DataFrame, gpd.GeoDataFrame],
                                               return_number: bool = False):
    gt_has_transcription = dataframe_gt[~dataframe_gt.ID.isna()]

    has_correct_transcription = \
        dataframe_correspondencies[
            dataframe_correspondencies.manual_ID == dataframe_correspondencies.best_transcription]

    if return_number:
        return sum(has_correct_transcription.is_outlier == False) / len(gt_has_transcription), \
               sum(has_correct_transcription.is_outlier == False)
    else:
        sum(has_correct_transcription.is_outlier == False) / len(gt_has_transcription)


def get_correspondencies(dataframe_shapes_automatically_produced: Union[pd.DataFrame, gpd.GeoDataFrame],
                         dataframe_shapes_mannually_produced: Union[pd.DataFrame, gpd.GeoDataFrame],
                         n_neighbors:int=3) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
***REMOVED***"


    :param dataframe_shapes_automatically_produced: must contain columns `uuid`, `best_transcription`, `geometry`,
    `image_filename`, `is_outlier`
    :param dataframe_shapes_mannually_produced: must_contain columns `ID`, `uuid`, `geometry`, `centroid`
    :return: a dataframe of the sime size as `dataframe_shapes_automatically_produced` with the corresponding shapes
    of the `dataframe_shapes_mannually_produced`
***REMOVED***"
    #  Need to create correspondencies between manually corrected and extracted
    df_correspondances = dataframe_shapes_automatically_produced.copy()
    columns_to_add = ['iou', 'manual_geom', 'manual_ID', 'manual_uuid']
    for lab in columns_to_add:
        df_correspondances[lab] = pd.Series(np.nan)

    nearestneigh_ncorresp = NearestNeighbors(n_neighbors=n_neighbors, radius=100.0)

    # create a array with cent_x, cent_y
    centroids_auto = np.stack([df_correspondances.centroid.apply(lambda c: c.x).values,
                               df_correspondances.centroid.apply(lambda c: c.y).values],
                              axis=1)
    nearestneigh_ncorresp.fit(centroids_auto)

    # Compute also the centroid for manually annotated data
    centroids_manual = np.stack([dataframe_shapes_mannually_produced.centroid.apply(lambda c: c.x).values,
                                 dataframe_shapes_mannually_produced.centroid.apply(lambda c: c.y).values],
                                axis=1)

    # Find correpondencies
    already_existing = pd.DataFrame()
    for i in tqdm(range(len(centroids_manual))):

        centroid_point = centroids_manual[i]
        # Find which are the closest neighbors
        distances, indexes_neighboring_centroids = nearestneigh_ncorresp.kneighbors([centroid_point])

        # Find 3 closest shapes in the auto processed shapes
        df_nns = df_correspondances.loc[df_correspondances.index[indexes_neighboring_centroids].values[0]]

        row_manual = dataframe_shapes_mannually_produced.iloc[i]
        nearest_ious = df_nns.geometry.apply(
            lambda s: s.intersection(row_manual.geometry).area / s.union(row_manual.geometry).area).sort_values(
            ascending=False)

        # TODO verify that there are no overlapping shapes in layer
        iou = nearest_ious.iloc[0]
        index = nearest_ious.index[0]

        if df_correspondances.loc[index, 'iou'] > iou:
            # print('iou exists already')
            already_existing.append(row_manual)
            continue

        df_correspondances.loc[index, 'iou'] = iou
        # -1 to indicate there was no ID
        df_correspondances.loc[index, 'manual_ID'] = int(row_manual.ID) if not np.isnan(row_manual.ID) else -1
        df_correspondances.loc[index, 'manual_geom'] = row_manual.geometry

        if iou > 0:
            df_correspondances.loc[index, 'manual_uuid'] = row_manual.uuid
        else:
            df_correspondances.loc[index, 'manual_uuid'] = -1

    return df_correspondances
