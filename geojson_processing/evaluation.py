***REMOVED***
__author__ = "solivr"
__license__ = "GPL"

from typing import Union
import pandas as pd
import geopandas as gpd


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