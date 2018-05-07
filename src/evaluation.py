***REMOVED***
***REMOVED***

from scipy.misc import imread
import numpy as np
import cv2
from collections import Counter
***REMOVED***
import datetime
import json
from typing import List
from .utils import MyPolygon, crop_with_margin
***REMOVED***
import Levenshtein


class ResultsLocalization:

    def __init__(self, **kwargs):
        self.true_positive = kwargs.get('true_positives', 0)
        # self.false_positive = kwargs.get('false_positives', 0)
        self.total_groundtruth = kwargs.get('total_truth')
        self.total_predicted = kwargs.get('total_predicted', 0)
        self.threshold = kwargs.get('thresh')
        self.use_iou = kwargs.get('iou', False)
        self.recall = kwargs.get('recall')
        self.precision = kwargs.get('precision')

    def compute_metrics(self):
        assert (self.true_positive and self.total_groundtruth), "True and False positives not initialized"

        self.recall = self.true_positive/self.total_groundtruth
        self.precision = self.true_positive/self.total_predicted


class ResultsTranscription:

    def __init__(self, **kwargs):
        self.correct_transcription = kwargs.get('true_positives', 0)
        # self.incorrect_transcription = kwargs.get('false_positives', 0)
        self.total_groundtruth = kwargs.get('total_truth')
        self.total_predicted = kwargs.get('total_predicted', 0)
        # self.partial_transcription = kwargs.get('partial_recognition')
        self.levenshtein_distance = kwargs.get('levenshtein_distance', 0)
        self.recall = kwargs.get('recall')
        self.precision = kwargs.get('precision')
        self.total_chars = kwargs.get('total_chars', 0)
        self.cer = kwargs.get('cer')

    def compute_metrics(self):
        assert (self.total_groundtruth and self.total_chars), \
            "True and False positives not initialized or partial_recognition is not initialized"

        self.recall = self.correct_transcription / self.total_groundtruth
        self.precision = self.correct_transcription / self.total_predicted

        # CER
        self.cer = self.levenshtein_distance / self.total_chars


class TranscriptionStats:
    def __init__(self, **kwargs):
        self.prediction = kwargs.get('prediction')
        self.groundtruth = kwargs.get('groundtruth')
        self.confidence = kwargs.get('confidence')
        self.total_chars = kwargs.get('total_chars')

        if self.groundtruth is not None:
            self.correctness = self._compute_correctness()
            # if not self.correctness:
            self.error_type, self.distance = self._compute_error()
            # else:
            #     self.error_type = None
            #     self.distance = 0
        else:
            self.correctness = False
            self.error_type = LabelErrorType.NOISE
            self.distance = 0

    def _compute_correctness(self) -> bool:
        return self.prediction == self.groundtruth

    def _compute_error(self) -> (str, float):
        groundtruth_str = str(self.groundtruth)
        prediction_str = str(self.prediction)
        if groundtruth_str == prediction_str:
            return None, 0
        else:
            if len(groundtruth_str) == len(prediction_str):
                error_type = LabelErrorType.SUBSTITUTION
                distance = Levenshtein.distance(groundtruth_str, prediction_str)
            elif len(groundtruth_str) > len(prediction_str):
                error_type = LabelErrorType.DELETION
                distance = Levenshtein.distance(groundtruth_str, prediction_str)
            elif len(groundtruth_str) < len(prediction_str):
                error_type = LabelErrorType.INSERTION
                distance = Levenshtein.distance(groundtruth_str, prediction_str)
            else:
                raise NotImplementedError

            return error_type, distance


class TranscriptionAnalysis:
    def __init__(self, **kwargs):
        self.list_stats = kwargs.get('stats')
        self.correct_transcriptions, self.incorrect_transcriptions = self._separate_correct_from_incorrect()
        self.insertions, self.deletions, self.substitutions, \
            self.noise, self.distance_count = self._compute_error_rates()

    def _separate_correct_from_incorrect(self) -> (list, list):
        correct_transcriptions_list, incorrect_transcriptions_list = list(), list()
        for transcription_stat in self.list_stats:
            # Get correct boxes
            if transcription_stat.correctness:
                correct_transcriptions_list.append(transcription_stat)
            else:
                incorrect_transcriptions_list.append(transcription_stat)

        return correct_transcriptions_list, incorrect_transcriptions_list

    def _compute_error_rates(self) -> (float, float, float, dict):
        insertions, deletions, substitutions, noise = 0, 0, 0, 0
        distances = list()
        for transcription in self.incorrect_transcriptions:
            type_error = transcription.error_type
            distances.append(transcription.distance)
            if type_error == LabelErrorType.INSERTION:
                insertions += 1
            elif type_error == LabelErrorType.DELETION:
                deletions += 1
            elif type_error == LabelErrorType.SUBSTITUTION:
                substitutions += 1
            elif type_error == LabelErrorType.NOISE:
                noise += 1

        # insertion_rate = insertion / len(self.incorrect_transcriptions)
        # deletion_rate = deletion / len(self.incorrect_transcriptions)
        # substitution_rate = substitution / len(self.incorrect_transcriptions)
        # noise_rate = noise / len(self.incorrect_transcriptions)
        distances_count = Counter(distances)

        return insertions, deletions, substitutions, noise, distances_count


class LabelErrorType:
    DELETION = 'DELETION'
    SUBSTITUTION = 'SUBSTITUTION'
    INSERTION = 'INSERTION'
    NOISE = 'NOISE'


def make_mask_pointPolyTest(mask_shape, contours, distance=False):
***REMOVED***"
    Computes the mask indicating if a pixel is within the polygon given by contours
    :param mask_shape: shape of the mask to compute
    :param contours: the contours of the polygons
    :param distance: to also compute the distance between a point and the closest contour (default=False)
    :return: mask
***REMOVED***"
    mask = np.zeros(mask_shape[:2], dtype=bool)
    xx, yy = np.mgrid[0:mask_shape[1], 0:mask_shape[0]]
    list_points = [tuple([xx.flatten()[i], yy.flatten()[i]]) for i in range(len(xx.flatten()))]

    for pt in list_points:
        val = cv2.pointPolygonTest(contours, pt, distance)
        if val > 0:
            mask[pt[1], pt[0]] = val

    return mask


# def minimum_edit_distance(s1, s2):
# ***REMOVED***"
#     Computes the Levenshtein distence between 2 strings s1, s2
#     Taken from https://rosettacode.org/wiki/Levenshtein_distance#Python
#     :param s1:
#     :param s2:
#     :return: Levenshtein distance
# ***REMOVED***"
#     if len(s1) > len(s2):
#         s1, s2 = s2, s1
#     distances = range(len(s1) + 1)
#     for index2, char2 in enumerate(s2):
#         newDistances = [index2+1]
#         for index1, char1 in enumerate(s1):
#             if char1 == char2:
#                 newDistances.append(distances[index1])
#             else:
#                 newDistances.append(1 + min((distances[index1],
#                                              distances[index1+1],
#                                              newDistances[-1])))
#         distances = newDistances
#     return distances[-1]


def get_labelled_digits_matrix(groundtruth_digits_filename: str) -> np.array:
***REMOVED***"
        From the RGB image (h x w x 3), reconstruct the matrix of labels of the digits for evaluation
        :param groundtruth_digits_filename: filename of labelled digit image
        :return: labels_matrix : matrix  of size (h x w) and type int of the digit labels
    ***REMOVED***"
    assert os.path.isfile(groundtruth_digits_filename), \
        'Groundtruth file not found at ***REMOVED******REMOVED***'.format(groundtruth_digits_filename)
    # Load image
    img_digit_labels = imread(groundtruth_digits_filename)
    img_digit_labels_bin = np.uint8(255 * (imread(groundtruth_digits_filename, mode='L') > 0))

    # Find contours
    _, contours, _ = cv2.findContours(np.uint8(255 * (imread(groundtruth_digits_filename, mode='L') > 0)),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels_matrix = np.zeros(img_digit_labels_bin.shape, dtype=int)
    for c in contours:
        pt = tuple(c[0][0])  # + tuple([2, 2])
        r_ch = img_digit_labels[pt[1], pt[0], 0]
        g_ch = img_digit_labels[pt[1], pt[0], 1]
        b_ch = img_digit_labels[pt[1], pt[0], 2]
        if g_ch == 0 or b_ch == 0:
            pt = tuple(c[0][0]) + tuple([-2, 2])
            r_ch = img_digit_labels[pt[1], pt[0], 0]
            g_ch = img_digit_labels[pt[1], pt[0], 1]
            b_ch = img_digit_labels[pt[1], pt[0], 2]

        # number = r_ch*256*256 + g_ch*256 + b_ch
        number = g_ch * 256 + b_ch

        # Replace the 3 channels RGB by one with float value
        col, row, width, height = cv2.boundingRect(c)
        mask = make_mask_pointPolyTest([height, width], c - [col, row])

        crop_label = labels_matrix[row:row + height, col:col + width]
        crop_label[mask] = number

    return labels_matrix


def get_labelled_parcels_matrix(groundtruth_parcels_filename: str) -> np.array:
    assert os.path.isfile(groundtruth_parcels_filename), \
        'Groundtruth file not found at ***REMOVED******REMOVED***'.format(groundtruth_parcels_filename)
    # Open image and give a unique label to each parcel
    image_parcels_gt = cv2.imread(groundtruth_parcels_filename, cv2.IMREAD_GRAYSCALE)
    image_parcels_gt = np.uint8(image_parcels_gt > 128) * 255
    n_labels_poly, parcels_labeled = cv2.connectedComponents(image_parcels_gt)

    return parcels_labeled


def evaluation_parcel_iou(groundtruth_parcel_image, polygon_coordinates, iou_thresh=0.8):
    extracted_poly = select_number_label_by_countour(groundtruth_parcel_image, [polygon_coordinates], erode=True)
    if len(np.unique(extracted_poly)) < 2:
        return False

    # Count which is the label that appears the most and consider that it is the label of the parcel
    label_poly = Counter(groundtruth_parcel_image[extracted_poly > 0]).most_common(1)[0][0]
    if label_poly == 0:
        return False

    gt_poly = np.uint8(groundtruth_parcel_image == label_poly) * 255
    intersection = cv2.bitwise_and(extracted_poly, gt_poly)
    union = cv2.bitwise_or(extracted_poly, gt_poly)
    IoU = np.sum(intersection.flatten()) / np.sum(union.flatten())

    if IoU >= iou_thresh:
        return True
    else:
        return False


def select_number_label_by_countour(numbers_image: np.array, contours: np.array, erode: bool=False) -> np.array:
    extracted_label = np.zeros(numbers_image.shape[:2], dtype='uint8')
    cv2.fillPoly(extracted_label, contours, 255)
    if erode:
        extracted_label = cv2.erode(extracted_label, np.ones((5, 5), np.uint8))
    return extracted_label


def evaluation_number_transcription(groundtruth_numbers_image: np.array, selected_label_mask: np.array,
                                    transcription: str, score_prediction: float=None):
    # Count which is the label that appears the most and consider that it is the label of the parcel
    correct_label = Counter(groundtruth_numbers_image[selected_label_mask > 0]).most_common(1)[0][0]
    correct_label = str(correct_label)
    if correct_label == '0':  # background
        transcription_stats = TranscriptionStats(groundtruth=None,
                                                 prediction=transcription,
                                                 confidence=score_prediction,
                                                 total_chars=0)
    else:
        transcription_stats = TranscriptionStats(groundtruth=correct_label,
                                                 prediction=transcription,
                                                 confidence=score_prediction,
                                                 total_chars=len(str(correct_label)))

    # if correct_label == '0':  # background
    #     result = False
    #     levenshtein_distance = 0 #len(correct_label)
    #     total_chars = 0
    # elif correct_label == transcription:
    #     result = True
    #     levenshtein_distance = 0
    #     total_chars = len(str(correct_label))
    # else:
    #     result = False
    #     levenshtein_distance = minimum_edit_distance(correct_label, transcription)
    #     total_chars = len(str(correct_label))

    # return result, levenshtein_distance, total_chars
    return transcription_stats


def evaluation_number_localization(groundtruth_numbers_image: np.array, selected_number_mask: np.array,
                                   threshold: float=0.8, use_iou: bool=False) -> bool:
    most_comon_label = Counter(groundtruth_numbers_image[selected_number_mask > 0]).most_common(1)
    correct_label = most_comon_label[0][0]

    if correct_label == 0:
        try:
            correct_label = most_comon_label[1][0]
            if correct_label == 0:
                return False
        except IndexError:
            return False

    # Compute intersection over union (IoU)
    groundtruth_number = np.uint8(groundtruth_numbers_image == correct_label) * 255
    intersection = cv2.bitwise_and(selected_number_mask, groundtruth_number)

    if use_iou:
        union = cv2.bitwise_or(selected_number_mask, groundtruth_number)
        IoU = np.sum(intersection.flatten(), dtype=float) / np.sum(union.flatten(), dtype=float)
        measure = IoU
    else:
        measure = np.sum(intersection.flatten(), dtype=float) / np.sum(groundtruth_number.flatten(), dtype=float)

    if measure >= threshold:
        return True
    else:
        return False


def evaluation_json_file(json_filename: str, **kwargs) -> None:

    json_dict = dict()
    results_parcels = kwargs.get('results_parcels')
    results_numbers_tuple = kwargs.get('result_numbers')

    date = datetime.datetime.now()

    json_dict['creation_date'] = '***REMOVED***:02d***REMOVED***.***REMOVED***:02d***REMOVED***.***REMOVED***:02d***REMOVED*** at ***REMOVED***:02d***REMOVED***:***REMOVED***:02d***REMOVED***'\
        .format(date.day, date.month, date.year, date.hour, date.minute)
    if results_parcels:
        json_dict['evaluation_parcels'] = vars(results_parcels)

    if results_numbers_tuple:
        result_localization, result_recognition = results_numbers_tuple
        json_dict['evaluation_digits'] = ***REMOVED***'localization': vars(result_localization),
                                          'recognition': vars(result_recognition)***REMOVED***

    with open(json_filename, 'w') as outfile:
        json.dump(json_dict, outfile)


def evaluate(polygons_list: List[MyPolygon], groundtruth_parcels, groundtruth_numbers, threshold_parcels: float,
             threshold_labels: float) -> (ResultsLocalization, ResultsLocalization, ResultsTranscription):
    result_parcel_localisation = ResultsLocalization(thresh=threshold_parcels, iou=True,
                                                     total_truth=len(np.unique(groundtruth_parcels)) - 1)
    result_label_localisation = ResultsLocalization(thresh=threshold_labels, iou=False,
                                                    total_truth=len(np.unique(groundtruth_numbers)) - 1)
    result_transcription = ResultsTranscription(total_truth=len(np.unique(groundtruth_numbers)) - 1)
    
    transcriptions_stats_list = list()

    for polygon in tqdm(polygons_list, total=len(polygons_list)):
        parcel_groundtruth_crop, \
        (x, y, w, h) = crop_with_margin(groundtruth_parcels,
                                        cv2.boundingRect(polygon.approximate_coordinates(epsilon=1)),
                                        margin=30, return_coords=True)
        # EVALUATION PARCEL
        parcel_evaluated = evaluation_parcel_iou(parcel_groundtruth_crop,
                                                 (polygon.contours[0][:, 0, :] - [x, y])[:, None, :],
                                                 iou_thresh=threshold_parcels)
        result_parcel_localisation.true_positive += parcel_evaluated
        result_parcel_localisation.total_predicted += 1

        # EVALUATION LABELS
        numbers_groundtruth_crop = crop_with_margin(groundtruth_numbers, (x, y, w, h), margin=0)
        if polygon.transcription == [] or polygon.label_contours == []:
            continue
        for transcription, contour in zip(polygon.transcription, polygon.label_contours):
            if contour is []:
                continue
            contour = (contour[:, 0, :] - [x, y])[:, None, :]
            selected_label_mask = select_number_label_by_countour(numbers_groundtruth_crop, [contour])
            # Evaluation transcription
            # transcription_evaluated, levenshtein_distance, total_chars \
            #     = evaluation_number_transcription(numbers_groundtruth_crop,
            #                                       selected_label_mask,
            #                                       transcription)
            transcription_stats = evaluation_number_transcription(numbers_groundtruth_crop,
                                                                  selected_label_mask,
                                                                  transcription)
            result_transcription.correct_transcription += transcription_stats.correctness
            result_transcription.total_predicted += 1
            result_transcription.levenshtein_distance += transcription_stats.distance
            result_transcription.total_chars += transcription_stats.total_chars

            transcriptions_stats_list.append(transcription_stats)

            # Evaluation label localization
            number_localization_evaluated = evaluation_number_localization(numbers_groundtruth_crop,
                                                                           selected_label_mask,
                                                                           threshold=threshold_labels,
                                                                           use_iou=False)

            result_label_localisation.true_positive += number_localization_evaluated
            result_label_localisation.total_predicted += 1

    result_label_localisation.compute_metrics()
    result_parcel_localisation.compute_metrics()
    result_transcription.compute_metrics()
    transcription_analysis = TranscriptionAnalysis(stats=transcriptions_stats_list)

    return result_parcel_localisation, result_label_localisation, result_transcription, transcription_analysis

