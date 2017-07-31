import cv2
import numpy as np
from collections import Counter
from scipy import misc
from helpers import minimum_edit_distance, count_correct_characters, \
    ResultsLocalization, ResultsRecognition, BoxLabelPrediction


def make_mask_pointPolyTest(mask_shape, contours, distance=False):
    """
    Computes the mask indicating if a pixel is within the polygon given by contours
    :param mask_shape: shape of the mask to compute
    :param contours: the contours of the polygons
    :param distance: to also compute the distance between a point and the closest contour (default=False)
    :return: mask
    """
    mask = np.zeros(mask_shape[:2], dtype=bool)
    xx, yy = np.mgrid[0:mask_shape[1], 0:mask_shape[0]]
    list_points = [tuple([xx.flatten()[i], yy.flatten()[i]]) for i in range(len(xx.flatten()))]

    for pt in list_points:
        val = cv2.pointPolygonTest(contours, pt, distance)
        if val > 0:
            mask[pt[1], pt[0]] = val

    return mask
# ------------------------------------------------------------------


def print_digit_counts(counts_digits) -> None:

    total_counts = sum(np.array([counts_digits[i] for i in counts_digits.keys()]))

    str_to_print = ''
    for i in sorted(counts_digits.keys(), reverse=True):
        str_to_print += '\t{} digit(s) : {}/{} ({:.02f})\n'.format(i, counts_digits[i], total_counts,
                                                                   counts_digits[i] / total_counts)
    return str_to_print

# ------------------------------------------------------------------


def get_labelled_digits_matrix(filename_digits_labelled: str) -> np.array:
    """
    From the RGB image (h x w x 3), reconstruct the matrix of labels of the digits for evaluation
    :param filename_digits_labelled: filename of labelled digit image
    :return: labels_matrix : matrix  of size (h x w) and type int of the digit labels
    """

    # Load image
    img_digit_lbl = misc.imread(filename_digits_labelled)
    img_digit_lbl_bin = np.uint8(255*(misc.imread(filename_digits_labelled, mode='L') > 0))

    # Find contours
    _, contours, _ = cv2.findContours(img_digit_lbl_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels_matrix = np.zeros(img_digit_lbl_bin.shape, dtype=int)
    for c in contours:
        pt = tuple(c[0][0]) + tuple([2, 2])
        # r_ch = img_digit_lbl[pt[1], pt[0], 0]
        g_ch = img_digit_lbl[pt[1], pt[0], 1]
        b_ch = img_digit_lbl[pt[1], pt[0], 2]
        if g_ch == 0 or b_ch == 0:
            pt = tuple(c[0][0]) + tuple([-2, 2])
            # r_ch = img_digit_lbl[pt[1], pt[0], 0]
            g_ch = img_digit_lbl[pt[1], pt[0], 1]
            b_ch = img_digit_lbl[pt[1], pt[0], 2]

        # number = r_ch*256*256 + g_ch*256 + b_ch
        number = g_ch * 256 + b_ch

        # Replace the 3 channels RGB by one with float value
        col, row, width, height = cv2.boundingRect(c)
        crop = img_digit_lbl[row:row + height, col:col + width, :].copy()

        mask = make_mask_pointPolyTest(crop.shape, c - [col, row])

        crop_label = labels_matrix[row:row + height, col:col + width]
        crop_label[mask] = number

    return labels_matrix
# ---------------------------------------------------------------------


def evaluation_digit_recognition(label_matrix: np.array, list_extracted_boxes: list,
                                 result_recognition: ResultsRecognition) -> list():

    # Number of false and true positives
    n_false_positives = 0
    n_true_positives = 0
    # List with Levensthein distances, correct digits and total number of digits
    list_partial_numbers_results = list()
    # List of tuples (predicted_number, correct_number) (only for partially correct numbers)
    list_tup_predicted_correct = list()
    # List of BoxLabelPrediction for evaluation
    box_prediction_list = list()

    for box in list_extracted_boxes:
        predicted_number = box.prediction_number[0]

        mask_box = np.zeros(label_matrix.shape, dtype='uint8')
        cv2.drawContours(mask_box, [box.original_box_pts], -1, 255, thickness=cv2.FILLED)
        # Find the corresponding label in label_matrix, and discard 0 label (background)
        list_values = iter(Counter(label_matrix[mask_box != 0]).most_common(2))
        correct_label = next(list_values)[0]
        if correct_label == 0:
            try:
                correct_label = next(list_values)[0]
            except StopIteration:
                n_false_positives += 1
                continue

        # Evaluate if the extracted digit is correctly recognized
        if not predicted_number:
            list_partial_numbers_results.append([0, 0, len(str(correct_label))])
            continue
        if correct_label == int(predicted_number):
            n_true_positives += 1
        else:
            # If not, compute Levenshtein distance
            lev_dist = minimum_edit_distance(str(correct_label), str(predicted_number))

            # We consider that if lev_dist > len(str(correct_label))
            # there is no digit that can be considered as correct
            if lev_dist < len(str(correct_label)):

                # Count correct digits
                correct_digits = count_correct_characters(str(predicted_number), str(correct_label))

                # Add this result to list_result
                list_partial_numbers_results.append([lev_dist, correct_digits, len(str(correct_label))])
                list_tup_predicted_correct.append((predicted_number, correct_label))

            else:
                # 0 digits are correct
                list_partial_numbers_results.append([lev_dist, 0, len(str(correct_label))])
                list_tup_predicted_correct.append((predicted_number, correct_label))

        box_prediction_list.append(BoxLabelPrediction(prediction=predicted_number,
                                                      groundtruth=correct_label,
                                                      confidence=box.prediction_number[1],
                                                      points=box.original_box_pts))

        result_recognition.total_chars += len(predicted_number)

    result_recognition.true_positive = n_true_positives
    result_recognition.false_positive = n_false_positives
    result_recognition.partial_recognition = np.array(list_partial_numbers_results)

    return box_prediction_list

# ---------------------------------------------------------------------


def evaluation_digit_localisation(digits_groundtruth, list_boxes: list, result_localization: ResultsLocalization,
                                  thresh=0.5, iou=True)-> list:
    nb_correct_box = 0
    nb_incorrect_box = 0

    list_correct_boxes = list()

    for box in list_boxes:

        img_extracted_box = np.zeros(digits_groundtruth.shape, dtype='uint8')
        cv2.drawContours(img_extracted_box, [box.original_box_pts], -1, 255, thickness=-1)
        img_extracted_box_bin = img_extracted_box > 0

        # Count which is the label that appears the most and consider that it is the label of the parcel
        most_comon_labs = Counter(digits_groundtruth[img_extracted_box_bin]).most_common(2)
        label_box = most_comon_labs[0][0]
        if label_box == 0:
            try:
                label_box = most_comon_labs[1][0]
                if label_box == 0:
                    nb_incorrect_box += 1
                    continue
            except IndexError:
                nb_incorrect_box += 1
                continue

        # Compute intersection over union (IoU)
        gt_box = np.uint8(digits_groundtruth == label_box) * 255
        intersection = cv2.bitwise_and(img_extracted_box, gt_box)

        if iou:
            union = cv2.bitwise_or(img_extracted_box, gt_box)
            IoU = np.sum(intersection.flatten(), dtype=float)/np.sum(union.flatten(), dtype=float)
            measure = IoU
        else:
            measure = np.sum(intersection.flatten(), dtype=float)/np.sum(gt_box.flatten(), dtype=float)

        if measure >= thresh:
            nb_correct_box += 1
            list_correct_boxes.append(box)
        else:
            nb_incorrect_box += 1

    result_localization.true_positive = nb_correct_box
    result_localization.false_positive = nb_incorrect_box
    return list_correct_boxes


def print_evaluation_digits(results_localization: ResultsLocalization, results_recognition: ResultsRecognition) -> None:
    print('\t__Evaluation of ID localization and recognition__')
    print('\t__Evaluation of ID localization')
    print('\tCorrect localized numbers : {}/{} (recall : {:.02f})'.format(results_localization.true_positive,
                                                                          results_localization.total_groundtruth,
                                                                          results_localization.recall))
    print('\tFalse positive : {}/{} (precision : {:.02f})'.format(results_localization.false_positive,
                                                                  results_localization.total_predicted,
                                                                  results_localization.precision))

    print('\t__Evaluation of ID recognition__')
    print('\tCorrect recognized numbers : {}/{} (recall : {:.02f})'.format(results_recognition.true_positive,
                                                                           results_recognition.total_groundtruth,
                                                                           results_recognition.recall))
    print('\t Character Error Rate (CER) : {}'.format(results_recognition.cer))


def global_digit_evaluation(final_boxes: list, groundtruth_labels_digits_filename: str, thresh=0.5, use_iou=False,
                            printing=True) -> (ResultsLocalization, ResultsRecognition, BoxLabelPrediction):
    """

    :param final_boxes:
    :param groundtruth_labels_digits_filename:
    :param use_iou:
    :param thresh:
    :param printing:
    :return:
    """

    labels_matrix = get_labelled_digits_matrix(groundtruth_labels_digits_filename)
    total_gt = len(np.unique(labels_matrix)) - 1
    total_predicted = len(final_boxes)

    # Localisation
    results_localization = ResultsLocalization(total_truth=total_gt,
                                               total_predicted=total_predicted,
                                               thresh=thresh)
    list_correct_boxes = evaluation_digit_localisation(labels_matrix, final_boxes, results_localization, thresh=thresh,
                                                       iou=use_iou)
    results_localization.compute_metrics()

    # Recognition
    results_recognition = ResultsRecognition(total_truth=len(list_correct_boxes),
                                             total_predicted=len(list_correct_boxes),
                                             thresh=thresh)
    box_prediction_list = evaluation_digit_recognition(labels_matrix, list_correct_boxes, results_recognition)
    results_recognition.compute_metrics()

    if printing:
        print_evaluation_digits(results_localization=results_localization,
                                results_recognition=results_recognition)

        print_digit_counts(results_recognition.partial_measure)

    return results_localization, results_recognition, box_prediction_list
