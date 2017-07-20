import cv2
import numpy as np
from collections import Counter
from scipy import misc
from helpers import minimum_edit_distance, count_correct_characters, ResultsLocalization, ResultsRecognition


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


def evaluation_digit_recognition(label_matrix: np.array, list_extracted_boxes: list) -> (int, int, np.array):

    # Number of false and true positives
    n_false_positives = 0
    n_true_positives = 0
    # List with Levensthein distances, correct digits and total number of digits
    list_partial_numbers_results = list()
    # List of tuples (predicted_number, correct_number) (only for partially correct numbers)
    list_tup_predicted_correct = list()

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

    # Transform to numpy array
    return n_true_positives, n_false_positives, np.array(list_partial_numbers_results)
# ---------------------------------------------------------------------


# def interpret_digit_results(n_true_positives, partial_numbers_results, printing=True):
#
#     n_partial_numbers = partial_numbers_results.shape[0]
#     # n_predicted_numbers = n_partial_numbers + n_true_positives + n_false_positives
#
#     # sums_results = np.sum(partial_numbers_results, axis=0)
#
#     # Character Error Rate
#     # CER = sums_results[0] / sums_results[2]
#     counts_digits = Counter(partial_numbers_results[:, 1])
#
#     if printing:
#         # print('CER : {:.02f}'.format(CER))
#         print('Partial retrieval {}/{} ({:.02f})'.format(n_partial_numbers, n_true_positives, n_partial_numbers / n_true_positives))
#         # Print results by number of correctly retrieved digit
#         print(print_digit_counts(counts_digits))
#
#     return counts_digits


def evaluation_digit_localisation(digits_groundtruth, list_boxes: list, thresh=0.5, iou=True) -> (int, int, list):
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

    return nb_correct_box, nb_incorrect_box, list_correct_boxes


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


# def global_digit_evaluation(final_boxes, groundtruth_labels_digits_filename,
#                             iou_thresh=0.5, inter_thresh=0.8, printing=True):
#     """
#
#     :param final_boxes:
#     :param groundtruth_labels_digits_filename:
#     :param iou_thresh:
#     :param inter_thresh:
#     :param printing:
#     :return: results_evaluation_digits : {
#                                             true_positive_box_iou
#                                             false_positive_box_iou
#                                             true_positive_box_inter
#                                             false_positive_box_inter
#                                             true_positive_numbers : using 'true_positive_box_inter' boxes
#                                             false_positive_numbers : using 'true_positive_box_inter' boxes
#                                             partial_numbers_results : using 'true_positive_box_inter' boxes
#                                             true_positive_numbers_iou : using 'true_positive_box_iou'
#                                             false_positive_numbers_iou : using 'true_positive_box_iou'
#                                             partial_numbers_results_iou : using 'true_positive_box_iou'
#                                             total_groundtruth :
#                                             total_predicted
#                                             total_predicted_iou
#                                             }
#     """
#
#     labels_matrix = get_labelled_digits_matrix(groundtruth_labels_digits_filename)
#     total_gt = len(np.unique(labels_matrix)) - 1
#     total_predicted = len(final_boxes)
#
#     # -- Case IOU --
#     # Localization
#     results_localization_iou = ResultsLocalization(total_truth=total_gt,
#                                                    total_predicted=total_predicted,
#                                                    thresh=iou_thresh)
#     results_localization_iou.true_positive, results_localization_iou.false_positive, \
#         list_correct_boxes_iou = evaluation_digit_localisation(labels_matrix, final_boxes, thresh=iou_thresh, iou=True)
#     results_localization_iou.compute_metrics()
#
#     # Recognition for IoU
#     results_recognition_iou = ResultsRecognition(total_truth=len(list_correct_boxes_iou),
#                                                  total_predicted=len(list_correct_boxes_iou),
#                                                  thresh=iou_thresh)
#     results_recognition_iou.true_positive, results_recognition_iou.false_positive, \
#         results_recognition_iou.partial_recognition = evaluation_digit_recognition(labels_matrix, list_correct_boxes_iou)
#     results_recognition_iou.compute_metrics()
#
#     if printing:
#         print_evaluation_digits(results_localization=results_localization_iou,
#                                 results_recognition=results_recognition_iou)
#         print_digit_counts(results_recognition_iou.partial_measure)
#
#     # -- Case Inter --
#     # Localisation (intersection)
#     results_localization_inter = ResultsLocalization(total_truth=total_gt,
#                                                      total_predicted=total_predicted,
#                                                      thresh=inter_thresh)
#     results_localization_inter.true_positive, results_localization_inter.false_positive, \
#         list_correct_boxes_inter = evaluation_digit_localisation(labels_matrix, final_boxes, thresh=inter_thresh, iou=False)
#     results_localization_inter.compute_metrics()
#
#     # Recognition for Intersection
#     results_recognition_inter = ResultsRecognition(total_truth=len(list_correct_boxes_inter),
#                                                    total_predicted=len(list_correct_boxes_inter),
#                                                    thresh=inter_thresh)
#     results_recognition_inter.true_positive, results_recognition_inter.false_positive, \
#         results_recognition_inter.partial_recognition \
#         = evaluation_digit_recognition(labels_matrix, list_correct_boxes_inter)
#     results_recognition_inter.compute_metrics()
#
#     if printing:
#         print_evaluation_digits(results_localization=results_localization_inter,
#                                 results_recognition=results_recognition_inter)
#
#         print_digit_counts(results_recognition_inter.partial_measure)
#
#     results = {'localization_inter': results_localization_inter,
#                'localization_iou': results_localization_iou,
#                'recognition_inter': results_recognition_inter,
#                'recognition_iou': results_recognition_iou}
#
#     return results

def global_digit_evaluation(final_boxes: list, groundtruth_labels_digits_filename: str,
                            thresh=0.5, use_iou=False, printing=True) -> (ResultsLocalization, ResultsRecognition):
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
    results_localization.true_positive, results_localization.false_positive, list_correct_boxes \
        = evaluation_digit_localisation(labels_matrix, final_boxes, thresh=thresh, iou=use_iou)
    results_localization.compute_metrics()

    # Recognition
    results_recognition = ResultsRecognition(total_truth=len(list_correct_boxes),
                                             total_predicted=len(list_correct_boxes),
                                             thresh=thresh)
    results_recognition.true_positive, results_recognition.false_positive, results_recognition.partial_recognition \
        = evaluation_digit_recognition(labels_matrix, list_correct_boxes)
    results_recognition.compute_metrics()

    if printing:
        print_evaluation_digits(results_localization=results_localization,
                                results_recognition=results_recognition)

        print_digit_counts(results_recognition.partial_measure)

    return results_localization, results_recognition
