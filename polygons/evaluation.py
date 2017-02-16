from collections import Counter
import numpy as np
import cv2


def evalutation_parcel_iou(parcels_groundtruth, dic_polygons, iou_thresh=0.8):
    """
    Evaluates the extraction of parcels using Intersection over Union metric (IoU)
    :param parcels_groundtruth: image with labelled parcels
    :param dic_polygons: dictionary with node graphs as keys and
                        tuple of (uuid, list of polygons in cv2 format) as values
    :param iou_thresh: IoU threshold to consider correct or incorrect extraction
    :return: correct_poly, incorrect_poly : number of correctly / incorrectly  extracted parcels
    """
    correct_poly = 0
    incorrect_poly = 0
    # final_img_poly = np.zeros(parcels_groundtruth.shape, dtype='uint8')

    for key, list_tup in dic_polygons.items():
        for tup in list_tup:
            # Draw polygon extracted
            extracted_poly = np.zeros(parcels_groundtruth.shape, dtype='uint8')
            cv2.fillPoly(extracted_poly, tup[1], 255)
            extracted_poly_bin = extracted_poly > 0

            # Count which is the label that appears the most and consider that it is the label of the parcel
            label_poly = Counter(parcels_groundtruth[extracted_poly_bin]).most_common(1)[0][0]
            if label_poly == 0:
                incorrect_poly += 1
                continue

            # Compute intersection over union (IoU)
            gt_poly = np.uint8(parcels_groundtruth == label_poly) * 255
            intersection = cv2.bitwise_and(extracted_poly, gt_poly)
            union = cv2.bitwise_or(extracted_poly, gt_poly)
            IoU = np.sum(intersection.flatten()) / np.sum(union.flatten())

            if IoU >= iou_thresh:
                correct_poly += 1
                # cv2.fillPoly(final_img_poly, tup[1], 255)
            else:
                incorrect_poly += 1

    return correct_poly, incorrect_poly


def print_evaluation_parcels(results_evaluation_parcels):
    print('\t --Evaluation polygon extraction --')

    print('\t\tNumber correct polygons : {}/{}, recall : {:.02f}'.
        format(results_evaluation_parcels['true_positive'],
               results_evaluation_parcels['total_groundtruth'],
               results_evaluation_parcels['recall']))

    print('\t\tNumber incorrect polygons : {}/{}'.
          format(results_evaluation_parcels['false_positive'],
                 results_evaluation_parcels['total_extracted']))

    print('\t\tPrecision : {:.02f}'.format(results_evaluation_parcels['precision']))


def global_evaluation_parcels(dic_polygon, groundtruth_parcels_filename, iou_thresh_parcels=0.6):
    # Open image and give a unique label to each parcel
    image_parcels_gt = cv2.imread(groundtruth_parcels_filename)
    image_parcels_gt = np.uint8(image_parcels_gt[:, :, 0] > 128) * 255
    n_labels_poly, parcels_labeled = cv2.connectedComponents(image_parcels_gt)

    # Evaluate
    correct_poly, incorrect_poly = evalutation_parcel_iou(parcels_labeled, dic_polygon,
                                                          iou_thresh=iou_thresh_parcels)

    results_evaluation_parcels = {'total_groundtruth': n_labels_poly - 1,
                                  'total_extracted': correct_poly + incorrect_poly,
                                  'true_positive': correct_poly,
                                  'false_positive': incorrect_poly,
                                  'precision': correct_poly / (correct_poly + incorrect_poly),
                                  'recall': correct_poly / (n_labels_poly - 1)
                                  }

    print_evaluation_parcels(results_evaluation_parcels)

    return results_evaluation_parcels
