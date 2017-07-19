import datetime
import numpy as np
# from text.evaluation import print_digit_counts


def write_log_file(filename, **kwargs):

    cadaster_filename = kwargs.get('cadaster_filename')
    size_image = kwargs.get('size_image')
    params_slic = kwargs.get('params_slic')
    list_dict_features = kwargs.get('list_dict_features')
    similarity_method = kwargs.get('similarity_method')
    stop_criterion = kwargs.get('stop_criterion')
    elapsed_time = kwargs.get('elapsed_time')
    classifier_filename = kwargs.get('classifier_filename')
    tf_model = kwargs.get('digit_tf_model')
    res_parcels = kwargs.get('results_parcels_eval')
    res_digits = kwargs.get('results_digits_eval')  # dictionnary with iou and inter results tuple (localization, recognition)

    minutes, seconds = divmod(np.float32(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)
    date = datetime.datetime.now()

    # Open file (or create it)
    log_file = open(filename, 'w+')
    log_file.write('Date of log creation : {:02d}.{:02d}.{:02d} at {:02d}:{:02d} \n'
                   .format(date.day, date.month, date.year, date.hour, date.minute))
    log_file.write('Time elapsed to process image : {:02d}:{:02d}:{:02d}\n\n'
                   .format(int(hours), int(minutes), int(seconds)))

    log_file.write('---- Image  -----\n')
    log_file.write('Filename : {}'.format(cadaster_filename))
    log_file.write(', size : {}x{} \n'.format(size_image[0], size_image[1]))
    log_file.write('---- Superpixels ----\n')
    log_file.write(' Params : {}\n'.format(params_slic))
    log_file.write('---- Features ----\n')
    log_file.write('{} \n'.format(list_dict_features))
    log_file.write(' ---- Merging ---- \n')
    log_file.write('Similarity method : {}\n'.format(similarity_method))
    log_file.write('Stop criterion : {}\n'.format(stop_criterion))
    log_file.write('---- Classification ----\n')
    log_file.write('Classifier file: {}\n'.format(classifier_filename))
    log_file.write('Digit recognizer TF model : {}\n'.format(tf_model))

    if res_parcels:
        log_file.write('---- Evaluation parcels ----\n')
        log_file.write('IoU threshold : {}\n'.format(res_parcels.threshold))
        log_file.write('Total parcels (groundtruth) : {}\n'.format(res_parcels.total_groundtruth))
        log_file.write('Total extracted parcels : {}\n'.format(res_parcels.total_predicted))
        log_file.write('True positives parcels : {}/{}  /  Recall : {:.02f}\n'.format(
                    res_parcels.true_positive,
                    res_parcels.total_groundtruth,
                    res_parcels.recall))
        log_file.write('False positives parcels : {}/{}  /  Precision : {:.02f}\n'.format(
                    res_parcels.false_positive,
                    res_parcels.total_predicted,
                    res_parcels.precision))

    if res_digits:
        for key, results_tuple in res_digits.items():
            result_localization, result_recognition = results_tuple
            log_file.write('---- Evaluation digits {} with thresh = {}----\n'.format(key, result_localization.threshold))

            log_file.write('** Localization\n')
            log_file.write('Total labels (groundtruth) : {}\n'.format(result_localization.total_groundtruth))
            log_file.write('Total extracted boxes : {}\n'.format(result_localization.total_predicted))
            log_file.write('True positive boxes : {}/{} (recall : {:.02f})\n'.format(
                                                            result_localization.true_positive,
                                                            result_localization.total_groundtruth,
                                                            result_localization.recall))
            log_file.write('False positive boxes : {}/{} (precision : {:.02f})\n'.format(
                                                            result_localization.false_positive,
                                                            result_localization.total_predicted,
                                                            result_localization.precision))
            log_file.write('** Recognition\n')
            log_file.write('Correct recognized numbers : {}/{} ({:.02f}) \n'.format(
                result_recognition.true_positive,
                result_recognition.total_groundtruth,
                result_recognition.recall))
            log_file.write('Character Error Rate (CER) : {:.02f}\n'.format(result_recognition.cer))

            log_file.write('Partial retrieval)\n')
            log_file.write(print_digit_counts(result_recognition.partial_recognition))

    # if res_digits:
    #     log_file.write('---- Evaluation digits ----\n')
    #
    #     log_file.write('** Localization\n')
    #     log_file.write('Total labels (groundtruth) : {}\n'.format(res_digits['total_groundtruth']))
    #     log_file.write('Total extracted boxes : {}\n'.format(res_digits['total_predicted']))
    #     log_file.write('> IoU threshold : {}\n'.format(iou_thresh_digits))
    #     log_file.write('True positive boxes : {}/{}\n'.format(res_digits['true_positive_box_iou'],
    #                                                           res_digits['total_groundtruth']))
    #     log_file.write('False positive boxes : {}/{}\n'.format(res_digits['false_positive_box_iou'],
    #                                                            res_digits['total_predicted']))
    #
    #     log_file.write('> Intersection threshold : {}\n'.format(inter_thresh_digits))
    #     log_file.write('True positive boxes : {}/{}\n'.format(res_digits['true_positive_box_inter'],
    #                                                           res_digits['total_groundtruth']))
    #     log_file.write('False positive boxes : {}/{}\n'.format(res_digits['false_positive_box_inter'],
    #                                                            res_digits['total_predicted']))
    #
    #     log_file.write('** Recognition\n')
    #     log_file.write('Correct recognized numbers : {}/{} ({:.02f}) \n'.format(res_digits['true_positive_numbers'],
    #                                                                   res_digits['true_positive_box_inter'],
    #                             res_digits['true_positive_numbers'] / res_digits['true_positive_box_inter']))
    #
    # if (CER is not None) and (counts_digits is not None):
    #     log_file.write('Character Error Rate (CER) : {:.02f}\n'.format(CER))
    #
    #     n_partial_numbers = sum(np.array([counts_digits[i] for i in counts_digits.keys()]))
    #     log_file.write('Partial retrieval {}/{} ({:.02f})\n'.format(n_partial_numbers,
    #                                                                 res_digits['true_positive_box_inter'],
    #                                                     n_partial_numbers / res_digits['true_positive_box_inter']))
    #     log_file.write(print_digit_counts(counts_digits))

    # Close file
    log_file.close()

def print_digit_counts(counts_digits):

    total_counts = sum(np.array([counts_digits[i] for i in counts_digits.keys()]))

    str_to_print = ''
    for i in sorted(counts_digits.keys(), reverse=True):
        str_to_print += '\t{} digit(s) : {}/{} ({:.02f})\n'.format(i, counts_digits[i], total_counts,
                                                                   counts_digits[i] / total_counts)
    return str_to_print