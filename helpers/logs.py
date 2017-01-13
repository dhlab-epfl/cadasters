import datetime
import numpy as np
from text import print_digit_counts


def write_log_file(filename, **kwargs):

    cadaster_filename = kwargs.get('cadaster_filename', None)
    size_image = kwargs.get('size_image', None)
    params_slic = kwargs.get('params_slic', None)
    list_dict_features = kwargs.get('list_dict_features', None)
    similarity_method = kwargs.get('similarity_method', None)
    stop_criterion = kwargs.get('stop_criterion', None)
    elapsed_time = kwargs.get('elapsed_time', None)
    classifier_filename = kwargs.get('classifier_filename', None)
    correct_poly = kwargs.get('correct_poly', None)
    incorrect_poly = kwargs.get('incorrect_poly', None)
    total_poly = kwargs.get('total_poly', None)
    true_positive_numbers = kwargs.get('true_positive_numbers', None)
    false_positive_numbers = kwargs.get('false_positive_numbers', None)
    missed_numbers = kwargs.get('missed_numbers', None)
    total_predicted_numbers = kwargs.get('total_predicted_numbers', None)
    CER = kwargs.get('CER', None)
    counts_digits = kwargs.get('counts_digits', None)

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

    if correct_poly and total_poly and incorrect_poly:
        log_file.write('---- Evaluation parcels ----\n')
        log_file.write('Correct parcels : {}/{}\n'.format(correct_poly, total_poly))
        log_file.write('Incorrect parcels : {}/{} \n'.format(incorrect_poly, total_poly))
        log_file.write('Precision : {:.02f}\n'.format(correct_poly/(correct_poly+incorrect_poly)))
        log_file.write('Recall : {:.02f}\n'.format(correct_poly/total_poly))

    if true_positive_numbers and false_positive_numbers and missed_numbers and total_predicted_numbers:
        # Calculate totals
        total_partial_numbers = sum(np.array([counts_digits[i] for i in counts_digits.keys()]))
        total_true_numbers = true_positive_numbers + missed_numbers + total_partial_numbers

        log_file.write('---- Evaluation digits ----\n')
        log_file.write('Correct recognized numbers : {}/{} ({:.02f})\n'.format(true_positive_numbers, total_true_numbers,
                                                                               true_positive_numbers / total_true_numbers))
        log_file.write('False positive : {}/{} ({:.02f})'.format(false_positive_numbers, total_predicted_numbers,
                                                                 false_positive_numbers / total_predicted_numbers))
        log_file.write('Missed numbers : {}/{} ({:.02f})'.format(missed_numbers, total_true_numbers,
                                                                 missed_numbers / total_true_numbers))
    if CER and counts_digits:
        log_file.write('Character Error Rate (CER) : {:.02f}'.format(CER))
        log_file.write('Partial retrieval {}/{} (:.02f)'.format(total_predicted_numbers, total_true_numbers,
                                                                total_predicted_numbers / total_true_numbers))
        log_file.write(print_digit_counts(counts_digits))

    # Close file
    log_file.close()
