#!/usr/bin/env python
__author__ = 'solivr'

import datetime
import numpy as np
from helpers import Params
from collections import Counter
import json


def write_log_file(params: Params, **kwargs) -> None:

    elapsed_time = kwargs.get('elapsed_time')
    res_parcels = kwargs.get('results_parcels_eval')
    res_digits = kwargs.get('results_digits_eval')  # dictionnary with iou and inter results tuple (localization, recognition)

    minutes, seconds = divmod(np.float32(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)
    date = datetime.datetime.now()

    # Open file (or create it)
    log_file = open(params.filename_log, 'w+')
    log_file.write('# Date of log creation : {:02d}.{:02d}.{:02d} at {:02d}:{:02d} \n'
                   .format(date.day, date.month, date.year, date.hour, date.minute))
    log_file.write('# Time elapsed to process image : {:02d}:{:02d}:{:02d}\n\n'
                   .format(int(hours), int(minutes), int(seconds)))

    log_file.write('-- Params --\n')
    log_file.write(json.dumps(vars(params)))

    if res_parcels:
        log_file.write('-- Evaluation parcels --\n')
        log_file.write(json.dumps(vars(res_parcels)))

    if res_digits:
        for key, results_tuple in res_digits.items():
            result_localization, result_recognition = results_tuple
            log_file.write('-- Evaluation digits {} --\n'.format(key))
            log_file.write('** Localization\n')
            log_file.write(json.dumps(vars(result_localization)))

            log_file.write('** Recognition\n')
            log_file.write(json.dumps(vars(result_recognition)))
            log_file.write('Partial retrieval)\n')
            log_file.write(print_digit_counts(result_recognition.partial_measure))

    # Close file
    log_file.close()


def print_digit_counts(counts_digits: Counter) -> str:

    total_counts = sum(np.array([counts_digits[i] for i in counts_digits.keys()]))

    str_to_print = ''
    for i in sorted(counts_digits.keys(), reverse=True):
        str_to_print += '\t{} digit(s) : {}/{} ({:.02f})\n'.format(i, counts_digits[i], total_counts,
                                                                   counts_digits[i] / total_counts)
    return str_to_print