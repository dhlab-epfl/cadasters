#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
from collections import Counter


class ResultsLocalization:

    def __init__(self, **kwargs):
        self.true_positive = kwargs.get('true_positives', 0)
        self.false_positive = kwargs.get('false_positives', 0)
        self.total_groundtruth = kwargs.get('total_truth', 0)
        self.total_predicted = kwargs.get('total_predicted', 0)
        self.threshold = kwargs.get('thresh', 0)
        self.recall = kwargs.get('recall', 0)
        self.precision = kwargs.get('precision', 0)

    def compute_metrics(self):
        assert (self.true_positive and self.false_positive and self.total_groundtruth), \
            "True and False positives not initialized"

        self.recall = self.true_positive/self.total_groundtruth
        self.precision = self.true_positive/(self.true_positive + self.false_positive)


class ResultsRecognition:

    def __init__(self, **kwargs):
        self.true_positive = kwargs.get('true_positives', 0)
        self.false_positive = kwargs.get('false_positives', 0)
        self.total_groundtruth = kwargs.get('total_truth', 0)
        self.total_predicted = kwargs.get('total_predicted', 0)
        self.partial_recognition = kwargs.get('partial_recognition')
        self.recall = kwargs.get('recall', 0)
        self.total_chars = kwargs.get('total_chars', 0)
        self.cer = kwargs.get('cer', 0)

    def compute_metrics(self):
        assert (self.true_positive != 0 and self.total_groundtruth != 0 and \
                self.total_chars != 0 and self.partial_recognition.any), \
            "True and False positives not initialized or partial_recognition is not initialized"

        self.recall = self.true_positive/self.total_groundtruth

        #  CER
        sums_partials = np.sum(self.partial_recognition, axis=0)
        self.cer = sums_partials[0] / self.total_chars

        self.partial_measure = Counter(self.partial_recognition[:, 1])
