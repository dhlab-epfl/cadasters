#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
from collections import Counter


class ResultsLocalization:

    def __init__(self, **kwargs):
        self.true_positive = kwargs.get('true_positives')
        self.false_positive = kwargs.get('false_positives')
        self.total_groundtruth = kwargs.get('total_truth')
        self.total_predicted = kwargs.get('total_predicted')
        self.threshold = kwargs.get('thresh')
        self.recall = kwargs.get('recall')
        self.precision = kwargs.get('precision')
        self.cer = kwargs.get('cer')

    def compute_metrics(self):
        assert (self.true_positive and self.false_positive and self.total_groundtruth), \
            "True and False positives not initialized"

        self.recall = self.true_positive/self.total_groundtruth
        self.precision = self.true_positive/(self.true_positive + self.false_positive)


class ResultsRecognition:

    def __init__(self, **kwargs):
        self.true_positive = kwargs.get('true_positives')
        self.false_positive = kwargs.get('false_positives')
        self.total_groundtruth = kwargs.get('total_truth')
        self.total_predicted = kwargs.get('total_predicted')
        self.partial_recognition = kwargs.get('partial_recognition')
        self.recall = kwargs.get('recall')

    def compute_metrics(self):
        assert (self.true_positive and self.total_groundtruth and self.partial_recognition), \
            "True and False positives not initialized or partial_recognition is not initialized"

        self.recall = self.true_positive/self.total_groundtruth

        #  CER
        sums_partials = np.sum(self.partial_recognition, axis=0)
        self.cer = sums_partials[0] / sums_partials[2]

        self.partial_measure = Counter(self.partial_recognition[:, 1])
