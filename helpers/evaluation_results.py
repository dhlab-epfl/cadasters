#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
from collections import Counter
from helpers import minimum_edit_distance


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
        assert ((self.true_positive + self.false_positive) and self.total_groundtruth), \
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
        assert (self.total_groundtruth != 0 and self.total_chars != 0), \
            "True and False positives not initialized or partial_recognition is not initialized"

        self.recall = self.true_positive/self.total_groundtruth

        #  CER
        sums_partials = np.sum(self.partial_recognition, axis=0)
        self.cer = sums_partials[0] / self.total_chars

        self.partial_measure = Counter(self.partial_recognition[:, 1])


class BoxLabelPrediction:

    def __init__(self, **kwargs):
        self.prediction = int(kwargs.get('prediction'))
        self.groundtruth = kwargs.get('groundtruth')
        self.confidence = kwargs.get('confidence', 0)
        self.box_points = kwargs.get('points')
        self.center = self._compute_center()
        self.correctness = self._compute_correctness()
        if not self.correctness:
            self.error_type, self.edit_distance = self._compute_error_type()

    def _compute_center(self):
        # point (x,y)
        x1 = np.min(self.box_points[:, 0])
        x2 = np.max(self.box_points[:, 0])
        y1 = np.min(self.box_points[:, 1])
        y2 = np.max(self.box_points[:, 1])
        xcenter = (x1 + x2)/2
        ycenter = (y1 + y2)/2
        return [xcenter, ycenter]

    def _compute_correctness(self):
        return self.prediction == self.groundtruth

    def _compute_error_type(self):
        groundtruth_str = str(self.groundtruth)
        prediction_str = str(self.prediction)
        if groundtruth_str == prediction_str:
            return None, None
        else:
            if len(groundtruth_str) == len(prediction_str):
                error_type = LabelErrorType.SUBSTITUTION
                distance = minimum_edit_distance(groundtruth_str, prediction_str)
            elif len(groundtruth_str) > len(prediction_str):
                error_type = LabelErrorType.DELETION
                distance = minimum_edit_distance(groundtruth_str, prediction_str)
            elif len(groundtruth_str) < len(prediction_str):
                error_type = LabelErrorType.ADDITION
                distance = minimum_edit_distance(groundtruth_str, prediction_str)
            else:
                raise NotImplementedError

            return error_type, distance


class LabelErrorType:
    DELETION = 'DELETION'
    SUBSTITUTION = 'SUBSTITUTION'
    INSERTION = 'INSERTION'

