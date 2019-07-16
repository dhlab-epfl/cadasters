#!/usr/bin/env python
__author__ = 'solivr'

import os


class Params:
    def __init__(self, **kwargs):
        self.input_filenames = kwargs.get('input_filenames')
        self.output_dir = kwargs.get('output_dir')
        self.classifier_filename = kwargs.get('classifier_filename')
        self.tf_model_dir = kwargs.get('tf_model_dir')  # Path of tensorflow model to be used for digit recognition.
        self.show_plots = kwargs.get('show_plots', True)  # To save intermediate plots of polygons and boxes
        # TODO : Add tag and signature def of tf model as parameter
        # To evaluate the results (parcel extraction and digit recognition). A ground truth must exist in
        # data/data_evaluation and should me named as nameCadasterFile_{parcels, digits}_gt.jpg
        self.evaluate = kwargs.get('evaluate', False)
        self.debug_flag = kwargs.get('debug', False)  # Saves graph and useful variables after each step to ease debug.
        self.gpu = kwargs.get('gpu', '')
        # method to measure the similarity between regions:
        #   'cie2000' (recommended) is based on LAB colors
        #   'coloredge' mixes edge features with LAB features,
        #   'edges' uses only edge features
        self.merging_similarity_method = kwargs.get('merging_similarity_method', 'cie2000')
        # Value to stop removing edges within a subgraph of the graph representation of the image. Maximum intraregion
        # dissimilarity variation. A value greater than stop_criterion indicates that there are edges linking
        # dissimilar vertices. When value < stop_criterion, the elements of the subgraph can be merged together
        # to form an homogeneous region.
        self.merging_stop_criterion = kwargs.get('merging_stop_criterion', 0.3)
        # Related to the number os desired segments. Percent is a percentage of the total number of pixels of the image
        self.slic_percent = kwargs.get('slic_percent', 0.01)
        self.slic_compactness = kwargs.get('slic_compactness', 25)  # parameter of compactness of SLIC algorithm
        self.slic_sigma = kwargs.get('slic_sigma', 2)  # width of gaussian kernel for smoothing in SLIC algorithm
        self.slic_mode = kwargs.get('slic_mode', 'RGB') # Channel(s) to which SLIC is applied
        self.iou_threshold_parcels = kwargs.get('iou_threshold_parcels', 0.7)
        self.iou_threshold_digits = kwargs.get('iou_threshold_digits', 0.5)
        self.inter_threshold_digits = kwargs.get('inter_threshold_digits', 0.8)
        self.list_features = kwargs.get('list_features')
        self.label_background_class = kwargs.get('label_background_class', 0)
        self.polygon_approx_epsilon = kwargs.get('polygon_approx_epsilon', 3)

        self.filename_geojson = os.path.join(self.output_dir, 'parcels_polygons.geojson')
        self.filename_log = os.path.join(self.output_dir, 'logs.txt')

        self._assert_init_values()
        self._intermediate_files_creation()

        if self.debug_flag:
            os.makedirs(self._intermediate_outputs_folder, exist_ok=True)
        if self.show_plots:
            self._plots_creation()

        os.makedirs(self.output_dir, exist_ok=True)
        # Create directory to save digits image
        self._dir_digits = os.path.join(self.output_dir, 'digits')
        os.makedirs(self._dir_digits, exist_ok=True)

    def _assert_init_values(self):
        assert self.merging_similarity_method in ['cie2000', 'coloredge', 'edges'], \
            "Merging method '{}' not known".format(self.merging_similarity_method)
        assert self.slic_mode in ['L', 'RGB'], 'SLIC Mode unknown {}'.format(self.slic_mode)
        assert 0.0 < self.slic_percent <= 1.0, 'SLIC percent must be in ]0,1]'

        assert self.list_features is not None
        assert self.tf_model_dir is not None, 'TF model direcory needed'
        assert self.classifier_filename is not None, 'Classifier filename needed'
        assert self.output_dir is not None
        assert self.input_filenames is not None

        if self.evaluate:
            path_eval_split = os.path.split(self.input_filenames)
            # Get filename labelled parcels
            filename = '{}_labelled_parcels_gt.jpg'.format(path_eval_split[1].split('.')[0])
            self._groundtruth_parcels_filename = os.path.join(path_eval_split[0], filename)

            # Get filename ground truth labelled digits
            filename = '{}_digits_label_gt.png'.format(path_eval_split[1].split('.')[0])
            self._groundtruth_labels_digits_filename = os.path.join(path_eval_split[0], filename)

            if os.path.exists(self._groundtruth_parcels_filename) and \
                    os.path.exists(self._groundtruth_labels_digits_filename):
                pass
            else:
                self.evaluate = False
                print('** ! WARNING ! : {} and/or {} ground truth file do not exist. Cannot perform evaluation'.format(
                    self._groundtruth_parcels_filename, self._groundtruth_labels_digits_filename))
                self._groundtruth_parcels_filename, self._groundtruth_labels_digits_filename = None, None

    def _intermediate_files_creation(self):
        self._intermediate_outputs_folder = os.path.join(self.output_dir, 'intermediate_files')
        self._saving_filename_feats = os.path.join(self._intermediate_outputs_folder, 'feats.pkl')
        self._saving_filename_graph = os.path.join(self._intermediate_outputs_folder, 'graph.pkl')
        self._saving_filename_nsegments = os.path.join(self._intermediate_outputs_folder, 'nsegments.pkl')
        self._saving_filename_listpoly = os.path.join(self._intermediate_outputs_folder, 'listpoly.pkl')
        self._saving_filename_dicpoly = os.path.join(self._intermediate_outputs_folder, 'dicpoly.pkl')
        self._saving_filename_boxes = os.path.join(self._intermediate_outputs_folder, 'boxes.pkl')

    def _plots_creation(self):
        self._dir_cropped_polygons = os.path.join(self.output_dir, 'cropped_polygons')
        os.makedirs(self._dir_cropped_polygons, exist_ok=True)

        self._plots_filename_superpixels = os.path.join(self.output_dir, 'sp_merged.jpg')
        self._plots_filename_ridge = os.path.join(self.output_dir, 'ridges_to_flood.jpg')
        self._plots_filename_polygons = os.path.join(self.output_dir, 'polygons.jpg')
        self._plots_filename_class = os.path.join(self.output_dir, 'predicted3class.jpg')
        self._plots_filename_finalbox = os.path.join(self.output_dir, 'finalBox.jpg')
        self._plots_filename_allbox = os.path.join(self.output_dir, 'allBox.jpg')


    @property
    def intermediate_outputs_folder(self):
        return self._intermediate_outputs_folder

    @property
    def saving_filename_feats(self):
        return self._saving_filename_feats

    @property
    def saving_filename_graph(self):
        return self._saving_filename_graph

    @property
    def saving_filename_nsegments(self):
        return self._saving_filename_nsegments

    @property
    def saving_filename_listpoly(self):
        return self._saving_filename_listpoly

    @property
    def saving_filename_dicpoly(self):
        return self._saving_filename_dicpoly

    @property
    def saving_filename_boxes(self):
        return self._saving_filename_boxes

    @property
    def groundtruth_parcels_filename(self):
        return self._groundtruth_parcels_filename

    @property
    def groundtruth_labels_digits_filename(self):
        return self._groundtruth_labels_digits_filename

    @property
    def plots_filename_superpixels(self):
        return self._plots_filename_superpixels

    @property
    def plots_filename_ridge(self):
        return self._plots_filename_ridge

    @property
    def plots_filename_polygons(self):
        return self._plots_filename_polygons

    @property
    def plots_filename_class(self):
        return self._plots_filename_class

    @property
    def plots_filename_finalbox(self):
        return self._plots_filename_finalbox

    @property
    def plots_filename_allbox(self):
        return self._plots_filename_allbox

    @property
    def dir_cropped_polygons(self):
        return self._dir_cropped_polygons

    @property
    def dir_digits(self):
        return self._dir_digits