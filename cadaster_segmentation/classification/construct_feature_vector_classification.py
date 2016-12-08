import numpy as np


def construct_feature_vector_classification(pixLbl, dict_feature, list_features):
    """
    Constructs the feature vector that will be used during classification (either training or prediction)

    :param pixLbl: pixels in the region
    :param dict_feature: dictionary of the features of all image
    :param list_features: list of features that are used to produce the feature vector
    :return: feature vector is a concatenation of all the features for the desired group of pixels
    """

    # Take pixels in superpixel and compute features
    feat_vector = np.zeros([1, 1])

    if 'Lab' in list_features:
        l_py = dict_feature['Lab'][:, :, 0]
        a_py = dict_feature['Lab'][:, :, 1]
        b_py = dict_feature['Lab'][:, :, 2]
        feat_vector = np.append(feat_vector, np.mean(l_py[pixLbl]))
        feat_vector = np.append(feat_vector, np.mean(a_py[pixLbl]))
        feat_vector = np.append(feat_vector, np.mean(b_py[pixLbl]))
    if 'laplacian' in list_features:
        feat_vector = np.append(feat_vector, np.mean(dict_feature['laplacian'][pixLbl]))
        feat_vector = np.append(feat_vector, np.std(dict_feature['laplacian'][pixLbl]))
    if 'frangi' in list_features:
        feat_vector = np.append(feat_vector, np.mean(dict_feature['frangi'][pixLbl]))
        feat_vector = np.append(feat_vector, np.std(dict_feature['frangi'][pixLbl]))
    if 'gabor' in list_features:
        feat_vector = np.append(feat_vector, np.mean(dict_feature['gabor'][pixLbl]))
        feat_vector = np.append(feat_vector, np.std(dict_feature['gabor'][pixLbl]))
    if 'sobel' in list_features:
        feat_vector = np.append(feat_vector, np.mean(dict_feature['sobelx'][pixLbl]))
        feat_vector = np.append(feat_vector, np.mean(dict_feature['sobely'][pixLbl]))

    feat_vector = np.delete(feat_vector, 0)

    return feat_vector
