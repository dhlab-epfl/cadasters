import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import color
from helpers import Params


def compute_slic(img: np.array, params: Params) -> np.array:
    """
    Compute superpixels of the image using SLIC algorithm
    :param img: 3-channel-RGB image.
    :param params
    :return: Superpixels. 2D map with a label at each pixel indicating wich superpixel it belongs to
    """

    # To change RGB color_space into LAB color space
    if params.slic_mode == 'L':
        lab_py = color.rgb2lab(img)
        img_slic = lab_py[:, :, 0]

    # No change in color space
    elif params.slic_mode == 'RGB':
        img_slic = img

    # Make suerpixels with SLIC
    # <<<<<<<<< THIS NEEDS TO BE CHANGED numSEGMENST MUST BE DEFINE OUTSIDE
    numSegments = np.uint64(np.round(params.slic_percent*np.prod(img.shape[:2])))

    superpixels = slic(img_as_float(img_slic), n_segments=numSegments, compactness=params.slic_compactness,
                       enforce_connectivity=True, sigma=params.slic_sigma)

    return superpixels
