import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import color


def compute_slic(img, params_slic):
    """
    Compute superpixels of the image using SLIC algorithm
    :param img: 3-channel-RGB image.
    :param params_slic: dictionnary of parameters
                        'mode' : 'L' or 'RGB' (recommended) channel(s) to which SLIC is applied
                        'percent' : related to the number os segments. Percent is a percentage of the image
                        'numCompact' : parameter of compacity of SLIC algorithm
                        'sigma' : width of gaussian kernel for smoothing in sSLIC algorithm
    :return: Superpixels. 2D map with a label at each pixel indicating wich superpixel it belongs to
    """

    # To change RGB color_space into LAB color space
    if params_slic['mode'] == 'L':
        lab_py = color.rgb2lab(img)
        img_slic = lab_py[:,:,0]

    # No change in color space
    elif params_slic['mode'] == 'RGB':
        img_slic = img

    # Make suerpixels with SLIC
    percent = params_slic['percent']  # <<<<<<<<< THIS NEEDS TO BE CHANGED numSEGMENST MUST BE DEFINE OUTSIDE
    numSegments = np.uint64(np.round(percent*np.prod(img.shape[:2])))

    superpixels = slic(img_as_float(img_slic), n_segments=numSegments, compactness=params_slic['numCompact'],
                       enforce_connectivity=True, sigma=params_slic['sigma'])

    return superpixels
