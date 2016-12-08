import numpy as np
import cv2
from preprocessing.frangiFilter2D import FrangiFilter2D
from helpers import rgb2gray, bgr2rgb
from skimage import color


def image_filtering(img, filters='nlmeans'):
    """
    Filters the input image to remove the noise (median smoothing, gaussain smoothing or
    non local means denoising)
    :param img: image to filter
    :param filters:
            'medFilt' : Median filtering
            'gaussFilt' : Gaussian filtering (assumes Gaussian noise)
            'nlmeans' : Non local means filtering (default)
    :return: image filtered, same size and type as input image
    """
    img_filt = img.copy()
    if 'medFilt' in filters:
        # Apply median filtering to remove partially the noise
        img_filt = cv2.medianBlur(img, 5)
    if 'gaussFilt' in filters:
        # Smooth the image to avid over segmentation and denoise assuming Gaussian noise
        # Gaussian filter of size [5 5]  and sigma = 2
        img_filt = cv2.GaussianBlur(img, (5, 5), 2)
    if 'nlmeans' in filters:
        k = 5
        img_filt = cv2.fastNlMeansDenoisingColored(img, h=k, hColor=k)
    else:
        print('Image not filtered. Filter {} does not exist'.format(filters))

    return img_filt


def features_extraction(img_bgr, list_features):
    """
    Extracts features listed in list of image and save it in a dictionnary
    :param img_bgr: image in BGR format
    :param list_features: list of features to extract (recommended : 'frangi', 'lab')
            'laplacian' : computes laplacian with kernel of size 3x3 on L channel
                        (The colorspace is previously changed to LAB)
            'sobel' : contours with Sobel filter (x and y)
            'gabor' : Gabor filter (size filter 5x5, sigma 2, lambda 1, gamma 0.2, 8 directions)
            'frangi' : Frangi filter, vesselness measure (refer to report) * REFERENCE
            'lab' : Image in LAB color space
            'RGB' : Image in RGB color space
    :return: dict_features is a dictionnary containing all the features computed
    """
    img_rgb = bgr2rgb(img_bgr) # <<<<<<<<<<< CHANGE THIS (OUTSIDE FUNCTION)

    # Convert BGR to L*a*b* color space and grayscale
    img_Lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # Separate a*b* channels for processing
    L,a,b = cv2.split(img_Lab)
    L_blur = cv2.GaussianBlur(L, (5, 5), 2)

    dict_features = {}

    # Features
    # -- Laplacian of Gaussian
    if 'laplacian' in list_features:
        laplacianL = cv2.Laplacian(L_blur, cv2.CV_64F, ksize=3)
        dict_features['laplacian'] = laplacianL

    # -- Sobel
    if 'sobel' in list_features:
        sobelx = cv2.Sobel(L_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(L_blur, cv2.CV_64F, 0, 1, ksize=3)
        dict_features['sobelx'] = sobelx
        dict_features['sobely'] = sobely

    # -- Gabor
    if 'gabor' in list_features:
        ksize = (5,5)
        sigma = 2
        lam = 1
        gamma = 0.2
        n_directions = 8 # even number and power of 2 if possible
        gabor = np.zeros([L.shape[0],L.shape[1], n_directions])
        for d in range(1, n_directions + 1):
            theta = (180/n_directions)*d
            k = cv2.getGaborKernel(ksize, sigma, theta, lam, gamma)
            gabor[:,:,d-1] = cv2.filter2D(L_blur, -1, k)

        combined_gabor = np.amax(gabor, axis=2)
        dict_features['gabor'] = combined_gabor

    # -- Frangi
    if 'frangi' in list_features:
        img_gray = rgb2gray(img_bgr, 'BGR')
        scaleRange = np.array([1, 4])
        frangiFilt = FrangiFilter2D(img_gray, FrangiScaleRange=scaleRange, FrangiScaleRatio=0.2)
        dict_features['frangi'] = frangiFilt

    # Lab using python
    if 'Lab' in list_features:
        lab_py = color.rgb2lab(img_rgb)
        dict_features['Lab'] = lab_py

    if 'RGB' in list_features:
        dict_features['RGB'] = img_rgb

    return dict_features



