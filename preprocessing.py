***REMOVED***
***REMOVED***

import numpy as np
import cv2


def image_filtering(img: np.array, filters: str='nlmeans') -> np.array:
***REMOVED***"
    Filters the input image to remove the noise (median smoothing, gaussain smoothing or
    non local means denoising)
    :param img: image to filter
    :param filters:
            'medFilt' : Median filtering
            'gaussFilt' : Gaussian filtering (assumes Gaussian noise)
            'nlmeans' : Non local means filtering (default)
    :return: image filtered, same size and type as input image
***REMOVED***"
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
        print('Image not filtered. Filter ***REMOVED******REMOVED*** does not exist'.format(filters))

    return img_filt
