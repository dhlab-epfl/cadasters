import numpy as np
import cv2
import math
from scipy import ndimage

# Normalization algorithm inspired by http://openmachin.es/blog/tensorflow-mnist


# Returns a vector to translate the center of mass of the digit to the center of the image
def get_translation_vector(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape

    shift_x = np.round(cols/2.0-cx).astype(int)
    shift_y = np.round(rows/2.0-cy).astype(int)

    return shift_x, shift_y


# Translates the image by a vector
def translate(img, sx, sy):
    rows, cols = img.shape
    matrix = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, matrix, (cols, rows))

    return shifted


# Normalizes the digit in the same way as the MNIST database
# See http://yann.lecun.com/exdb/mnist/ for explanations
def preprocess_digit(digit):

    # The image has to be in floats representation
    digit = digit / 255.0

    # Crop the image to remove whitespace around the digit
    while np.sum(digit[0]) == 0:
        digit = digit[1:]

    while np.sum(digit[:, 0]) == 0:
        digit = np.delete(digit, 0, 1)

    while np.sum(digit[-1]) == 0:
        digit = digit[:-1]

    while np.sum(digit[:, -1]) == 0:
        digit = np.delete(digit, -1, 1)

    rows, cols = digit.shape

    # The image has to be a square and we need to keep the aspect ratio
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        digit = cv2.resize(digit, (cols, rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        digit = cv2.resize(digit, (cols, rows))

    # Adds white padding around the digit to fit a 28 by 28 squared image
    cols_padding = (int(math.ceil((28-cols)/2.0)), int(math.floor((28-cols)/2.0)))
    rows_padding = (int(math.ceil((28-rows)/2.0)), int(math.floor((28-rows)/2.0)))
    digit = np.lib.pad(digit, (rows_padding, cols_padding), 'constant')

    # Translates the image so its center of mass is at the center of the image
    shift_x, shift_y = get_translation_vector(digit)
    shifted = translate(digit, shift_x, shift_y)

    return shifted
