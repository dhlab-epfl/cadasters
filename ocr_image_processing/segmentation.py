import numpy as np
import cv2


def segment_number_inner(img, gradient_factor=4):
    img_orig = np.copy(img)

    # Gradient to get the outline of digits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gradient_factor, gradient_factor))
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))

    img_contours = np.copy(img)
    img_contours, contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        # Create bounding boxes
        x, y, w, h = cv2.boundingRect(cnt)
        # Draw the filled contours on a black mask
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        # Count the ratio of non-empty pixels
        ratio = cv2.countNonZero(mask[y:y+h, x:x+w])/(w*h)

        # We need to find a better and more accurate condition
        if ratio > .1 and w > 10 and h > 15:
            digit = cv2.morphologyEx(img_orig, cv2.MORPH_CLOSE, kernel)
            digit[mask == 0] = 0
            digits.append((x+w, digit[y:y+h+1, x:x+w+1]))

    digits.sort(key=lambda tupl: tupl[0])
    return [digit for end_x, digit in digits]


def segment_number(img, number_of_digits=None):
    if number_of_digits is None:
        return segment_number_inner(img)
    else:
        kernel_sizes = [4, 3, 2, 5, 6]

        for factor in kernel_sizes:
            result = segment_number_inner(img, gradient_factor=factor)
#             if len(result) == number_of_digits:
#                 return result
            if len(result) >= number_of_digits:
                return result

        # Impossible to find the correct number of digits, return the default segmentation
        return segment_number_inner(img)
