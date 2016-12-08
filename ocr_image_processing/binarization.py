import cv2


def binarize_with_preprocess(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    divided = cv2.divide(img, closed, 1, 255)

    return cv2.threshold(divided, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
