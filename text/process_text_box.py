import numpy as np
import cv2


def crop_box(box, image, fill=True):
    """
    Crops box and returns the image in the box

    :param box: Box object
    :param image: full image to crop
    :param fill: boolean to fill the outside of the box with max_value or not
    :return: cropped image, coordinates of the crop
    """

    xmin = min(box.box_pts[:, 0])
    xmax = max(box.box_pts[:, 0])
    ymin = min(box.box_pts[:, 1])
    ymax = max(box.box_pts[:, 1])
    box.zero_offset_box(np.array([xmin, ymin]))
    crop_img = image[ymin:ymax+1, xmin:xmax+1].copy()

    # Fill cropped image's borders
    if fill:
        crop_img = fill_outside_box(crop_img, box)

    return crop_img, (xmin, xmax, ymin, ymax)
# ------------------------------------------------------


def fill_outside_box(image, box):
    """
    Fills the white/black border of cropped image with max_value

    :param image: cropped image to be filled
    :param box: Box object that crops the object
    :return: filled image
    """
    # Fill pixels outside the box with maximal value
    mask_box = np.zeros(image.shape, 'int8')
    cv2.fillPoly(mask_box, [box.box_pts], 1)
    mask_box_inv = mask_box < 1
    image = np.uint8(mask_box*image)
    max_val = np.max(image)
    image[mask_box_inv] = max_val

    return image
# ------------------------------------------------------


def find_orientation(blob):
    """
    Uses PCA to find the orientation of binary blob image
    :param blob: binary blob image
    :return: center point, eignevectors, angle of orientation (in degrees)
    """
    # Find contours
    tmp, contours, tmp = cv2.findContours(blob.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find orientation with PCA
    cnt = contours[0].reshape(contours[0].shape[0], 2)  # must be of size nx2
    mean, eigvect = cv2.PCACompute(np.float32(cnt), mean=np.array([]))

    center = mean[0]
    diagonal = eigvect[0]
    angle = np.arctan2(diagonal[1], diagonal[0])  # orientation in radians
    angle *= (180/np.pi)

    if diagonal[0] > 0:  # 1st and 4th cadran
        angle = angle
    elif diagonal[0] < 0:  # 2nd and 3rd cadran
        angle = 90 - angle

    return center, eigvect, angle
# -----------------------------------------------------------


def crop_object(bin_img):
    """
    Crops binary object in image

    :param bin_img: binary image
    :return: cropped image, coordinates of crop
    """
    # Get contours
    tmp, cnt_rotated, tmp = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assemble all points
    pts = np.array([tuple(p[0]) for c in cnt_rotated for p in c])
    y, x, h, w = cv2.boundingRect(pts)
    # Add an extra border to the bounding rect
    nborder = 1
    y = max([y-nborder, 0])
    x = max([x-nborder, 0])
    h = min([bin_img.shape[1], x+h+2*nborder]) - x
    w = min([bin_img.shape[0], y+w+2*nborder]) - y

    return bin_img[x:x+w, y:y+h], (y, x, h, w)
