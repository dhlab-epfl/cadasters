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


def find_text_orientation_from_box(box, full_img):
    # Expand box
    box.expand_box(padding=2)

    img_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

    # Crop
    crop_img_gray, (xmin, xmax, ymin, ymax) = crop_box(box, img_gray)

    # Binarize to have the general shape so that we can dilate it as a
    # blob and find the orientation of the blob

    # Binarization
    blur = cv2.GaussianBlur(crop_img_gray, (3, 3), 0)
    ret, binary_crop = cv2.threshold(blur, 0, np.max(crop_img_gray), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_crop = np.uint8(255 * (binary_crop < 1))
    # Morphology _ dilation
    dilated = cv2.dilate(inv_crop, np.ones((5, 5), np.uint8))
    # Find orientation with PCA
    center, eigvect, angle = find_orientation(dilated)

    return center, eigvect, angle


def crop_with_margin(crop_coords, full_img, margin=2):
    """

    :param box_coords: (xmin, xmax, ymin, ymax)
    :param full_img:
    :param margin:
    :return:
    """

    xmin, xmax, ymin, ymax = crop_coords
    shape = full_img.shape[:2]

    xmin = np.maximum(0, xmin - margin)
    xmax = np.minimum(shape[1], xmax + margin)
    ymin = np.maximum(0, ymin - margin)
    ymax = np.minimum(shape[0], ymax + margin)

    crop = full_img[ymin:ymax+1, xmin:xmax+1].copy()

    return crop


def custom_bounding_rect(box_points):
    xmin = np.min(box_points[:, 0])
    xmax = np.max(box_points[:, 0])
    ymin = np.min(box_points[:, 1])
    ymax = np.max(box_points[:, 1])

    return np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])


def add_margin_to_rectangle(rect_coords, margin):
    margin_array = np.array([[-margin, -margin], [-margin, margin], [margin, margin], [margin, -margin]])
    new_coords = rect_coords + margin_array

    return new_coords


def check_validity_points(points, max_shape):
    points[:, 0] = np.maximum(points[:, 0], np.zeros(points[:, 0].shape))
    points[:, 0] = np.minimum(points[:, 0], max_shape[1] * np.ones(points[:, 0].shape))
    points[:, 1] = np.maximum(points[:, 1], np.zeros(points[:, 1].shape))
    points[:, 1] = np.minimum(points[:, 1], max_shape[0] * np.ones(points[:, 1].shape))

    return points


def get_crop_indexes_from_points(points):
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])

    return xmin, xmax, ymin, ymax
