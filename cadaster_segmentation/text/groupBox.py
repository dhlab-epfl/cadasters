import numpy as np
import cv2
from text.Box import Box


def min_distance_box(box1, box2):
    """
    Computes the minimal distance between 2 boxes (distance corner to corner)
    :param box1:
    :param box2:
    :return: minimal distance
    """
    # 4 x pt1, 4 x pt2, ..., 4 x pt4
    stack_box2pts = np.tile(box2.box_pts[0], (4, 1))
    for pt in box2.box_pts[1:]:
        stack_box2pts = np.vstack((stack_box2pts, np.tile(pt, (4, 1))))

    # 4 x (pt 1,2,3,4)
    rep_box1pts = np.tile(box1.box_pts, (4, 1))

    # distances between points
    d = np.sqrt(np.sum((np.float64(rep_box1pts) - np.float64(stack_box2pts))**2, axis=1))

    return np.min(d)
# ----------------------------------------------------------------------


def group_box_with_lbl(list_boxes, false_boxes, maximum_distance):
    """
    Groups boxes that are close to form bigger ones

    :param list_boxes: List of Box objects to group, that is updated with new boxes
    :param false_boxes: list of possible false boxes
    :param maximum_distance: maximum distance between boxes to accept grouping
    :return: list of Box objects that have been grouped
    """

    grouped_boxes = list()

    for box1 in list_boxes:
        boxes_to_group = (None, None)
        dmin = 9999
        for box2 in list_boxes:

            # For cases where one of the box has no lbl_polygon (XOR : only true when two terms are different)
            if bool(box1.lbl_polygon) != bool(box2.lbl_polygon):
                d = min_distance_box(box1, box2)
                if d < dmin:
                    dmin = d
                    boxes_to_group = (box1, box2)

            # only compare two boxes that are in the same labeled zone
            elif box1 != box2 and box1.lbl_polygon == box2.lbl_polygon:
                d = min_distance_box(box1, box2)
                if d < dmin:
                    dmin = d
                    boxes_to_group = (box1, box2)

        if dmin < maximum_distance and (boxes_to_group[0] not in false_boxes or boxes_to_group[1] not in false_boxes):
            # put both cnt together and find minimum area rect
            cnt = np.vstack((boxes_to_group[0].cnt, boxes_to_group[1].cnt))
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # make a bigger box with both cnt
            new_box = Box(cnt, box, box1.lbl_polygon)
            # add new bigger box remove individual boxes
            list_boxes.append(new_box)
            grouped_boxes.append(new_box)
            list_boxes.remove(boxes_to_group[0])
            list_boxes.remove(boxes_to_group[1])

    return grouped_boxes
# ---------------------------------------------------------------------


def group_box_with_isolates(trueBoxes, falseBoxes, maximum_distance):
    """
    Groups true Box with false Box to group for example too small (false) boxes with bigger (true) ones

    :param trueBoxes: list of possible true Box object
    :param falseBoxes: list of possible false Box objects
    :param maximum_distance: maximum distance between boxes to accept grouping
    :return:
    """

    grouped_boxes = list()

    for trueb in trueBoxes:
        boxes_to_group = (None, None)
        dmin = 9999

        for falseb in falseBoxes:
            d = min_distance_box(trueb, falseb)
            if d < dmin:
                dmin = d
                boxes_to_group = (trueb, falseb)

        if dmin < maximum_distance:
            # put both cnt together and find minimum area rect
            cnt = np.vstack((boxes_to_group[0].cnt, boxes_to_group[1].cnt))
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # make a bigger box with both cnt
            new_box = Box(cnt, box, trueb.lbl_polygon)
            # add new bigger box remove indivudual boxes
            trueBoxes.append(new_box)
            grouped_boxes.append(new_box)
            trueBoxes.remove(boxes_to_group[0])
            falseBoxes.remove(boxes_to_group[1])

    return grouped_boxes
# -----------------------------------------------------------------------


