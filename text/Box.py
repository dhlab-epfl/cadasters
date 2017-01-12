import numpy as np
import cv2
from scipy.spatial import distance as dist


class Box:
    box_counter = 0

    def __init__(self, cnt, box_pts, lbl_polygon):
        Box.box_counter += 1
        # Counter counts the number of boxes that have been created and assigns an unique id to each new created box
        self.box_id = Box.box_counter
        # current 4 corner points of the box, ordered clockwise
        self.box_pts = self.order_pts_clockwise(box_pts)
        # Points of the original box (before any transformation)
        self.original_box_pts = self.order_pts_clockwise(box_pts)
        # Contour of the elements inside the box
        self.cnt = cnt
        self.lbl_polygon = lbl_polygon
        self.box_ratio = self._comp_box_ratio()
        self.dimensions = [np.linalg.norm(self.box_pts[0] - self.box_pts[1]),
                           np.linalg.norm(self.box_pts[0] - self.box_pts[3])]
        self.box_area = np.product(self.dimensions)
        self.cnt_area = cv2.contourArea(self.cnt)
        self.hull = cv2.convexHull(self.cnt)
        self.hull_area = cv2.contourArea(self.hull)
        self.extent = self.cnt_area/self.box_area
        self.solidity = self.cnt_area/self.hull_area
        self.surf_ratio = self.hull_area/self.box_area
        self.defectmax = np.max(self._comp_defects())
        # Predicted number tuple(predicted number, confidence level)
        self.prediction_number = None

    def _comp_box_area(self):
        self.dimensions = [np.linalg.norm(self.box_pts[0] - self.box_pts[1]),
                           np.linalg.norm(self.box_pts[0] - self.box_pts[3])]
        self.box_area = np.product(self.dimensions)
        self.extent = self.cnt_area/self.box_area
        self.surf_ratio = self.hull_area/self.box_area

    def _comp_box_ratio(self):
        width_height = [np.linalg.norm(self.box_pts[0] - self.box_pts[1])/np.linalg.norm(self.box_pts[0] - self.box_pts[3]), # [A/B,
                        np.linalg.norm(self.box_pts[0] - self.box_pts[3])/np.linalg.norm(self.box_pts[0] - self.box_pts[1])]  # B/A]
        return np.max(width_height)

    def _comp_defects(self):
        hull = cv2.convexHull(self.cnt,returnPoints = False)
        defects = cv2.convexityDefects(self.cnt,hull)
        return np.float32(defects[:, 0, -1])/256.0  # Take only defect distances and change it to floating point values

    def order_pts_clockwise(self, pts):
        """
        order : top left, top right, bottom right, bottom left
        """
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:,0]), :]
        # grab the left-most and right-most points from the sorted x-coordinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their y-coordinates
        # so we can grab the top-left and bottom-left points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # now that we have the top-left coordinate, use it as an anchor to calculate the
        # Euclidean distance between the top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # return the coordinates in top-left, top-right, bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="int64")

    def zero_offset_box(self, offset):
        zero_offset_box = self.box_pts - offset
        self.box_pts = zero_offset_box
        return zero_offset_box

    def copy(self):
        return Box(self.cnt, self.original_box_pts, self.lbl_polygon)

    def expand_box(self, padding):
        """
        Expand box by padding the border with padding
        :param padding:
        :return: padded box
        """
        expanded_box = np.float64(self.box_pts.copy())
        # Compute unitary vectors
        # top edge (orientation : west)
        unit_top = expanded_box[0, :] - expanded_box[1, :]
        unit_top /= np.linalg.norm(unit_top)
        # side right (orientation : north)
        unit_side = expanded_box[1, :] - expanded_box[2, :]
        unit_side /= np.linalg.norm(unit_side)

        # expand rectangle (west-east)
        expanded_box[0, :] += padding*unit_top
        expanded_box[3, :] += padding*unit_top
        expanded_box[1, :] -= padding*unit_top
        expanded_box[2, :] -= padding*unit_top
        # expand rectangle (north-south)
        expanded_box[0, :] += padding*unit_side
        expanded_box[1, :] += padding*unit_side
        expanded_box[2, :] -= padding*unit_side
        expanded_box[3, :] -= padding*unit_side

        # Round values to have ints
        expanded_box = np.round(expanded_box)

        # Make sure none of the points are negatives
        negatives = (expanded_box < 0)
        if negatives.flatten().any():
            print('Box expansion - warning : one or more points have negative value. They will be set to zero.')
            expanded_box[negatives] = 0

        self.box_pts = np.int32(expanded_box)
        # Update dimension and area
        self._comp_box_area()
        return expanded_box


