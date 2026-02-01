import cv2
from skimage import morphology
import numpy as np
import skimage
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import math
from PaintingRectification.utils import Utils

u = Utils()

class RectangleRectifier():

    def get_point_for_homography(self,points, bb_points):
        output_points = []
        for bb_pt in bb_points:
            x1, y1 = bb_pt[0], bb_pt[1]
            distances = []
            for p in points:
                dist = u.calculateDistance(x1, y1, p[0], p[1])
                distances.append(dist)
            min_dist = distances.index(min(distances))
            res_point = points[min_dist]
            output_points.append(res_point)
        return output_points

    def rectify(self, img):
        cavia = deepcopy(img)

        #preprocessing of the image, with segmentation and opening
        res = u.preprocessing(cavia, True, True)

        #it gets the corners with Harris Corner Detection
        fast_img, points = u.harris(res)
        if points is None:
            print("no points detected")
            return img

        points_array = np.array(points)
        ind = np.lexsort((points_array[:, 1], points_array[:, 0]))

        #bounding box points
        bb_points = [[0,0],
                 [0, img.shape[0]],
                 [img.shape[1], 0],
                 [img.shape[1], img.shape[0]]]

        fast_img, points_array[ind]

        #it filters the harris points w.r.t the bounding box points
        res_points = self.get_point_for_homography(points, bb_points)

        sort_x = list((sorted(points_array[ind], key=lambda x: (x[0]))))
        sort_y = list((sorted(points_array[ind], key=lambda x: (x[1]))))

        #gets the minimum and maximum point of the four detected and filtered previously
        x_min = sort_x[0][0]
        x_max = sort_x[-1][0]
        y_min = sort_y[0][1]
        y_max = sort_y[-1][1]

        rect_points = [[x_min, y_min],
                       [x_min, y_max],
                       [x_max, y_min],
                       [x_max, y_max]]

        # rectification with WarpPerspective
        input_pts = np.float32(np.array(res_points))
        output_pts = np.float32(np.array(rect_points))
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        return out





