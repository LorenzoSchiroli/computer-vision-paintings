import numpy as np
from copy import deepcopy
import cv2
import math
from PaintingRectification.utils import Utils
import matplotlib.pyplot as plt

u = Utils()

#coordinates of the oval common pattern
oval_coordinates = [[400, 104], [400, 695], [186, 368], [612, 368]]

class OvalRectifier():
    def __init__(self):
        pass

    #get points from the oval painting in the image for compute the homography
    def get_point_for_homography(self, original_image, modified_image):
        #it gets the top and the bottom point
        bottom_point, top_point = u.get_extreme_y_points(modified_image)
        print(top_point, bottom_point)

        #it draws a line which connects the top and the bottom point
        line_color = (200, 0, 0)
        with_line_A = cv2.line(modified_image, (top_point[0], top_point[1]), (bottom_point[0], bottom_point[1]),
                                  line_color, 1)

        #it computes the middle point of the line A
        middle = u.get_middle_between_extreme(with_line_A, top_point, bottom_point, line_color)

        #it gets the corners with Harris Corner Detection
        output, pts = u.harris(original_image)

        #It calculates the slope of the line A
        if top_point[0] - bottom_point[0] != 0:
            m = (top_point[1] - bottom_point[1]) / (top_point[0] - bottom_point[0])
        else:
            m = math.inf

        #It gets the remaining points of the perpendicular one w.r.t. the line A
        perp, obtained_coeff = u.get_perpendicular_line_points(pts, m, top_point[0], bottom_point[0], middle)
        modified_image = cv2.line(with_line_A, (perp[0][0], perp[0][1]), (perp[1][0], perp[1][1]), line_color, 1)

        left_point = [perp[0][0], perp[0][1]]
        right_point = [perp[1][0], perp[1][1]]
        top_point = [top_point[0], top_point[1]]
        bottom_point = [bottom_point[0], bottom_point[1]]

        points = np.float32([top_point, bottom_point, left_point, right_point])
        return points

    def rectify(self, img):
        # I pick the oval pattern

        #Reads the oval pattern
        oval = cv2.imread('PaintingRectification/OvalPaintingRectification/Utils/OvalPattern.jpg', flags=cv2.IMREAD_GRAYSCALE)
        original = deepcopy(img)
        cavia = deepcopy(img)

        #preprocessing of the image
        res = u.preprocessing(cavia, False, False)
        img = deepcopy(res)

        try:
            #gets the points for homography
            obtained_points = self.get_point_for_homography(img, res)
        except Exception as e:
            return original

        original_oval = u.binarize(oval)

        correct_points = np.float32(oval_coordinates)

        # rectification with WarpPerspective
        M = cv2.getPerspectiveTransform(obtained_points, correct_points)
        bc = original[0][0]
        border_color = np.array((int(bc[0]),int(bc[1]),int(bc[2])))
        dst = cv2.warpPerspective(original, M, (oval.shape[1], oval.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[int(bc[0]), int(bc[1]), int(bc[2])])
        h1, w1, _ = original.shape
        dst = cv2.resize(dst,(w1,h1))

        result = dst
        return result

