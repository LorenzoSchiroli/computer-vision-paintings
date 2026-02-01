from io import BytesIO
import numpy as np
import cv2
import math
from copy import deepcopy
from matplotlib import pyplot as plt

class Utils():
    def __init__(self):
        pass

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def stamp(self, img, title):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

    def find_further(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmax()
        return array[idx], idx

    def indices_multiple_nearest(self, array, value):
        array = np.asarray(array)
        return [index for index in range(len(array)) if (np.abs(array[index] - value)) < 0.4]

    def get_extreme_y_points(self, img):
        indices = np.where(img == [0])  # first array = y, second array = x

        max_y = np.max(indices[0])
        multiple_args_max_y = np.argwhere(indices[0] == max_y)
        argmax_y = np.take(multiple_args_max_y, multiple_args_max_y.size // 2)
        max_coordinates = (indices[1][argmax_y], max_y)

        min_y = np.min(indices[0])
        multiple_args_min_y = np.argwhere(indices[0] == min_y)
        argmin_y = np.take(multiple_args_min_y, multiple_args_min_y.size // 2)
        min_coordinates = (indices[1][argmin_y], min_y)

        return max_coordinates, min_coordinates

    def get_extreme_x_points(self, img):
        indices = np.where(img == [0])  # first array = y, second array = x

        max_x = np.max(indices[1])
        multiple_args_max_x = np.argwhere(indices[1] == max_x)
        argmax_x = np.take(multiple_args_max_x, multiple_args_max_x.size // 2)
        max_coordinates = (max_x, indices[0][argmax_x])

        min_x = np.min(indices[1])
        multiple_args_min_x = np.argwhere(indices[1] == min_x)
        argmin_x = np.take(multiple_args_min_x, multiple_args_min_x.size // 2)
        min_coordinates = (min_x, indices[0][argmin_x])

        return max_coordinates, min_coordinates

    def get_middle_between_extreme(self, img, top, bottom, line_color):
        # assumption: the color is in greyscale
        # assumption2: the line has a tickness equal to 1
        indices = np.where(img == line_color[0])  # first array = y, second array = x
        potential_middle_y = abs(bottom[1] - top[1]) // 2
        middle_y, _ = self.find_nearest(indices[0], potential_middle_y)
        arg_middle_y = np.argwhere(indices[0] == middle_y)
        coordinates = (indices[1][arg_middle_y][0][0], middle_y)
        return coordinates

    def harris(self, img):
        try:
            gray = img
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        except:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        ret, dst = cv2.threshold(dst, 0.0001 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        points = list()
        for i in corners:
            x, y = i.ravel()
            points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
        return img, corners

    #it takes the points that are the extreme of the best approximation to the real perpendicular line
    #it takes also the middle coordinates of the referencing line for the contstraint to pass through the middle
    def get_perpendicular_line_points(self, pts, m, lht, rht, middle):

        returnList = list()

        left_pts = [lp for lp in pts if lp[0] < lht]
        left_pts.sort(key=lambda x: x[1])
        right_pts = [lp for lp in pts if lp[0] > rht]
        right_pts.sort(key=lambda x: x[1], reverse=True)

        #filtering points considering the expected_coefficient
        if m == math.inf:
            expected_coeff = 0
        else:
            expected_coeff = -1/m

        #left check
        coefficients_l = {}
        for i in range(len(left_pts)):
            if (left_pts[i][0] - middle[0]) != 0:
                coeff = (left_pts[i][1] - middle[1]) / (left_pts[i][0] - middle[0])
                pt = left_pts[i]
                key = ((pt[0], pt[1]))
                coefficients_l[key] = coeff

        #get candidates: multiple points that are nearest to the expected angular coefficient
        candidates_indices = self.indices_multiple_nearest(list(coefficients_l.values()), expected_coeff)

        if len(candidates_indices) == 0:
            raise Exception("Error while rectification")

        #get the indices and filter the left points along those indexes
        filtered = [left_pts[i] for i in candidates_indices]

        #select the further w.r.t. the middle point x-wise
        only_x_from_filtered = [v[0] for v in filtered]
        correct_x , index = self.find_further(only_x_from_filtered, middle[0])
        left_correct_point = [pt for pt in filtered if pt[0] == correct_x][0]

        #claculate the obtained angular coefficient
        obtained_coeff_left = (left_pts[index][1] - middle[1])/(left_pts[index][0] - middle[0])

        # right check
        coefficients_r = {}
        for i in range(len(right_pts)):
            if (right_pts[i][0] - middle[0]) != 0:
                coeff = (right_pts[i][1] - middle[1]) / (right_pts[i][0] - middle[0])
                pt = right_pts[i]
                key = ((pt[0], pt[1]))
                coefficients_r[key] = coeff

        # get candidates: multiple points that are nearest to the expected angular coefficient
        candidates_indices = self.indices_multiple_nearest(list(coefficients_r.values()), expected_coeff)

        if len(candidates_indices) == 0:
            raise Exception("Error while rectification")

        # get the indices and filter the left points along those indexes
        filtered = [right_pts[i] for i in candidates_indices]

        # select the further w.r.t. the middle point x-wise
        only_x_from_filtered = [v[0] for v in filtered]
        correct_x, index = self.find_further(only_x_from_filtered, middle[0])
        right_correct_point = [pt for pt in filtered if pt[0] == correct_x][0]

        # claculate the obtained angular coefficient
        obtained_coeff_right = (right_pts[index][1] - middle[1]) / (right_pts[index][0] - middle[0])


        return [left_correct_point, right_correct_point], obtained_coeff_left

    def segment_image(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except:
            gray = img

        img_segmented = img

        mask, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        rect_x = [0, img.shape[1]]
        rect_y = [0, img.shape[0]]

        y, x = np.indices(img.shape[:2])

        # threshold based on otsu's method
        img_segmented[mask < thresh] = 255
        # set everything outside the rectangle to 255
        img_segmented[(x < rect_x[0])] = 255
        img_segmented[(x > rect_x[1])] = 255
        img_segmented[(y < rect_y[0])] = 255
        img_segmented[(y > rect_y[1])] = 255

        return img_segmented

    def erode(self, img):
        kernel = np.ones((8, 8), np.uint8)
        img_eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        return img_eroded

    def open(self, img):
        kernel = np.ones((8, 8), np.uint8)
        img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img_opened

    def dilate(self, img):
        kernel = np.ones((10, 10), np.uint8)
        img_dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        return img_dilated

    def binarize(self, img):
        try:
            _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binarized
        except:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binarized

    def bilateral(self, img):
        try:
            bilateral = cv2.bilateralFilter(img, 8, 155, 155)
            return bilateral
        except:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            bilateral = cv2.bilateralFilter(gray, 8, 155, 155)
            return bilateral

    def calculateDistance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    def preprocessing(self, img, with_segmentation, with_opening):
        cavia = deepcopy(img)

        cavia = self.bilateral(cavia)

        if with_opening:
            cavia = self.open(cavia)

        if with_segmentation:
            cavia = self.segment_image(cavia)

        dilated_img = cv2.dilate(cavia, np.ones((15, 15), np.uint8))
        diff_img = cv2.absdiff(cavia, dilated_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        norm_img = cv2.erode(norm_img, np.ones((8, 8), np.uint8))
        cavia = cv2.Canny(norm_img, 10, 50)

        cavia = (255-cavia)

        return cavia