

class Control():
    def __init__(self):
        pass

    def is_valid_person_coordinate(self, person_coord, boxes_paint):
        x_center_person = (person_coord[0] + person_coord[2]) / 2
        y_center_person = (person_coord[1] + person_coord[3]) / 2
        not_valid = True

        for box_paint in boxes_paint:
            x1, y1, x2, y2 = box_paint[0], box_paint[1], box_paint[2], box_paint[3]
            if (x_center_person > x1 and x_center_person < x2 and y_center_person > y1 and y_center_person < y2):
                not_valid = False
        return not_valid
