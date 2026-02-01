from PeopleLocalization.people_localization import PeopleLocalization
import cv2

class BoundingBoxWriter():

    def __init__(self):
        pass

    def write_for_people(self, frame, boxes):
        # drawing the filtered bb
        for box in boxes:
            # print(f'box: {box}')
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.putText(frame, 'person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 12), 2)
        return frame

    def write_for_painting(self, frame, single_coord, imgid):
        # drawing the filtered bb
        local = PeopleLocalization()
        if imgid == -1:
            label = 'painting'
        else:
            label = local.info(imgid)[0]
        x1, y1, x2, y2 = int(single_coord[0]), int(single_coord[1]), int(single_coord[2]), int(single_coord[3])
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return frame


class BoundingBoxReader():
    def __init__(self):
        pass

    def get_painting_bb(self, img, coord):
        x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])
        return img[y1:y2, x1:x2]