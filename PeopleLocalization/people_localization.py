from collections import Counter
import cv2
import pandas

class PeopleLocalization():

    def __init__(self):
        self.data = pandas.read_csv("PeopleLocalization/data.csv")
        self.rooms_position = pandas.read_csv("PeopleLocalization/rooms_position.csv")
        self.rooms_position.index += 1
        self.imgs = []
        self.clear = cv2.imread("PeopleLocalization/map.png")
        self.current_room = None
        self.current_center = None
        self.n_pers = 0

    # extract painting informations given the index
    def info(self, img_id):
        info = self.data.loc[self.data['Image'] == str(img_id) + ".png"]
        info = info.values[0].tolist()  # Title,Author,Room,Image
        return info

    # extract room given the painting index
    def extract_room(self, img_id):
        return self.info(img_id)[2]

    # retrive the coordinates (in map.png) of a room
    def extract_area(self, room):
        area = self.rooms_position.loc[room]
        area = area.values.tolist()
        return area

    # compute the most popular room goven a list of paintings
    def room_majority(self):
        votes = []
        for img in self.imgs:
            votes.append(self.extract_room(img))
        self.current_room = Counter(votes).most_common(1)[0][0]
        area = self.extract_area(self.current_room)
        self.current_center = (area[0] + area[1]) // 2, (area[2] + area[3]) // 2

    # draw a circle in the corrent room
    def draw_circle(self, img):
        cv2.circle(img, self.current_center, radius=50, color=(0, 0, 255), thickness=10)

    # draw the current number of people
    def draw_PN(self, img):
        cv2.putText(img=img, text="People count: "+str(self.n_pers), org=(350, 350), color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=4)

    # localize people
    def localize(self):
        dirty = self.clear.copy()
        self.draw_PN(dirty)
        if len(self.imgs) == 0 and self.current_room == None:
            return dirty
        if len(self.imgs) > 0:
            self.room_majority()
        self.draw_circle(dirty)
        return dirty
