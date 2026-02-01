from PeopleLocalization.people_localization import PeopleLocalization
from PaintingRetrieval.image_retrieval import ImageMatching
from Interface.interface import Interface

class GUI():
    def __init__(self):
        pass

    def get_museum_map_with_people(self, localization, list_of_detected_paintings_correctly, correct_people_N):
        localization.imgs = list_of_detected_paintings_correctly
        localization.n_pers = correct_people_N
        print(f"Location Num People: {localization.n_pers}")
        return localization.localize()

    def create(self, video, map, paintings):
        local = PeopleLocalization()
        retrieval = ImageMatching()
        inter = Interface()
        complete_list = [(x[0], x[1], local.info(x[0]), retrieval.read_painting(x[0])) for x in paintings]
        return inter.show(video, map, complete_list)