from detecto import core, utils, visualize
from torchvision import transforms
from detecto.core import Model

class PeopleDetector():

    def __init__(self):
        self.model = Model()

    def get_model(self):
        return self.model

    def detect(self, image):
        predictions = self.model.predict(image)
        return predictions