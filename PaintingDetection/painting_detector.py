from detecto import core, utils, visualize
from torchvision import transforms
from detecto.core import Model
import numpy as np
import cv2

class PaintingDetector:

    def __init__(self, file_weights):
        self.file_weights = file_weights
        self.dataset = self.get_data()
        self.model = self.get_model()

    def get_data(self):
        augmentations = transforms.Compose([
            transforms.Lambda(lambda x: x[:, :, :3]),
            transforms.ToTensor(),
            utils.normalize_transform(),
        ])
        # dataset
        dataset = core.Dataset('images/', transform=augmentations)
        return dataset

    def get_model(self):
        return Model.load(self.file_weights, ['oval_painting', 'rectangle_painting'])

    def detect(self, image):
        predictions = self.model.predict(image)
        return predictions