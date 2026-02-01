
from detecto import core

class Train:
    def __init__(self):
        pass

    def train(self, dataset, file, labels):
        model = core.Model(labels)
        model.fit(dataset)
        model.save(file)