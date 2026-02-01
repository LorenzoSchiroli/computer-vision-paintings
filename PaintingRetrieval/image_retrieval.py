import cv2
import numpy as np
import json
import os
import math

class ImageMatching:

    def __init__(self):
        self.images_path = "PaintingRetrieval/paintings_db"
        self.images_features_path = "PaintingRetrieval/features.json"
        self.alg = cv2.ORB_create(nfeatures=250)
        self.threshold = 35
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.images = None
        #self.packing()
        self.unpacking()
        self.old_paintings = []

    # compute keypoints and descriptors with ORB
    def orb(self, img):
        kp, dsc = self.alg.detectAndCompute(img, None)
        return kp, dsc

    # extract features from the painting
    def extract_features(self, img):
        img = cv2.resize(img, (400,400))
        return self.orb(img)

    # serialize paintings features to be able to store inside JSON file
    def json_serialize(self, tup):
        kp, dsc = tup
        kp_ser = []
        for i in kp:
            kp_ser.append((i.angle, i.class_id, i.octave, i.pt, i.response, i.size))
        dsc_ser = dsc.tolist()
        return kp_ser, dsc_ser

    # deserialize paintings features to be able to extract them from JSON file
    def json_deserialize(self, tup):
        kp_ser, dsc_ser = tup
        kp = []
        for i in kp_ser:
            kp.append(cv2.KeyPoint(_angle=i[0], _class_id=i[1], _octave=i[2], x=i[3][0], y=i[3][1], _response=i[4], _size=i[5]))
        dsc = np.array(dsc_ser).astype('uint8')
        return kp, dsc

    # compute features of all database paintings and store them in JSON file
    def packing(self):
        files = [os.path.join(self.images_path, p) for p in sorted(os.listdir(self.images_path))]
        images_json = {}
        for f in files:
            img_id = os.path.splitext(os.path.basename(os.path.normpath(f)))[0]
            img = cv2.imread(f)
            images_json[img_id] = self.json_serialize(self.extract_features(img))
        with open(self.images_features_path, 'w') as images_features:
            json.dump(images_json, images_features)

    # extract features of all database paintings from JSON file
    def unpacking(self):
        with open(self.images_features_path) as images_features:
            images_json = json.load(images_features)
        images = {}
        for k, v in images_json.items():
            images[k] = self.json_deserialize(v)
        self.images = images
        
    # returns the rank
    # compare target painting keypoints with all the database paintings keypoints and count strong couples
    def matching(self, img, top=10):  # higher score higher similarity
        kp1, dsc1 = self.extract_features(img)
        dist = {}
        for k, v in self.images.items():
            kp2, dsc2 = v
            matches = self.bf.match(dsc1, dsc2)
            count = 0
            for x in matches:
                if x.distance < self.threshold:
                    count = count + 1
            dist[k] = count
        dist = sorted(dist.items(), key=lambda x: -x[1])
        return dist[:top]
        
    # from painting index compute painting path
    def read_painting(self, index):
        return cv2.imread(self.images_path + "/" + index + ".png")

    # compare the paintings' bounding box of the last frame with the current ones in order to track them
    def remember(self, current_paintings):
        
        if len(self.old_paintings) > 0:
            for i, q in enumerate(current_paintings):
                lowest_dist = 1000000
                last = None
                for ql in self.old_paintings:
                    dist = math.hypot(q[2][0]-ql[2][0], q[2][1]-ql[2][1])
                    if lowest_dist > dist:
                        lowest_dist = dist
                        last = ql
                if lowest_dist < 25 and last[1] > q[1]:
                    current_paintings[i] = (last[0], last[1], q[2], q[3], q[4])
        self.old_paintings = current_paintings
        
        return self.old_paintings

    def retrieve_image_in_db(self, image):
        rank = ImageMatching().matching(image)
        #print(f"Ranking: {rank}")
        if rank[0][1] > 2:  # 2 or greater
            return rank[0][0], rank[0][1], rank
        else:
            return -1, None, rank

