import cv2
import numpy as np

class Interface:

    def __init__(self):
        self.merge = None
        
    # compute output interface to display the results of current frame
    def show(self, video, map, paint_id_score):

        #resize video
        ratio = video.shape[1] / video.shape[0]
        height = 500
        video = cv2.resize(video, (int(height * ratio), height))

        #resize video
        ratio = map.shape[1] / map.shape[0]
        height = 400
        map = cv2.resize(map, (int(height * ratio), height))

        padding = np.full((video.shape[0] - map.shape[0], map.shape[1], 3), 255, dtype=np.uint8)
        map_padding = np.concatenate((map, padding), axis=0)
        top = np.concatenate((video, map_padding), axis=1)

        # bottom text
        mignH = 150
        whitH = 150
        blockW = 150

        segW = 10
        bottom = np.full((60, top.shape[1], 3), 255, dtype=np.uint8)

        cv2.putText(img=bottom, text="DETECTED PAINTINGS:", org=(5, 40), color=(0, 0, 0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=3)
        stripe = np.full((mignH + whitH, segW, 3), 255, dtype=np.uint8)
        for i, painting in enumerate(paint_id_score):
            index, score, info, mignature = painting
            mignature = cv2.resize(mignature, dsize=(blockW, mignH), interpolation=cv2.INTER_CUBIC)
            white = np.full((whitH, blockW, 3), 255, dtype=np.uint8)
            seg = np.full((mignH + whitH, segW, 3), 255, dtype=np.uint8)
            try:
                cv2.putText(img=white, text=f"{info[0]}", org=(5, 20), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=2)
                cv2.putText(img=white, text=f"{info[1]}", org=(5, 40), color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, thickness=1)
            except SystemError:
                pass
            cv2.putText(img=white, text="Score: " + str(score), org=(5, 65), color=(0, 0, 0),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, thickness=1)

            block = np.concatenate((mignature, white), axis=0)
            stripe = np.concatenate((stripe, block, seg), axis=1)
        if bottom.shape[1] - stripe.shape[1] > 0:
            padding = np.full((stripe.shape[0], bottom.shape[1] - stripe.shape[1], 3), 255, dtype=np.uint8)
            stripe = np.concatenate((stripe, padding), axis=1)
        elif bottom.shape[1] - stripe.shape[1] < 0:
            stripe = stripe[:, :bottom.shape[1], :]
        bottom = np.concatenate((bottom, stripe), axis=0)
        final = np.concatenate((top, bottom), axis=0)
        
        return final
        