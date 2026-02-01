import cv2
import torch
import numpy as np
import os
from PaintingRectification.RectanglePaintingRectification.main import RectangleRectifier
from PaintingRetrieval.image_retrieval import ImageMatching
from PeopleLocalization.people_localization import PeopleLocalization
from PaintingRectification.OvalPaintingRectification.main import OvalRectifier
from Utils import GUI
from Controls import Control
from BoundingBox import BoundingBoxWriter
from BoundingBox import BoundingBoxReader
import matplotlib.pyplot as plt
from PaintingDetection.painting_detector import PaintingDetector
from PeopleDetection.people_detector import PeopleDetector

bb_writer = BoundingBoxWriter()
bb_reader = BoundingBoxReader()

#It detects the painting in the image and classifies them with two labels
def painting_detection(img, painting_detector):
    painting_predictions = painting_detector.detect(img)
    labels_paint, boxes_paint, scores_paint = painting_predictions
    print('Scores detected paintings:', f'{scores_paint}')
    print('Label associated:', f'{labels_paint}')

    # Here we filter the detections of the paintings using the score threshold 30%
    all_paintings_coordinates = None
    if scores_paint.nelement() != 0:
        mask = np.argwhere(scores_paint > 0.30)
        all_paintings_coordinates = boxes_paint[mask][0]
        scores_paint = scores_paint[mask][0]
        boxes_paint = boxes_paint[mask][0]

    print('Filtered scores detected paintings' + ': ' + f'{scores_paint}')
    return all_paintings_coordinates, boxes_paint, labels_paint

#It detects the people in the image
def people_detection(img, people_detector, boxes_paint):
    ctr = Control()
    predictions = people_detector.detect(img)
    labels_person, boxes_person, scores_person = predictions

    if scores_person.nelement() != 0:
        labels_person2 = np.asarray(labels_person)
        mask = np.argwhere(scores_person > 0.85)
        mask2 = np.argwhere(labels_person2[mask][0] == 'person')
        mask2 = torch.from_numpy(mask2)
        mask2 = np.reshape(mask2, (1, -1))
        scores_person = scores_person[mask2][0]
        boxes_person = boxes_person[mask2][0]
    print(f"People number: {boxes_person.size()[0]}")

    valid_people = np.empty((0, 4))
    for person in boxes_person:
        if ctr.is_valid_person_coordinate(person, boxes_paint):
            valid_people = np.append(valid_people, np.expand_dims(person.numpy(), axis=0), axis=0)
    correct_people_N = valid_people.shape[0]
    img = bb_writer.write_for_people(img, valid_people)
    return correct_people_N, img

#It rectifies the image according to the label
def rectify(label, img, x1, y1, x2, y2):
    if label == "rectangle_painting":
        return RectangleRectifier().rectify(img)

    # Oval paintings case
    if label == "oval_painting":
        rectified = OvalRectifier().rectify(img)
        return cv2.resize(rectified, (x2 - x1, y2 - y1))
    return None

#It retrieves the image from Database with ORB
def retrieve(img, single_coord,center, retriever):
    correct_painting = retriever.retrieve_image_in_db(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    correct_painting_index = correct_painting[0]
    if correct_painting_index == -1:
        print(f"Painting detected: not detected")
        return (-1, -1, center, single_coord, correct_painting[2])
    else:
        print(f"Painting detected: index {correct_painting_index}")
        return (correct_painting[0], correct_painting[1], center, single_coord, correct_painting[2])

#Localization and museum_map extraction
def localize(correct_people_N, localization, list_of_detected_paintings_correctly, gui):
    print(f'Number of people : {correct_people_N}')
    only_index = [x[0] for x in list_of_detected_paintings_correctly if x[0] != -1]
    museum_map = gui.get_museum_map_with_people(localization, only_index, correct_people_N)
    return museum_map

#main pipeline
def pipeline(img, retriever, localization, show_rectification):

    #utils classes
    gui = GUI()
    painting_file_weights = "PaintingDetection/painting_weights.pth"
    painting_detector = PaintingDetector(painting_file_weights)
    people_detector = PeopleDetector()

    #PAINTING DETECTION
    paintings_coordinates, boxes_paint, labels_paint = painting_detection(img, painting_detector)

    #PEOPLE DETECTION
    correct_people_N, img = people_detection(img, people_detector, boxes_paint)

    #RECTIFICATION, RETRIEVAL and LOCALIZATION
    list_of_detected_paintings_correctly = []
    museum_map = cv2.imread("PeopleLocalization/map.png")
    if paintings_coordinates != None:
        for index, box in zip(range(boxes_paint.nelement()), boxes_paint):

            print(f'Painting type: {labels_paint[index]}')

            # extract the painting
            single_coord = box
            x1, y1, x2, y2 = int(single_coord[0]), int(single_coord[1]), int(single_coord[2]), int(single_coord[3])
            center = (abs(x1 + x2) / 2, abs(y1 + y2) / 2)
            painting = bb_reader.get_painting_bb(img, single_coord)

            # RECTIFICATION
            rectified = rectify(labels_paint[index], painting, x1, y1, x2, y2)

            if show_rectification:
                img[y1:y2, x1:x2] = rectified

            # RETRIEVE
            if rectified is not None:
               list_of_detected_paintings_correctly.append(retrieve(rectified, single_coord, center, retriever))
        list_of_detected_paintings_correctly = retriever.remember(list_of_detected_paintings_correctly)

        #LOCALIZATION
        museum_map = localize(correct_people_N, localization, list_of_detected_paintings_correctly,gui)

        #PREPARING THE OUTPUT INTERFACE
        for paint in list_of_detected_paintings_correctly:
            img = bb_writer.write_for_painting(img, paint[3], paint[0])
            if paint[0] != -1:
                print(f"Best painting of the last frames: {paint[0]}")
                print(f"Rank of current frame: {paint[4]}")
    #result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img
    result = cv2.resize(result, (640, 480))
    if len(list_of_detected_paintings_correctly) > 0:
        list_of_detected_paintings_correctly = [x for x in list_of_detected_paintings_correctly if x[0] != -1]

    return gui.create(result, museum_map, list_of_detected_paintings_correctly)

#iterates in a video and processes the pipeline frame by frame in an output video
def executeVideo(media_path, out_name, show_rectification):
    cap = cv2.VideoCapture(media_path)

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input < 15:
        fps_input = 15

    _, input_filename = os.path.split(media_path)
    out_path = out_name+'.avi' #os.path.join(app.config['VIDEO_OUTPUTS_FOLDER'], media_path)
    out = None

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("number of frames: ", number_of_frames)

    retriever = ImageMatching()
    localization = PeopleLocalization()

    while(cap.isOpened() and number_of_frames > 0):
        ret, frame = cap.read()

        print("**** FRAME NUMBER {number} ****".format(number=number_of_frames))

        if frame is None:
            print("Frame not readable")
            pass

        if ret:
            result = pipeline(frame, retriever, localization,show_rectification)
            height, width, layers = result.shape
            size = (width, height)
            if out is None:
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps_input, size)
            out.write(result)
        else:
            print("Frame not readable")

        #decrease the number of remaining frames
        number_of_frames = number_of_frames -1

    cap.release()
    out.release()
    return out_path

#takes an image in input and processes it. It shows the result in the output
def executeImage(media_path, show_rectification):
    retriever = ImageMatching()
    localization = PeopleLocalization()
    img = cv2.imread(media_path, cv2.COLOR_BGR2RGB)
    if img is not None:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("input image")
        plt.show()
        result = pipeline(img, retriever, localization,show_rectification)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("result")
        plt.show()
    
