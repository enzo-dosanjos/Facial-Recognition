import json
import os
import cv2
import csv
import tensorflow as tf
from FaceDetection import frontalFaceDetection, profileFaceDetection
from Preprocessing import crop

BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

trainingData = os.path.join(dataFold, "DataSet")

# trainingData = os.path.join(dataFold, "images")

# create a dictionary with the peoples id and their associated name
identity = {}
identity_meta = csv.reader(open(os.path.join(trainingData, "identity_meta.csv"), encoding="utf8"))
for ind, row in enumerate(identity_meta):
    if ind == 0:
        continue
    identity[row[0]] = row[1]

# remove the person that I don't have in my reduced dataset (to remove later)
to_del = []
for id in identity.keys():
    if id not in os.listdir(trainingData):
        to_del.append(id)
for key in to_del:
    identity.pop(key, None)

# Limit GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')


# training data for tests
i = 0
for dirs in os.listdir(trainingData):
    path = os.path.join(trainingData, dirs)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith("png") or file.endswith("jpg"):
                # create the label files
                label_file = file.split('.')[0]+'.json'
                label_path = os.path.join(path, label_file)
                if not os.path.exists(label_path):
                    open(label_path, 'a').close()

                # read the image
                img = cv2.imread(os.path.join(path, file))

                # detect the frontal face
                face_coord = frontalFaceDetection(img)

                # detect profile face
                if len(face_coord) == 0:
                    face_coord = profileFaceDetection(img)

                # detect the best result for the face
                x1_big, y1_big, x2_big, y2_big = [0, 0, 0, 0]
                if len(face_coord) > 0:
                    x1_big, y1_big, x2_big, y2_big = face_coord[0]

                    for (x1, y1, x2, y2) in face_coord:

                        # find the biggest face among the detected ones
                        if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                            x1_big, y1_big, x2_big, y2_big = (x1, y1, x2, y2)

                # put data in label files
                print([x1_big, y1_big, x1_big + x2_big, y1_big + y2_big])
                label_dict = {'path': os.path.join(path, file), 'name': identity[dirs],
                              'face_coord': [int(x1_big), int(y1_big), int(x1_big + x2_big), int(y1_big + y2_big)]}
                json_object = json.dumps(label_dict, indent=4)
                with open(label_path, 'w') as label:
                    label.write(json_object)
    i+=1
    print(i)
"""
        # get the labels

        label = dirs.lower()
        output.append(label)

        # detect the frontal face
        face_coord = frontalFaceDetection(img)

        # detect profile face
        if len(face_coord) == 0:
            face_coord = profileFaceDetection(img)

        # crop the frame around the face
        if len(face_coord) > 0:
            biggest_rectangle = face_coord[0]
            x1_big, y1_big, x2_big, y2_big = face_coord[0]

            for (x1, y1, x2, y2) in face_coord:

                # find the biggest face among the detected ones
                if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                    biggest_rectangle = (x1, y1, x2, y2)

                # crop the frame around the  face
                scale_around_face = 1 / 100
                roi = crop(img, biggest_rectangle, scale_around_face)
        # cv2.imshow("img", roi)
        # cv2.waitKey(0)
        """
# should save all datas in a file, so it doesn't have to go through it each time
