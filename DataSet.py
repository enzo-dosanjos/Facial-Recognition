import json
import os
import cv2
import csv
import random
import tensorflow as tf
from FaceDetection import frontalFaceDetection, profileFaceDetection
from Preprocessing import resize, crop

BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")
dataSet = os.path.join(dataFold, "DataSet")
VGG2 = os.path.join(dataSet, "VGG2")

# create a dictionary with the peoples id and their associated name
identity = {}
identity_meta = csv.reader(open(os.path.join(dataSet, "identity_meta.csv"), encoding="utf8"))
for ind, row in enumerate(identity_meta):
    if ind == 0:
        continue
    identity[row[0]] = row[1]

# remove the person that I don't have in my reduced dataset (to remove later)
to_del = []
for id in identity.keys():
    if id not in os.listdir(VGG2):
        to_del.append(id)
for key in to_del:
    identity.pop(key, None)

# Limit GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

# remove the old files in TestData and TrainingData
dataFiles = ["TestData", "TrainingData"]
for dataFile in dataFiles:
    dataFilePath = os.path.join(dataSet, dataFile)
    for dirs in os.listdir(dataFilePath):
        path = os.path.join(dataFilePath, dirs)
        for file in os.listdir(path):
            if file.endswith("png") or file.endswith("jpg") or file.endswith("json"):
                os.remove(os.path.join(path, file))
    print(f"files in {dataFile} removed")

# make the training and test data
i = 0
for dirs in os.listdir(VGG2):
    path = os.path.join(VGG2, dirs)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith("png") or file.endswith("jpg"):
                # read the image
                img = cv2.imread(os.path.join(path, file))

                # detect the frontal face
                face_coord = frontalFaceDetection(img)

                # detect profile face
                if len(face_coord) == 0:
                    face_coord = profileFaceDetection(img)

                # detect the best result for the face
                row, col, rgb = img.shape
                if len(face_coord) > 0:
                    x1_big, y1_big, x2_big, y2_big = face_coord[0]

                    for (x1, y1, x2, y2) in face_coord:
                        # find the biggest face among the detected ones
                        if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                            x1_big, y1_big, x2_big, y2_big = (x1, y1, x2, y2)

                else:
                    # take the whole image if no face is detected
                    x1_big, y1_big, x2_big, y2_big = [0, 0, col, row]

                # crop the frame around the face
                scale_around_face = 1 / 100
                img = crop(img, (x1_big, y1_big, x2_big, y2_big), scale_around_face)

                # resize the image to 224px by 224px
                size = 224
                img = resize(img, size)

                # name the label files
                label_file = identity[dirs].replace("\"", "") + "_" + file.split('.')[0] + '.json'

                # distribute 80% of images into training images
                if random.random() < 0.80:
                    trainingData = os.path.join(dataSet, "TrainingData")
                    image_path = os.path.join(trainingData, "Images")
                    # write the image
                    os.chdir(image_path)
                    cv2.imwrite(identity[dirs].replace("\"", "") + "_" + file, img)
                    # get the path for the label
                    label_path = os.path.join(trainingData, "Labels", label_file)

                # distribute 20% of images into training images
                else:
                    testData = os.path.join(dataSet, "TestData")
                    image_path = os.path.join(testData, "Images")
                    # write the image
                    os.chdir(image_path)
                    cv2.imwrite(identity[dirs].replace("\"", "") + "_" + file, img)
                    # get the path for the label
                    label_path = os.path.join(testData, "Labels", label_file)

                # create the label file if it does not exist
                if not os.path.exists(label_path):
                    open(label_path, 'a').close()

                # put data in label files
                label_dict = {'path': os.path.join(image_path, file), 'name': identity[dirs].replace("\"", ""),
                              'face_coord': [int(x1_big), int(y1_big), int(x1_big + x2_big), int(y1_big + y2_big)]}
                json_object = json.dumps(label_dict, indent=4)
                with open(label_path, 'w') as label:
                    label.write(json_object)
    i += 1
    print(f"person {i} done")
