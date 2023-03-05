from keras.models import load_model
import cv2
import os
import numpy as np
import json
from Preprocessing import resize, crop
from FaceDetection import frontalFaceDetection, profileFaceDetection


BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(os.path.join(dataFold, "face_recognition_cnn.h5"), compile=False)

# Load the labels
class_names = open(os.path.join(dataFold, "face_labels.txt"), "r").readlines()

# get the video capture
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error opening video stream or file")
    exit()

while cam.isOpened():
    # capture frame by frame
    ret, frame = cam.read()  # ret: bool that check if the frame is read correctly

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # detect the frontal face
    face_coord = frontalFaceDetection(frame)

    # detect profile face
    if len(face_coord) == 0:
        face_coord = profileFaceDetection(frame)

    # display the frame with the detected faces
    if len(face_coord) > 0:
        x1_big, y1_big, x2_big, y2_big = face_coord[0]

        for (x1, y1, x2, y2) in face_coord:
            # find the biggest face among the detected ones
            if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                x1_big, y1_big, x2_big, y2_big = (x1, y1, x2, y2)

            # crop the frame around the face
            scale_around_face = 1 / 100
            roi = crop(frame, (x1_big, y1_big, x2_big, y2_big), scale_around_face)

            # resize the image to 224px by 224px
            size = 224
            roi = resize(roi, size)
            image_array = np.array(roi, "uint8")
            roi = image_array.reshape(1, size, size, 3)
            roi = roi.astype('float32')
            roi /= 255

            # draw a rectangle around every detected faces for the video
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), color, thickness)

            # Predicts the model
            prediction = model.predict(roi)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Display the label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'{class_name[12:-3]} ({str(np.round(confidence_score * 100))[:-2]}%)', (x1_big, y1_big - 8),
                        font, 0.5, color, thickness, cv2.LINE_AA)

    # display the capture
    cv2.imshow("frame", frame)

    # stop if the x key is pressed
    if cv2.waitKey(1) == ord('x'):
        break

# release the capture
cam.release()
cv2.destroyAllWindows()
