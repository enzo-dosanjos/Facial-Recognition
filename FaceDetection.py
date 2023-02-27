import cv2
import os


projectDir = os.getcwd()
dataFile = os.path.join(projectDir, 'Data')
testFiles = os.path.join(dataFile, "Enzo")

def faceDetection(img):
    haarcascade_model = os.path.join(dataFile, "haarcascade_frontalface_default.xml")

    # Load the cascade function
    face_cascade = cv2.CascadeClassifier(haarcascade_model)

    # convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    scale_factor = 1.1  # resizing scale factor
    min_neighbor = 4  # number of neighbor each candidate rectangle should have to retain it
    detectFaceCoord = face_cascade.detectMultiScale(grey_img, scale_factor, min_neighbor)  # detectMultiScale take as
    # argument: a greyscale image, scale factor and min neighbor

    return detectFaceCoord
