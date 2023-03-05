import cv2
import os

BASE_DIR = os.getcwd()
dataFile = os.path.join(BASE_DIR, 'Data')
testFiles = os.path.join(dataFile, "Enzo")

HaarFiles = os.path.join(dataFile, "Haarcascade")


def frontalFaceDetection(img):
    haarcascade_model = os.path.join(HaarFiles, "haarcascade_frontalface_alt.xml")

    # Load the cascade function
    face_cascade = cv2.CascadeClassifier(haarcascade_model)

    # convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    scale_factor = 1.1  # resizing scale factor
    min_neighbor = 5  # number of neighbor each candidate rectangle should have to retain it
    front_face_coord = face_cascade.detectMultiScale(grey_img, scale_factor, min_neighbor)

    return front_face_coord


def profileFaceDetection(img):
    haarcascade_model = os.path.join(HaarFiles, "haarcascade_profileface.xml")

    # Load the cascade function
    face_cascade = cv2.CascadeClassifier(haarcascade_model)

    # convert image to greyscale
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    scale_factor = 1.3  # resizing scale factor
    min_neighbor = 5  # number of neighbor each candidate rectangle should have to retain it
    profile_face_coord = face_cascade.detectMultiScale(grey_img, scale_factor, min_neighbor)

    return profile_face_coord
