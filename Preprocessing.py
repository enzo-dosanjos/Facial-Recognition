import numpy as np
import cv2
from math import sqrt


def resize(img, size):

    img = cv2.resize(img, (size, size))

    return img


def crop(img, face_coord, scale_around_face):
    y1 = face_coord[0]
    x1 = face_coord[1]
    y2 = face_coord[2]
    x2 = face_coord[3]
    row, col, rgb = img.shape

    # verify if the frame is big enough to crop the face with 40% around it
    if (x2 + 2 * int(scale_around_face * x2) < 0.95 * col) and (y2 + 2 * int(scale_around_face * x2) < 0.95 * row) and (x1 - int(scale_around_face * x2)) > 0 and (y1 - int(scale_around_face * x2)) > 0:
        img = img[x1 - int(scale_around_face * x2): x1 + x2 + int(scale_around_face * x2), y1 - int(scale_around_face * x2): y1 + y2 + int(scale_around_face * x2)]

    return img

def normalize(array):
    # Normalize  image array
    array = (array.astype(np.float32) / 127.5) - 1

    return array

# brightness
