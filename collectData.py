import cv2
import os
from FaceDetection import frontalFaceDetection, profileFaceDetection
from Preprocessing import crop, resize


BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

name = str(input("Enter your Name: ")).lower()
newDataFold = os.path.join(dataFold, name)

# create the new data directory if it does not exist
if not os.path.exists(newDataFold):
    os.mkdir(newDataFold)

# get the image
# img = cv2.imread(os.path.join(newData, "Enzo_0001.jpg"), 1)

# get the video capture
cam = cv2.VideoCapture(0)

count = 0

if not cam.isOpened():
    print("Error opening video stream or file")
    exit()

while cam.isOpened():
    # capture frame by frame
    ret, frame = cam.read()  # ret: bool that check if the frame is read correctly

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # resizing the picture to make it smaller
    if frame.shape[0] > 1500:
        scale = 50 / 100
        frame = resize(frame, scale)

    # detect the frontal face
    face_coord = frontalFaceDetection(frame)

    # detect profile face
    if len(face_coord) == 0:
        face_coord = profileFaceDetection(frame)

    # crop the frame around the face
    if len(face_coord) > 0:
        biggest_rectangle = face_coord[0]
        x1_big, y1_big, x2_big, y2_big = face_coord[0]

        for (x1, y1, x2, y2) in face_coord:

            # find the biggest face among the detected ones
            if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                biggest_rectangle = (x1, y1, x2, y2)

            # crop the frame around the face
            scale_around_face = 40 / 100
            roi = crop(frame, biggest_rectangle, scale_around_face)

            # save the image of the face
            cv2.imwrite(os.path.join(newDataFold, str(count) + ".png"), roi)
            print("Creating Image............." + os.path.join(newDataFold, str(count) + ".png"))

            count += 1

            # draw a rectangle around every detected faces for the video
            rectangle_color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), rectangle_color, thickness)
            frame = crop(frame, biggest_rectangle, scale_around_face)

    if count > 299:
        break

    # display the capture
    cv2.imshow("frame", frame)

    # stop if the x key is pressed
    if cv2.waitKey(1) == ord('x'):
        break

# release the capture
cam.release()
cv2.destroyAllWindows()
