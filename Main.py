from FaceDetection import *
from Preprocessing import *

# get the path of files
currentPath = os.getcwd()
dataFold = os.path.join(currentPath, "Data")

newData = os.path.join(dataFold, "Enzo")

# get the image
# img = cv2.imread(os.path.join(newData, "Enzo_0001.jpg"), 1)

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

    # resizing the picture to make it smaller
    if frame.shape[0] > 1500:
        scale = 50 / 100
        frame = resize(frame, scale)

    # detect the face
    face_coord = faceDetection(frame)

    # crop the frame around the face if there is only one detected
    scale_around_face = 40 / 100

    if len(face_coord) == 1:
        frame = crop(frame, face_coord[0], scale_around_face)

    # find the face if there is more than one face detected
    if len(face_coord) > 1:
        x1_big, y1_big, x2_big, y2_big = face_coord[0]

        for (x1, y1, x2, y2) in face_coord:
            cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (255, 0, 0), 2)  # draw a rectangle around every detected faces

            if (x1_big < x1 and y1_big < y1) or (x1_big + x2_big < x1 + x2 and y1_big + y2_big < y1 + y2):
                biggest_rectangle = (x1, y1, x2, y2)

        # crop the frame around the biggest detected face
        frame = crop(frame, biggest_rectangle, scale_around_face)

    # display the capture
    cv2.imshow("frame", frame)

    # stop if the x key is pressed
    if cv2.waitKey(1) == ord('x'):
        break

# release the capture
cam.release()
cv2.destroyAllWindows()
