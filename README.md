# Facial-Recognition
Recognize people from video

This is a personal project to learn convolutional neural network and it is still being improved

------------------------------------
## Dependencies:

- cv2 (opencv-python)
- keras
- tensorflow
- numpy

you can install these dependencies by using pip install + the name of the dependency in any python console or use conda install + the name of the dependency in an anaconda prompt

------------------------------------
## How to use it:
To create your own dataset for the program to recognize you, run collectData.py (make sure you have a working webcam)

You can add more training data by putting pictures in the VGG file and adding the directory name and the corresponding name in the identity_meta.csv file and then running DataSet.py

Then, you will have to run NeuralNetwork.py to train and save the CNN (I could not upload the file to github, it was too large)

Finally, you can run main.py and you should see your camera feed with your name above your face
