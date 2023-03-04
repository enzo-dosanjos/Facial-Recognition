import os
import keras as ker
import tensorflow as tf

BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

trainingData = os.path.join(dataFold, "DataSet")

images_path = []
for dirs in os.listdir(trainingData):
    path = os.path.join(trainingData, dirs)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith("png") or file.endswith("jpg"):
                images_path += [os.path.join(path, file)]

print(images_path)

# train_img = tf.data.Dataset.
# train_label = tf.data.Dataset
"""
# Creating the neural network
model = ker.models.Sequential()

# Input layer
model.add(ker.layers.Dense(units=3, input_shape=[1]))

# Intermediate layer
model.add(ker.layers.Dense(units=2))

# Output layer
model.add(ker.layers.Dense(units=1))

# compile the neural network
model.compile(loss='mean_squared_error', optimizer='sgd')

faces = []
names = []

# transmit training data to the NN
model.fit(x=faces, y=names, epochs=1000)

# to predict result
random_person = os.path.join(dataFold, "n000001")"""
# model.predict(os.path.join(random_person, "0010_01.jpg"))