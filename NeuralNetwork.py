import os
import keras as ker
import tensorflow as tf

BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

trainingData = os.path.join(dataSet, "TrainingData")
testData = os.path.join(dataSet, "TestData")

train_img = tf.data.Dataset.list_files(os.path.join(trainingData, "Images", "*.jpg"), shuffle=False)
train_img = train_img.map(cv2.imread)

test_img = tf.data.Dataset.list_files(os.path.join(testData, "Images", "*.jpg"), shuffle=False)
test_img = test_img.map(load_image)


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


# transmit training data to the NN
# model.fit(x=faces, y=names, epochs=1000)

# to predict result
# random_person = os.path.join(dataFold, "n000001")
# model.predict(os.path.join(random_person, "0010_01.jpg"))