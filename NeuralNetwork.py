import os
import json

from keras_vggface.vggface import VGGFace

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

BASE_DIR = os.getcwd()
dataFold = os.path.join(BASE_DIR, "Data")

dataSet = os.path.join(dataFold, "DataSet")
trainingData = os.path.join(dataSet, "TrainingData")
testData = os.path.join(dataSet, "TestData")

# augment training data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(trainingData, "Images"),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# Creating the neural network
vgg_model = VGGFace(include_top=False,
                    model="resnet50",
                    input_shape=(224, 224, 3))

last_layer = vgg_model.output
x = GlobalAveragePooling2D()(last_layer)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# get the number of different person the CNN will be able to output
nb_categories = 0
for dirs in os.listdir(os.path.join(dataSet, "TrainingData", "Images")):
    nb_categories += 1

out = Dense(nb_categories, activation='softmax')(x)

# create a new model with the model's original input and the new model's output
custom_vgg_model = Model(vgg_model.input, out)

trained_layers = len(vgg_model.layers)  # to know the number of layers that we don't have to train

# don't train the first 174 layers because they were already trained by VGG
for layer in custom_vgg_model.layers[:trained_layers]:
    layer.trainable = False

# make our layers trainable
for layer in custom_vgg_model.layers[trained_layers:]:
    layer.trainable = True

custom_vgg_model.compile(optimizer='Adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

custom_vgg_model.fit(train_generator,
                     batch_size=1,
                     verbose=1,
                     epochs=100)

# save the CNN in a h5 file
custom_vgg_model.save(os.path.join(dataFold, "face_recognition_cnn.h5"))

# save the classes to txt
label_filename = 'face_labels.json'
label_file_path = os.path.join(dataFold, label_filename)
class_dictionary = train_generator.class_indices
label_dict = {
    value: key for key, value in class_dictionary.items()
}
json_object = json.dumps(label_dict, indent=4)
with open(label_file_path, 'w') as label:
    label.write(json_object)
