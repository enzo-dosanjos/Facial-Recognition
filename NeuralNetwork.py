import os

currentPath = os.getcwd()
dataFold = os.path.join(currentPath, "Data")

trainingData = os.path.join(dataFold, "lfw")

# training data for tests
count = 0
for dirs in os.listdir(trainingData):
    face = os.path.join(trainingData, dirs)
    for pics in os.listdir(face):
        count += 1
print(count)
