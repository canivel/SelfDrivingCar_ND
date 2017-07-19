import csv
import cv2
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split


#read lines from drive log csv file
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#get the images for each line
images = []
measurements = []
for line in lines:
    source_path = line[0]
    # filename = source_path.split('\\')[-1]
    # current_path = 'recordings/IMG/{}'.format(filename)
    image = cv2.imread(source_path)
    images.append(image)

    #locate at 4 token
    measurement = float(line[3])
    measurements.append(measurement)

#Augmenting
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#simple NN test

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Normalize the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
#cropping top and bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=2)

# from keras.models import Model
# import matplotlib.pyplot as plt
#
# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data =
#     validation_generator,
#     nb_val_samples = len(validation_samples),
#     nb_epoch=5, verbose=1)
#
# ### print the keys contained in the history object
# print(history_object.history.keys())
#
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

#save the model

model.save('model_v3.h5')