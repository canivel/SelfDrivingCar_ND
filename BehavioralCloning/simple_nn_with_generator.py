import csv
import cv2
import numpy as np
from tqdm import tqdm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format


#simple NN test

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

#cropping top and bottom
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Conv2D(36, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Conv2D(48, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3,
                    verbose=2)

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

model.save('model_v4.h5')