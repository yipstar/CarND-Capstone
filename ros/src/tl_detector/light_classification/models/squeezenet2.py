import time
import json
import argparse
import squeezenet_model as km
from keras.optimizers import Adam
from keras.optimizers import SGD

import os
import h5py
import glob
import numpy as np
import pandas as pd
import augmentations as aug

from keras.callbacks import EarlyStopping, History, ModelCheckpoint

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

def print_time(t0, s):
    """Print how much time has been spent
    @param t0: previous timestamp
    @param s: description of this step
    """

    print("%.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()

# Set the seed for reproducible results
np.random.seed(111)

# dimensions of our images.
img_width, img_height = 224, 224

# epochs = 1
epochs = 100
batch_size = 32
nb_train_samples = 1790
nb_valid_samples = 47

# Define paths for training and validation images
train_img_path = './training/'
val_img_path = './validation/'

# Read the training and validation file containing name and label of all the images
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')

print("Number of training pairs : {}  Validation pairs : {} : ".format(len(train), len(valid)))

x_train = train['image']
y_train = train['label']

x_valid = valid['image']
y_valid = valid['label']

del train, valid

# One-hot encoding for labels
tr_labels = to_categorical(y_train).astype(np.int8)
val_labels = to_categorical(y_valid).astype(np.int8)

del y_train, y_valid

# Define a custom train data generator
def train_data_gen():
    batch_images = np.zeros((batch_size, 3,224,224), dtype=np.float32)
    batch_labels = np.zeros((batch_size,3), dtype=np.int8)
    count = 0

    while 1:
        # Select samples randomly
        i = np.random.randint(len(x_train))
        img = image.load_img(train_img_path + x_train[i], target_size=(224,224))
        img = image.img_to_array(img)
        # print("img shape", img.shape)

        img = img.transpose((-1, 0, 1))
        # print("img shape", img.shape)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = np.squeeze(img)
        label = tr_labels[i]

        # Do a random augmentation
        j = np.random.randint(3)
        if j==0:
            img = aug.random_rotation(img, 30)
        elif j==1:
            img = aug.random_shear(img, 0.2)
        elif j==2:
            img = aug.random_zoom(img, (0.2, 0.2))

        # Add the file and label to the arrays
        batch_images[count] = img
        batch_labels[count] = label
        count +=1

        if count == batch_size:
            count = 0
            yield batch_images, batch_labels

# Define a custom valid data generator
def valid_data_gen():
    batch_images = np.zeros((batch_size, 3,224,224), dtype=np.float32)
    batch_labels = np.zeros((batch_size,3), dtype=np.int8)
    count = 0

    while 1:
        for i in range(len(x_valid)):
            img = image.load_img(val_img_path + x_valid[i], target_size=(224,224))
            img = image.img_to_array(img)
            img = img.transpose((-1, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img = np.squeeze(img)
            label = val_labels[i]

            batch_images[count] = img
            batch_labels[count] = label
            count +=1

            if count == batch_size:
                count = 0
                yield batch_images, batch_labels
        if count > 0:
            yield batch_images, batch_labels

train_generator = train_data_gen()
valid_generator = valid_data_gen()

# Add an optimizer
opt = SGD(lr=0.001, decay=0.0002, momentum=0.9)

t0 = time.time()

nb_class = 3
channels = 3
width = 224
height = 224

# Instantiate early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)

t0 = print_time(t0, 'initialize data')
model = km.SqueezeNet(nb_class, inputs=(channels, height, width))

# dp.visualize_model(model)
t0 = print_time(t0, 'build the model')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

t0 = print_time(t0, 'compile model')

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=nb_valid_samples // batch_size,
                    callbacks=[early_stop])

t0 = print_time(t0, 'train model')

# serialize model to JSON
model_json = model.to_json()
with open("model_squeezenet2.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_squeezenet2.h5", overwrite=True)
t0 = print_time(t0, 'saved model')
