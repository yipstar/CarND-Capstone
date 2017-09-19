import os
import h5py
import glob
import numpy as np
import pandas as pd
import augmentations as aug

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from keras.layers import MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.optimizers import SGD, Adam
from keras.regularizers import l2 
from keras.callbacks import  EarlyStopping, History, ModelCheckpoint

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import yaml
from sklearn.cross_validation import train_test_split

# Set the seed for reproducible results
np.random.seed(111)

train_yaml_path = "../../../../../data/bosch/train.yaml"
train_img_path = "../../../../../data/bosch/"
val_img_path = "../../../../../data/bosch/"

with open(train_yaml_path, 'r') as f:
    dataset = yaml.load(f)
    # df = pd.io.json.json_normalize(dataset)

# image_paths = df["path"]
# labels = df[""]

# print(dataset)

image_paths = []
labels = []

# In TrafficLgiht UNKNOWN=4, GREEN=2, YELLOW=1, and RED=0
red_count = yellow_count = green_count = 0

for row in dataset:
    # print(row)
    if len(row["boxes"]) > 0:
        for box in row["boxes"]:
            if box["label"] == "Red":
                labels.append(0)
                image_paths.append(row["path"])
                red_count += 1
                break
            elif box["label"] == "Yellow":
                labels.append(1)
                image_paths.append(row["path"])
                yellow_count += 1
                break
            elif box["label"] == "Green":
                labels.append(2)
                image_paths.append(row["path"])
                green_count += 1
                break

# print(labels)
print("red: {}, yellow: {}, green: {}".format(red_count, yellow_count, green_count))
# red: 1139, yellow: 167, green: 1755

x_train, x_valid, y_train, y_valid = train_test_split(image_paths, labels, test_size=0.10, random_state=42, stratify=labels)

# print("x_tran: ", len(x_train))
# print("y_tran: ", len(y_train))

# print("x_valid: ", len(x_valid))
# print("y_valid: ", len(y_valid))

orig_width = 1280
orig_height = 720

# # dimensions of our images.
img_width, img_height = 224, 224

epochs = 20
batch_size = 32
nb_train_samples = len(x_train)
nb_valid_samples = len(x_valid)

# print("Number of training pairs : {}  Validation pairs : {} : ".format(len(train), len(valid)))

# del train, valid

# # One-hot encoding for labels
tr_labels = to_categorical(y_train).astype(np.int8)
val_labels = to_categorical(y_valid).astype(np.int8)

del y_train, y_valid

base_model = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

# Make base layers(upto the last conv block) non-trainable
for layer in base_model.layers[:-4]:
    layer.trainable=False

base_model_output = base_model.output

# Add new layers
x = Conv2D(1024, (1,1), padding='valid',name='fc6')(base_model_output)
x = Dropout(0.5, name='drop1')(x)
x = Conv2D(512, (1,1), padding='valid', name='fc7')(x)
x = Dropout(0.5, name='drop2')(x)
x = Conv2D(3, (1,1), padding='valid', name='logits', activation=None)(x)
x = AveragePooling2D(pool_size=(7,7), name='avgpool')(x)
x = Flatten()(x)
x = Activation('softmax', name='probs')(x)

model = Model(base_model.input, outputs=x)
model.summary()

###############################################################################################################

# Define a custom train data generator
def train_data_gen():
    batch_images = np.zeros((batch_size, 224,224,3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,3), dtype=np.int8)
    count = 0

    while 1:
        # Select samples randomly
        i = np.random.randint(len(x_train))
        img = image.load_img(train_img_path + x_train[i], target_size=(224,224))
        img = image.img_to_array(img)
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


################################################################################################


# Define a custom valid data generator
def valid_data_gen():
    batch_images = np.zeros((batch_size, 224,224,3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,3), dtype=np.int8)
    count = 0
    
    while 1:
        for i in range(len(x_valid)):
            img = image.load_img(val_img_path + x_valid[i], target_size=(224,224))
            img = image.img_to_array(img)
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


##################################################################################################


train_generator = train_data_gen()
valid_generator = valid_data_gen()


# Add an optimizer
opt = Adam(lr=1e-4)

# Instantiate early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)


###################################################################################################


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=nb_valid_samples // batch_size,
                    callbacks=[early_stop])

model_json = model.to_json()
with open("model_vgg_latest.json", "w") as json_file:
    json_file.write(model_json)

#Save weights
model.save_weights('latest_weights.h5')


