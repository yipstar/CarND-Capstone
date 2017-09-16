import os
import h5py
import glob
import numpy as np
import pandas as pd
import augmentations as aug

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.optimizers import SGD, Adam
from keras.regularizers import l2 
from keras.callbacks import  EarlyStopping, History, ModelCheckpoint

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

############################################################################################################

# Set the seed for reproducible results
np.random.seed(111)

# dimensions of our images.
img_width, img_height = 224, 224

epochs = 100
batch_size = 32
nb_train_samples = 1790
nb_valid_samples = 47

# Define paths for training and validation images
train_img_path = './training/'
val_img_path = './validation/'

##################################################################################################################


# Read the training and validation file containing name and label of all the images
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')

print("Number of training pairs : {}  Validation pairs : {} : ".format(len(train), len(valid)))

x_train = train['image']
y_train = train['label']

x_valid = valid['image']
y_valid = valid['label']

del train, valid


##################################################################################################################


# One-hot encoding for labels
tr_labels = to_categorical(y_train).astype(np.int8)
val_labels = to_categorical(y_valid).astype(np.int8)

del y_train, y_valid

##################################################################################################################3


def fire_module(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x


def build_squeezeNet(input_shape=(224,224,3)):
    img_input = Input(shape=input_shape)

    x = Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="valid")(x)

    x = fire_module(x, (16, 64, 64), name="fire2")
    x = fire_module(x, (16, 64, 64), name="fire3")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="valid")(x)

    x = fire_module(x, (32, 128, 128), name="fire4")
    x = fire_module(x, (32, 128, 128), name="fire5")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="valid")(x)

    x = fire_module(x, (48, 192, 192), name="fire6")
    x = fire_module(x, (48, 192, 192), name="fire7")

    x = fire_module(x, (64, 256, 256), name="fire8")
    x = fire_module(x, (64, 256, 256), name="fire9")
    
    model = Model(img_input, x, name="squeezenet")
    
    # Load the weights for the layers
    file=h5py.File('notop_squeezenet.h5','r')
    weight = []
    
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    
    # Make the base layers non-trainable
    for layer in model.layers:
        layer.trainable = False
    
    return model


def build_final_model(base_model):
    # Add layers for transfer learning/ fine-tuning
    base_model_ouput = base_model.output
    x = Dropout(0.5, name='dropout9')(base_model_ouput)
    x = Convolution2D(3, (1, 1), padding='valid',name='conv10')(x)
    x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
    x = Flatten(name='flatten10')(x)
    x = Activation("softmax", name='logits')(x)
    
    final_model = Model(inputs=base_model.input, outputs=x)
    
    return final_model


##########################################################################################################################


# Get the base model and add new layers
base_model = build_squeezeNet()
model = build_final_model(base_model)
# print(model.summary())

########################################################################################################################


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


####################################################################################################################


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


######################################################################################################################            


train_generator = train_data_gen()
valid_generator = valid_data_gen()


# Add an optimizer
opt = Adam(lr=1e-4)

# Instantiate early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10)


######################################################################################################################


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=nb_valid_samples // batch_size,
                    callbacks=[early_stop])


#######################################################################################################################

# Rename layers for saving
# TODO: this causes load_weights in tl_classifier to fail
i = 0
for layer in model.layers:
    layer.name = "renamed_model_{0}".format(i)
    i += 1


# serialize model to JSON
model_json = model.to_json()
with open("model_squeezenet.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_squeezenet.h5")
print("Saved model to disk")


#######################################xo#############################################################################
