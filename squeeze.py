import os
import h5py
import glob
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation, Dense
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.optimizers import SGD, Adam
from keras.regularizers import l2 
from keras.callbacks import  EarlyStopping, History, ModelCheckpoint

from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file


# Set the seed for reproducible results
np.random.seed(111)

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = './data/Training'
validation_data_dir = './data/Validation'
nb_train_samples = 1790
nb_validation_samples = 47
epochs = 100
batch_size = 32


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
    for layer in model.layers[:-2]:
        layer.trainable = False
    
    return model


def build_final_model(base_model):
    # Add layers for transfer learning/ fine-tuning
    base_model_ouput = base_model.output
    x = Dropout(0.5, name='dropout9')(base_model_ouput)
    x = Convolution2D(3, (1, 1), padding='valid', kernel_regularizer=l2(0.001) ,name='conv10')(x)
    x = AveragePooling2D(pool_size=(13, 13), name='avgpool10')(x)
    x = Flatten(name='flatten10')(x)
    x = Activation("softmax", name='logits')(x)
    
    final_model = Model(inputs=base_model.input, outputs=x)
    
    return final_model


base_model = build_squeezeNet()
model = build_final_model(base_model)
print(model.summary())


# Define generators
train_data_generator = ImageDataGenerator(rescale=1./255) 
                                    # shear_range=0.2, 
                                    # rotation_range=30,  
                                    # zoom_range=0.4)
                                  

valid_data_generator = ImageDataGenerator(rescale=1./255)


# Read the images from the directories
train_generator = train_data_generator.flow_from_directory(directory=train_data_dir,
                                                      batch_size=batch_size, 
                                                      class_mode='categorical', 
                                                      classes=['Green', 'Red', 'Yellow'],
                                                      shuffle=True, 
                                                      target_size=(img_height, img_width))

valid_generator = valid_data_generator.flow_from_directory(directory=validation_data_dir,
                                                      batch_size=batch_size, 
                                                      class_mode='categorical', 
                                                      classes=['Green', 'Red', 'Yellow'],
                                                      shuffle=True, 
                                                      target_size=(img_height, img_width))



# Add an optimizer
opt = SGD(lr=1e-5)

# Instantiate early stopping and model checkpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25)

chkpt = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                        monitor='val_loss', 
                        save_best_only=True, 
                        mode='min')

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[early_stop])


# Save the weights of the model
file = h5py.File('final_model_weights.h5','w')
weight = model.get_weights()

for i in range(len(weight)):
    file.create_dataset('weight'+str(i),data=weight[i])
file.close()
