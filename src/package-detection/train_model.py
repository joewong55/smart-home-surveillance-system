# Owner: Joseph Wong
# Last updated: 3/21/20

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import plot_model

img_width, img_height = 300, 300

# Metadata
train_data_dir = '../data/train'
validation_data_dir = '../data/validate'
nb_train_samples = 2000
nb_validation_samples = 944
batch_size = 16

def save_bottlebeck_features():
    """
    Description: Extract bottleneck features from CNN
    Parameters: None
    Return: None
    """

    # Augmentation configuration
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Use VGG16 model
    model = applications.VGG16(include_top=False, weights='imagenet')

    # Get train bottleneck features
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    # Save train bottleneck features as .npy
    np.save(open('../models/bottle/features/bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    # Get validation bottleneck features
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    # Save train bottleneck features as .npy
    np.save(open('../models/bottle/features/bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model(epoch):
    """
    Description: Train model on top of bottleneck features
    Parameters: epochs - number of epochs
    Return: None
    """

    # Load train bottleneck features and get labels
    train_data = np.load(open('../models/bottle/features/bottleneck_features_train.npy','rb'))
    train_label_array = (nb_train_samples // 2)
    train_labels = np.array([0] * int(train_label_array) + [1] * int(train_label_array))

    # Load validation bottleneck features and get labels
    validation_data = np.load(open('../models/bottle/features/bottleneck_features_validation.npy','rb'))
    validation_label_array = (nb_validation_samples // 2)
    validation_labels = np.array([0] * int(validation_label_array) + [1] * int(validation_label_array))

    # Top model architecture
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    
    # Save model and weights to server/models directory
    model.save_weights('../server/models/bottleneck_30_epochs_weights.h5')
    model.save('../server/models/bottleneck_30_epochs.h5')

if __name__ == '__main__':
    """
    Description: Train model to detect packages
    Parameters: None
    Return: None
    """
 
    epoch = 30 # Set epochs, currently using 30

    save_bottlebeck_features()
    train_top_model(epoch)
