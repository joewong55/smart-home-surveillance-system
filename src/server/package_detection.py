# Owner: Joseph Wong
# Last updated: 3/21/20
import os
import cv2
import numpy as np
import globals as g

from PIL import Image
from keras import applications
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

DEBUG = 0

def package_recognition():
    """
    Description: Predicts if package is present in frame or not
    Parameters: None
    Return: None
    """

    print("[INFO] Loading package recognition models")

    model = load_model('../../models/bottleneck_30_epochs.h5') # load offline trained model
    top_model = applications.VGG16(include_top=False,weights='imagenet') # load top model

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print("[INFO] Starting package recognition thread")
    print("[INFO] Ready to accept connections!")

    while(True):
        if g.package_candidates: # loop through global lists in order if they ar enot empty
            image = g.package_candidates[0]
            frame_number = g.package_candidates[1]

            # preprocess image
            image = Image.fromarray(image)
            image = image.resize((300,300),Image.NEAREST)
            image = img_to_array(image)
            image = image.reshape((1,)+image.shape)

            # make prediction
            feature_img = top_model.predict(image)
            classes = model.predict_classes(feature_img)
            prediction = 'package' if classes[0][0] else 'no_package'

            # handle prediction
            if prediction == 'package':
                if DEBUG:
                    print("[INFO] Detected: package from trigger frame " + str(frame_number))
                g.package_alert_queue.append(frame_number)
                g.package_alert_queue.append('package')
            else:
                if DEBUG:
                    print("[INFO] No package detected in trigger frame " + str(frame_number))
                g.package_alert_queue.append(frame_number)
                g.package_alert_queue.append('no_package')

            del g.package_candidates[:2] # remove from global list
