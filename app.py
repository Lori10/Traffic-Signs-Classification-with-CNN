# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model.make_predict_function()          # Necessary for vgg19 in order to make prediction


def model_predict(img_path, model):
    # Load the image
    # img is PIL.Image.Image with mode=RGB and size=224x224 size
    img = image.load_img(img_path, target_size=(32, 32))

    # Preprocessing the image : we must do the same preprocessing that we did on training images

    # Convert image to an array (image will have shape 32x32x3 since it has mode=RGB
    x = image.img_to_array(img)

    # exapand_dims will add dimension=1 in front of the image of shape.
    # before applying expand_dims image shape = (32x32x3). After shape becomes (1x32x32x3)
    # The shape of image when we pass to the model for prediction must be of shape (1x32x32x3)
    x = np.expand_dims(x, axis=0)

    # Convert RGB Image to Gray Scale Image (from 1x32x32x3 to 1x32x32x1)
    x = np.sum(x / 3, axis=3, keepdims=True)

    # scale the image as we scaled during training
    x = x / 255

    # # make predictions using our model
    # # preds will be a 2D Array with the probabilities that this image belongs to each of 1000 classes.
    pred = np.argmax(model.predict(x), axis=-1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request (html form)
        f = request.files['file']

        # Save the uploaded file to ./uploads
        # base_path is the current directory
        basepath = os.path.dirname(__file__)
        # create file path
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        # save image
        f.save(file_path)

        # Make prediction
        classes = {0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
                   2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
                   4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
                   6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
                   8: 'Speed limit (120km/h)', 9: 'No passing',
                   10: 'No passing for vehicles over 3.5 metric tons',
                   11: 'Right-of-way at the next intersection', 12: 'Priority road',
                   13: 'Yield', 14: 'Stop', 15: 'No vehicles',
                   16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry',
                   18: 'General caution', 19: 'Dangerous curve to the left',
                   20: 'Dangerous curve to the right', 21: 'Double curve',
                   22: 'Bumpy road', 23: 'Slippery road',
                   24: 'Road narrows on the right', 25: 'Road work',
                   26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
                   29: 'Bicycles crossing', 30: 'Beware of ice/snow',
                   31: 'Wild animals crossing',
                   32: 'End of all speed and passing limits', 33: 'Turn right ahead',
                   34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
                   37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
                   40: 'Roundabout mandatory', 41: 'End of no passing',
                   42: 'End of no passing by vehicles over 3.5 metric tons'}

        pred = model_predict(file_path, model)
        class_label = classes[pred[0]]
        return class_label
    return None


if __name__ == '__main__':
    app.run(debug=True)
