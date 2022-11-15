# import numpy as np
# import tensorflow as tf
# import tensorflow_text as tf_text
# import typing
# from typing import Any,Tuple
# from flask import Flask
from tensorflow import keras
import cv2
import numpy as np
model = keras.models.load_model('keras_model')

input_image = cv2.imread('download.png')


def image_sample(a):
    input_image = cv2.imread(a)
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize/255
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28])
    input_prediction = model.predict(image_reshaped)
    input_pred_label = str(np.argmax(input_prediction))
    return input_pred_label
