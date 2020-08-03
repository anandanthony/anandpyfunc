# General libs
from datetime import datetime
import logging
import os
# Image processing libs
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from PIL import Image
# Additional libs
import numpy as np
from urllib.request import urlopen
import requests
from io import BytesIO
from numpy import genfromtxt
from scipy.spatial import distance

scriptpath = os.path.abspath(__file__)
dir = os.path.dirname(scriptpath)
image = os.path.join(dir, 'file.jpeg')
model_weights = os.path.join(dir, 'keras.h5')
dataset = os.path.join(dir, 'dataset.tsv')
classes = os.path.join(dir, 'classes.txt')
database = genfromtxt(dataset, delimiter='\t')
classes_list = genfromtxt(classes, delimiter='\n',dtype=None)
size = 480

def exctract_roi(image): # Exctract object from an image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blurred, 0,100, 3)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
    return ROI

def url_to_image(url): # Download image from URL and open in opencv
	#image = urlopen(url)
	#image = np.asarray(bytearray(image.read()), dtype="uint8")
	#image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image) 
    return image

def image_preprocessing(image_url):
    image = url_to_image(image_url)
    image = exctract_roi(image)
    image = np.array(image)
    tensor = tf.convert_to_tensor(image)
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = tf.image.resize(tensor, (size,size))
    return tf.expand_dims(tensor,0)

def result_post_processing(result):
    distances = []
    for i in database:
        dist = distance.euclidean(i,result)
        distances.append(dist)
    id = np.take(classes_list,np.argmin(distances))
    return id.decode("utf-8")

def predict_image_from_url(image_url):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='relu', input_shape=(480,480,3)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        #tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu'),
        #tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=None),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    model.load_weights(model_weights)
    model.compile(loss=tfa.losses.TripletSemiHardLoss(margin = 4.0))
    
    result = model.predict(image_preprocessing(image_url))
    
    mongoid = result_post_processing(result)
    return mongoid