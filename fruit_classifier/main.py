#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:18:27 2019

@author: mloucks
"""

import os
import tensorflow as tf
import numpy as np
import random
import cv2
from os import system
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation

import pickle

# TODO: remove these globals
CATEGORIES = ["apples", "oranges", "blueberries", "bananas"]
DATA_DIR= "/Users/mloucks/fruit_classifier/img/"
IMG_SIZE = 50
IS_TRAINING = False

def load_data():
    '''
    Load and convert the data images into a numpy array.
    This data will be pickled and saved for later
    
    Returns:
        (X, y): A (n, 50, 50, 3) numpy array and a 1d list of labels
    '''
    
    
    training_data = []
    
    # load images into numpy array
    for label in CATEGORIES:
        
        category = CATEGORIES.index(label)
        path = os.path.join(DATA_DIR, label)
        print("Loading", path, "...")
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img_resize = np.divide(img_resize, 255)
                training_data.append([img_resize, category])
            except Exception:
                continue
            
    random.shuffle(training_data)
        
    X = []
    y = []
    for feature, label in training_data:
        X.append(feature)
        y.append(label)
    
    print("\nSaving pickle data...")
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    
    
    save_data_pickle(X, y)
    
    return X, y


def save_data_pickle(X, y):
    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    
    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    
    
def load_data_pickle():
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)
    pickle_in.close()
    
    return X, y


def prepare_img(path):
    '''
    Prepares an image for classification or training by converting it into
    IMG_SIZE by IMG_SIZE pixels
    '''
    img_array = cv2.imread(path)
    img_resize = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return img_resize.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    

def train_model():
    '''Train the CNN. Be sure the file paths are correct'''

    X, y = None
    
    if os.path.isfile("X.pickle") and os.path.isfile("y.pickle"):
        X, y = load_data_pickle()
    else:
        X, y = load_data()
  
    model = Sequential()
    
    model.add(Conv2D(256, (5, 5), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(X, y, batch_size=32, epochs=5, 
              validation_split=0.2, use_multiprocessing=True)
    
    model.save('Fruit-CNN.model')
    
    return model

def camera_prediction(model, speak=False):
    '''
    open the user's camera in a new window and print a prediction based
    on what the model sees.
    
    Will also optionally say the command (Mac only)
    
    Args:
        model (tensorflow.keras.models.Sequential: The model to be processed
        speak (bool): use the say command (Mac only)
    '''
    
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    
    rval, frame = vc.read()
    
    while True:
      if frame is not None:   
         
         img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
         
         new_img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

         prediction = model.predict([new_img])
         cv2.imshow("preview", img)
         
         if speak:
             system("say " + get_classification(prediction))
             
         print(get_classification(prediction), prediction)
      rval, frame = vc.read()
    
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    
    cv2.destroyAllWindows()
    
    
def get_classification(prediction):
    '''Return a classification for the given model prediction'''
    return CATEGORIES[np.argmax(prediction[0])]
    
    
def main():
    
    #load model
    model = None
    
    if IS_TRAINING:
        model = train_model()
    else:
        try:
            model = tf.keras.models.load_model("Fruit-CNN.model")
        except Exception as e:
            print(e)

    if model == None:
        print("No model file found")
        exit()
              
    camera_prediction(model)


if __name__ == "__main__":
    main()











