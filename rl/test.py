import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import initializers
from tensorflow.python.framework.ops import disable_eager_execution
import time
disable_eager_execution()

import eel
from datetime import datetime
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

from game.logic.game import Game

game = Game(3, False, 8081)

def create_standard_shape(shape):
    standard_shape = np.zeros((5, 5))
    for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col>0 else 0
    standard_shape = np.resize(standard_shape, (5, 5, 1))
    standard_shape /=10
    return standard_shape


X_train = np.array([create_standard_shape(shape) for shape in game.shapes])
y_train = [i for i, shape in enumerate(X_train)]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)

""" Get shape nn --> Pretrained"""

input_shape = Input(shape=(5, 5, 1))

model_game_shape = Conv2D(1, (2, 2), activation="relu")(input_shape)

model_game_shape = Flatten()(model_game_shape)

model_game_shape = Dense(19, activation="softmax")(model_game_shape)

model_game_shape = Model(inputs=[input_shape], outputs=model_game_shape)

model_game_shape.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy'])

history = model_game_shape.fit(X_train, y_train, epochs=100, validation_data=(X_train, y_train))

# Plot history
# Check for underfitting / overfitting based on the model loss history
# Determine the optimal number of training epochs. Use early stopping with model checkpoint saving.

# Accuray 
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'b')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Loss 
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'b')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Los functie gebruiken om model te valideren

