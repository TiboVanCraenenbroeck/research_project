import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import collections
import random
import numpy as np



import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import initializers
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.utils import class_weight
from tensorflow.python.framework.ops import disable_eager_execution
import time
disable_eager_execution()

import eel
from datetime import datetime
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from game.logic.game import Game
from game.models.shape import Shape


def check_space_available(env, shape: Shape, row: int = 0, col: int = 0) -> bool:
    # Check if the space is available
    space_available: bool = True
    for r in range(shape.nr_rows):
        # Check cols
        for c in range(shape.nr_cols):
            # Check if the cell is not out of the grid
            if r + row >= 5 or c + col >= 5:
                return False
            nr_cols_available = env[r+row, c+col]
            # Check if it is an empty block
            if shape.shape[r][c] != -1:
                if nr_cols_available>0 and shape.shape[r][c]>0:
                    return False
    return True

def create_standard_shape(shape):
    standard_shape = np.zeros((5, 5))
    for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col>0 else 0
    return standard_shape

def change_state_shape_queue(shape_queue):
    shape_queue = [np.array(shape, dtype=np.float) for shape in shape_queue]
    
    shape_queue = [np.resize(shape,(5, 5, 1)) for shape in shape_queue]

    shape_queue = [shape/10 for shape in shape_queue]
    
    return shape_queue

def create_random_envs(n=10):
    envs = []
    for i in range(n):
        env = np.random.rand(5, 5)
        env[env<=0.5] = 0
        env[env>0.5] = 1
        env = np.array(env, dtype=np.float)
        env = np.resize(env, (5, 5, 1))
        env /= 10
        envs.append(env)
    return envs

def add_shape_to_env(envs, shapes):
    shapes_c = [create_standard_shape(shape) for shape in shapes]
    shapes_c = change_state_shape_queue(shapes_c)
    X_train_envs = []
    X_train_shapes = []
    y_train = []
    for env in envs:
        for i, shape in enumerate(shapes):
            X_train_envs.append(env)
            X_train_shapes.append(shapes_c[i])
            past = check_space_available(env, shape)
            past = 1 if past else 0
            y_train.append(past)
    return X_train_envs, X_train_shapes, y_train
""" def add_shape_to_env(envs, shapes):
    shapes_c = [create_standard_shape(shape) for shape in shapes]
    shapes_c = change_state_shape_queue(shapes_c)
    X_train_envs = []
    X_train_shapes = []
    y_train = []
    for env in envs:
        for i, shape in enumerate(shapes):
            X_train_envs.append(env)
            X_train_shapes.append(shapes_c[i])
            past = check_space_available(env, shape)
            past = 1 if past else 0
            y_train.append(past)
    return X_train_envs, X_train_shapes, y_train """



game = Game(3, False, 8080)


""" class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
print(class_weights)
class_weights= dict(enumerate(class_weights)) """

input_shape_state = Input(shape=(5, 5, 1))
input_shape_shape = Input(shape=(5, 5, 1))

dropout = 0.5
model_game_shape = add([input_shape_state, input_shape_shape])

model_game_shape = Conv2D(4, (2, 2), activation="relu")(model_game_shape)
model_game_shape = Dropout(dropout)(model_game_shape)
model_game_shape = Flatten()(model_game_shape)

model_game_shape = Dense(1, activation="sigmoid")(model_game_shape)

model_game_shape = Model(inputs=[input_shape_state, input_shape_shape], outputs=model_game_shape)

model_game_shape.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy'])


replay_memory_size = 1000
space_availible_shapes_true = collections.deque(maxlen=replay_memory_size)
space_availible_shapes_false = collections.deque(maxlen=replay_memory_size)

def add_to_memory(sa: bool, step):
    if sa:
        space_availible_shapes_true.append(step)
    else:
        space_availible_shapes_false.append(step)

def train():
    bs = 64
    if len(space_availible_shapes_true)>=bs and len(space_availible_shapes_false)>=bs:
        samples_true = random.sample(space_availible_shapes_true, bs)
        samples_false = random.sample(space_availible_shapes_false, bs)
        mini_batches = samples_true + samples_false
        X_train_env = np.asarray(list(zip(*mini_batches))[0], dtype=float)
        X_train_shape = np.asarray(list(zip(*mini_batches))[1], dtype=float)
        y_train = np.asarray(list(zip(*mini_batches))[2], dtype=float)

        model_game_shape.fit([X_train_env, X_train_shape], y_train, epochs=1, verbose=0)


nr = 263
for i in range(nr):
    envs = create_random_envs(1)
    X_train_envs, X_train_shapes, y_train = add_shape_to_env(envs, game.shapes)
    
    X_train_envs = np.array(X_train_envs)
    X_train_shapes = np.array(X_train_shapes)
    y_train = np.array(y_train)

    for i, t in enumerate(X_train_envs):
        step = (X_train_envs[i], X_train_shapes[i], y_train[i])
        add_to_memory(y_train[i], step)
    train()
    """ i = np.random.randint(0, len(game.shapes))

    X_train_env = np.array([X_train_envs[i]])
    X_train_shape = np.array([X_train_shapes[i]])
    y_train = np.array([y_train[i]]) """

    """ class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
    class_weights= dict(enumerate(class_weights))  """
    #history = model_game_shape.fit([X_train_env, X_train_shape], y_train, epochs=1, verbose=0)



envs = create_random_envs(1000)

X_test_envs, X_test_shapes, y_test = add_shape_to_env(envs, game.shapes)

X_test_envs = np.array(X_test_envs)
X_test_shapes = np.array(X_test_shapes)
y_test = np.array(y_test)

# Evaluation on the test set 

y_pred = model_game_shape.predict([X_test_envs, X_test_shapes])

y_pred_1 = [0 if pre<0.5 else 1 for pre in y_pred]

print('\n')
print(classification_report(y_test, y_pred_1))

cf = confusion_matrix(y_test, y_pred_1)

print(cf)
print(accuracy_score(y_test, y_pred_1) * 100)