import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.python.keras.backend import batch_normalization, conv2d
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pickle
from pathlib import Path
from datetime import datetime
import eel
import time
from tensorflow.keras import initializers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add
from tensorflow.keras import backend as K
from tensorflow import keras
import cv2
import numpy as np
import sys
import os
import math


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
from game.logic.game import Game

# Vars
dim_hidden_state = 16
dim_prediction_function_policy: int = 16

number_layers_prediction_function: int = 4
number_layers_dynamic_function: int = 4

number_neurons_prediction_function: int = 16
number_neurons_dynamic_function: int = 16

# Input
input_game_grid = Input(shape=(4, 4, 1))
input_shape = Input(shape=(2, 2, 1))

# Initiainference

# Hidden state

model_game_grid = Conv2D(8, (3, 3), activation="relu")(input_game_grid)
model_game_grid = Flatten()(model_game_grid)
model_game_grid = Dense(32, activation="relu")(model_game_grid)
model_game_grid = Dense(32, activation="relu")(model_game_grid)
model_game_grid = Model(inputs=[input_game_grid], outputs=[model_game_grid])

model_shape = Conv2D(8, (2, 2), activation="relu")(input_shape)
model_shape = Flatten()(model_shape)
model_shape = Dense(8, activation="relu")(model_shape)
model_shape = Dense(8, activation="relu")(model_shape)
model_shape = Model(inputs=[input_shape], outputs=[model_shape])

model_hidden_state = concatenate([model_game_grid.output, model_shape.output])

model_hidden_state = Dense(dim_hidden_state)(model_hidden_state)
model_hidden_state = Model(inputs=[input_game_grid, input_shape], outputs=[model_hidden_state])


# Policy/value model --> Prediction function
input_s0 = Input(shape=(dim_hidden_state, 1))

model_prediction = input_s0

for i in range(number_layers_prediction_function):
    model_prediction = Dense(number_neurons_prediction_function, activation="relu")(model_prediction)
    model_prediction = batch_normalization()(model_prediction)

model_prediction_policy = Dense(dim_prediction_function_policy, activation="softmax")(model_prediction)
model_prediction_value = Dense(1, activation="linear")(model_prediction)

model_prediction = Model(inputs=[input_s0], outputs=[model_prediction_policy, model_prediction_value])


# Dynamic function
input_s0 = Input(shape=(dim_hidden_state, 1))
input_current_action = Input(shape=(dim_prediction_function_policy, 1))

model_dynamic = concatenate([input_s0, input_current_action])

for i in range(number_layers_dynamic_function):
    model_dynamic = Dense(number_neurons_dynamic_function, activation="relu")(model_dynamic)
    model_dynamic = batch_normalization()(model_dynamic)

model_dynamic_s1 = Dense(dim_hidden_state)(model_dynamic)
model_dynamic_reward = Dense(1, activation="linear")(model_dynamic)
