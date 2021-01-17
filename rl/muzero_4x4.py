import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import math
import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import initializers
import time
import eel
from datetime import datetime
from pathlib import Path
import pickle


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
from game.logic.game import Game

class MuZero:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

        self.start()

        self.nn_hidden_state, self.nn_prediction, self.nn_dynamic = self.build_nn()
        self.make_model()
    
    def save_model(self):
        self.nn_hidden_state.save("h.h5")
        self.nn_prediction.save("f.h5")
        self.nn_dynamic.save("g.h5")
        self.model.save("base_model.h5")

    def start(self):
        # Vars
        self.dim_hidden_state = 16
        self.dim_prediction_function_policy: int = 16
        self.dim_dynamic_function_current_action: int = 16

        self.number_layers_prediction_function: int = 4
        self.number_layers_dynamic_function: int = 4

        self.number_neurons_prediction_function: int = 16
        self.number_neurons_dynamic_function: int = 16
    
    def build_nn(self):
        # Input
        input_game_grid = Input(shape=(4, 4, 1))
        input_shape = Input(shape=(2, 2, 1))

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

        model_hidden_state = Dense(self.dim_hidden_state)(model_hidden_state)
        model_hidden_state = Model(inputs=[input_game_grid, input_shape], outputs=[model_hidden_state])

        # Policy/value model --> Prediction function
        input_s0 = Input(shape=(self.dim_hidden_state, 1))

        model_prediction = input_s0

        for i in range(self.number_layers_prediction_function):
            model_prediction = Dense(self.number_neurons_prediction_function, activation="relu")(model_prediction)
            model_prediction = BatchNormalization()(model_prediction)

        model_prediction_policy = Dense(self.dim_prediction_function_policy, activation="softmax")(model_prediction)
        model_prediction_value = Dense(1, activation="linear")(model_prediction)

        model_prediction = Model(inputs=[input_s0], outputs=[model_prediction_policy, model_prediction_value])

        # Dynamic function
        input_current_action = Input(shape=(self.dim_dynamic_function_current_action, 1))

        model_dynamic = concatenate([input_s0, input_current_action])

        for i in range(self.number_layers_dynamic_function):
            model_dynamic = Dense(self.number_neurons_dynamic_function, activation="relu")(model_dynamic)
            model_dynamic = BatchNormalization()(model_dynamic)

        model_dynamic_s1 = Dense(self.dim_hidden_state)(model_dynamic)
        model_dynamic_reward = Dense(1, activation="linear")(model_dynamic)

        model_dynamic = Model(inputs=[input_s0, input_current_action], outputs=[model_dynamic_s1, model_dynamic_reward])

        return model_hidden_state, model_prediction, model_dynamic

    def make_model(self):
        outputs, loss = [], []

        def softmax_ce_logits(y_true, y_pred):
            return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        input_game_grid = Input(shape=(4, 4, 1))
        input_shape = Input(shape=(2, 2, 1))
        
        model_hidden_state = self.nn_hidden_state([input_game_grid, input_shape])

        model_policy, model_value = self.nn_prediction([model_hidden_state])
        outputs += [model_policy, model_value]
        loss += ["mse", softmax_ce_logits]

        input_current_action = Input(shape=(self.dim_dynamic_function_current_action, 1), name="test")
        model_state_1, model_reward = self.nn_dynamic([model_hidden_state, input_current_action])
        outputs += [model_reward]
        loss += ["mse"]

        self.model = Model(inputs=[input_game_grid, input_shape, input_current_action], outputs=outputs)
        self.model.compile(loss=loss, optimizer=Adam(self.learning_rate))


        


learning_rate = 0.01
test = MuZero(learning_rate)