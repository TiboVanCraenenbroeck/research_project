import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate

import cv2

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)


""" img_game_grid = cv2.imread(f"{BASE_DIR}/game_history/26_12_20__12_1408/game_gird/game_gird_1.jpg")
img_shape_queue_0 = cv2.imread(f"{BASE_DIR}/game_history/26_12_20__12_1408/queue_shapes/queue_shapes_00.jpg")

imgs_game_grid = np.array([img_game_grid], dtype=np.float)
img_shape_queue_0 = np.array([img_shape_queue_0], dtype=np.float)

imgs_game_grid /= 255
img_shape_queue_0 /= 255 """

""" model = Sequential()
model.add(Conv2D(3, (3, 3), input_shape=input_shape))
model.add(Conv2D(3, (3, 3)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(300, activation='linear'))
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

print(model.summary()) """


input_game_grid = Input(shape=(78, 78, 3))
input_shape_queue0 = Input(shape=(48, 48, 3))
input_shape_queue1 = Input(shape=(48, 48, 3))
input_shape_queue2 = Input(shape=(48, 48, 3))

model_game_grid = Conv2D(3, (3, 3))(input_game_grid)
model_game_grid = Conv2D(3, (3, 3))(model_game_grid)
model_game_grid = Flatten()(model_game_grid)
model_game_grid = Model(inputs=input_game_grid, outputs=model_game_grid, name="Model_GameGrid")

model_shape_queue0 = Conv2D(3, (3, 3))(input_shape_queue0)
model_shape_queue0 = Conv2D(3, (3, 3))(model_shape_queue0)
model_shape_queue0 = Flatten()(model_shape_queue0)
model_shape_queue0 = Model(inputs=input_shape_queue0, outputs=model_shape_queue0, name="Model_ShapeQueue")

model_shape_queue1 = Conv2D(3, (3, 3))(input_shape_queue1)
model_shape_queue1 = Conv2D(3, (3, 3))(model_shape_queue1)
model_shape_queue1 = Flatten()(model_shape_queue1)
model_shape_queue1 = Model(inputs=input_shape_queue1, outputs=model_shape_queue1, name="Model_ShapeQueue")

model_shape_queue2 = Conv2D(3, (3, 3))(input_shape_queue2)
model_shape_queue2 = Conv2D(3, (3, 3))(model_shape_queue2)
model_shape_queue2 = Flatten()(model_shape_queue2)
model_shape_queue2 = Model(inputs=input_shape_queue2, outputs=model_shape_queue2, name="Model_ShapeQueue")

combined = concatenate([model_game_grid.output, model_shape_queue0.output, model_shape_queue1.output, model_shape_queue2.output])

model_output = Dense(4, activation="relu")(combined)
model_output = Dense(1, activation="linear")(model_output)
model_output = Model(inputs=[model_game_grid.input, model_shape_queue0.input, model_shape_queue1.input, model_shape_queue2.input], outputs=model_output, name="Model_Output")

model_output.compile(loss="mse", optimizer=Adam(lr=0.001))

print(model_output.summary())


""" y = np.array([1])
history = model_output.fit([imgs_game_grid, img_shape_queue_0], y, epochs=10)

pred_y = model_output.predict([imgs_game_grid, img_shape_queue_0])
print(pred_y) """