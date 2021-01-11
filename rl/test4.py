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
#from tensorflow.python.framework.ops import disable_eager_execution
import time
#disable_eager_execution()


def build_nn__chance_space_availible(nr):
    input_shape_state = Input(shape=(5, 5, 1), name=f"input_state_{nr}")
    input_shape_shape = Input(shape=(5, 5, 1), name=f"input_shape_{nr}")
    dropout = 0.5

    model = add([input_shape_state, input_shape_shape], name=f"add_{nr}")

    model = Conv2D(4, (2, 2), activation="relu", name=f"conv2d_{nr}")(model)
    model = Dropout(dropout, name=f"dropout_{nr}")(model)
    model = Flatten(name=f"flatten_{nr}")(model)

    model = Dense(1, activation="sigmoid", name=f"dense_{nr}")(model)

    model = Model(inputs=[input_shape_state, input_shape_shape], outputs=model)

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy'])

    return model

models = []
models_input = []
models_output = []
for nr in range(100):
    input_shape_state = Input(shape=(5, 5, 1), name=f"input_state_{nr}")
    input_shape_shape = Input(shape=(5, 5, 1), name=f"input_shape_{nr}")
    dropout = 0.5

    model = add([input_shape_state, input_shape_shape], name=f"add_{nr}")

    model = Conv2D(4, (2, 2), activation="relu", name=f"conv2d_{nr}")(model)
    model = Dropout(dropout, name=f"dropout_{nr}")(model)
    model = Flatten(name=f"flatten_{nr}")(model)

    model = Dense(1, activation="sigmoid", name=f"dense_{nr}")(model)

    model = Model(inputs=[input_shape_state, input_shape_shape], outputs=model)

    models.append(model)
    models_input.append(input_shape_state)
    models_input.append(input_shape_shape)
    models_output.append(model.output)



""" nr = 1
input_shape_state = Input(shape=(5, 5, 1), name=f"input_state_{nr}")
input_shape_shape = Input(shape=(5, 5, 1), name=f"input_shape_{nr}")
dropout = 0.5

model = add([input_shape_state, input_shape_shape], name=f"add_{nr}")

model = Conv2D(4, (2, 2), activation="relu", name=f"conv2d_{nr}")(model)
model = Dropout(dropout, name=f"dropout_{nr}")(model)
model = Flatten(name=f"flatten_{nr}")(model)

model = Dense(1, activation="sigmoid", name=f"dense_{nr}")(model)

model = Model(inputs=[input_shape_state, input_shape_shape], outputs=model)

nr=2
input_shape_state_1 = Input(shape=(5, 5, 1), name=f"input_state_{nr}")
input_shape_shape_1 = Input(shape=(5, 5, 1), name=f"input_shape_{nr}")
dropout = 0.5

model_1 = add([input_shape_state_1, input_shape_shape_1], name=f"add_{nr}")

model_1 = Conv2D(4, (2, 2), activation="relu", name=f"conv2d_{nr}")(model_1)
model_1 = Dropout(dropout, name=f"dropout_{nr}")(model_1)
model_1 = Flatten(name=f"flatten_{nr}")(model_1)

model_1 = Dense(1, activation="sigmoid", name=f"dense_{nr}")(model_1) """

#model_1 = Model(inputs=[input_shape_state_1, input_shape_shape_1], outputs=model_1)

model_output = concatenate(models_output)
model_output = Dense(100, activation="softmax")(model_output)
model_output = Model(inputs=models_input, outputs=model_output)

model_output.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.01))

print(model_output.summary())


model_output.save("test6.h5")