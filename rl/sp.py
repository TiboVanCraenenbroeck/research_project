import os
import pathlib
import pickle
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

import numpy as np
from game.logic.game import Game

def create_standard_shape(shape):
    standard_shape = np.zeros((10, 10))
    for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col>0 else 0
    return standard_shape

def change_state_shape_queue(state, shape_queue):
    state = np.array(state, dtype=np.float)
    state[state>0] = 1
    shape_queue = [np.array(shape, dtype=np.float) for shape in shape_queue]
    
    shape_queue = [np.resize(shape,(10, 10, 1)) for shape in shape_queue]
    state = np.resize(state,(10, 10, 1))

    state /= 10
    shape_queue_0 = shape_queue[0]
    shape_queue_1 = shape_queue[1]
    shape_queue_2 = shape_queue[2]

    shape_queue_0 /= 10
    shape_queue_1 /= 10
    shape_queue_2 /= 10
    return state, shape_queue_0, shape_queue_1, shape_queue_2

env = Game(3, False, 8081)

stats = []

done: bool = False
state = env.game_env
shapes_queue = env.shapes_queue
shapes_queue = [create_standard_shape(shape) for shape in shapes_queue]
state, shapes_queue_0, shapes_queue_1, shapes_queue_2 = change_state_shape_queue(state, shapes_queue)

version = input("Version: ")
env.render()
while not done:
    input_shape: int = int(input("Select a shape (0, 1, 2): "))
    input_place_row: int = int(input("Row: "))
    input_place_col: int = int(input("Col: "))
    chosen_action = int(str(input_shape) + str(input_place_row) + str(input_place_col))
    reward, full_lines, state_new, done, shapes_queue_new, uid = env.step(env.shapes_queue[input_shape], input_place_row, input_place_col)
    
    shapes_queue_new = [create_standard_shape(shape) for shape in shapes_queue_new]

    state_new, shapes_queue_0_new, shapes_queue_1_new, shapes_queue_2_new = change_state_shape_queue(state_new, shapes_queue_new)

    stats.append({"reward": reward, "chosen_action": chosen_action, "nr_full_lines": full_lines, "state": state," shapes_queue_0": shapes_queue_0," shapes_queue_1": shapes_queue_1, "shapes_queue_2": shapes_queue_2, "new_state": state_new, "shapes_queue_0_new":shapes_queue_0_new, "shapes_queue_1_new":shapes_queue_1_new, "shapes_queue_2_new":shapes_queue_2_new, "done": done})
    
    state = state_new
    shapes_queue_0, shapes_queue_1, shapes_queue_2 = shapes_queue_0_new, shapes_queue_1_new, shapes_queue_2_new

    env.render()

# save it to a file
pathlib(f"{BASE_DIR}/sp/").mkdir(parents=True, exist_ok=True)
with open(f'{BASE_DIR}/sp/{version}.pkl', 'wb') as f:
    pickle.dump(stats, f)