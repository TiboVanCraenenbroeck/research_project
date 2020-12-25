from typing import List
import numpy as np
from IPython import display
from time import sleep, time, strftime, gmtime
import random, eel, json
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
from models.shape import Shape
from logic.game_view import GameView
class Game:
    def __init__(self):
        # 10x10
        self.game_play: bool = True
        self.game_env = np.zeros((10, 10))
        self.shapes: List[Shape] = []

        self.make_shapes()

        self.total_reward: int = 0 # 3 = 600 | 2 = 300 | 1 = 100
        self.reward_score: dict = {"no": -10, "yes": 10, "line": 20}

        self.shapes_queue: List[Shape] = []
        self.shapes_queue_max: int = 3
        self.get_random_shapes(3)

        self.game_view = GameView()
    
    def make_shapes(self):
        # x x x x x
        self.shapes.append(Shape(1, 5, [[1, 1, 1, 1, 1]]))
        self.shapes.append(Shape(5, 1, [[1], [1], [1], [1], [1]]))
        # x x x x
        self.shapes.append(Shape(1, 4, [[2, 2, 2, 2]]))
        self.shapes.append(Shape(4, 1, [[2], [2], [2], [2]]))
        # x x x
        self.shapes.append(Shape(1, 3, [[3, 3, 3]]))
        self.shapes.append(Shape(3, 1, [[3], [3], [3]]))
        """ 
        x   | xx |  xx |   x
        xx  | x  |   x |  xx
         """
        self.shapes.append(Shape(2, 2, [[4, -1], [4, 4]]))
        self.shapes.append(Shape(2, 2, [[4, 4], [4, -1]]))
        self.shapes.append(Shape(2, 2, [[4, 4], [-1, 4]]))
        self.shapes.append(Shape(2, 2, [[-1, 4], [4, 4]]))


        """ 
            xx
            xx
         """
        self.shapes.append(Shape(2, 2, [[5, 5], [5, 5]]))

        """ x """
        self.shapes.append(Shape(1, 1, [[6]]))

        """ 
        xxx
        xxx
        xxx
         """
        self.shapes.append(Shape(3, 3, [[7, 7, 7], [7, 7, 7], [7, 7, 7]]))

        """ xx """
        self.shapes.append(Shape(1, 2, [[8, 8]]))
        self.shapes.append(Shape(2, 1, [[8], [8]]))

        """ 
        x         |    x x x     |    x x x    |        x
        x         |        x     |    x        |        x
        x x x     |        x     |    x        |    x x x
         """
        self.shapes.append(Shape(3, 3, [[9, -1, -1], [9, -1, -1], [9, 9, 9]]))
        self.shapes.append(Shape(3, 3, [[9, 9, 9], [-1, -1, 9], [-1, -1, 9]]))
        self.shapes.append(Shape(3, 3, [[9, 9, 9], [9, -1, -1], [9, -1, -1]]))
        self.shapes.append(Shape(3, 3, [[-1, -1, 9], [-1, -1, 9], [9, 9, 9]]))
    
    def check_on_full_lines(self):
        # Check the rows and columns
        game_env_reversed = self.game_env.copy()
        del_rows: list = self.check_on_full_lines_rows(game_env_reversed)
        game_env_reversed = game_env_reversed.T
        del_cols: list = self.check_on_full_lines_rows(game_env_reversed)

        # Set the rows and columns on 0.0 on the selected lines
        # For the columns
        for col in del_cols:
            game_env_reversed[col] = [0.0 for i in game_env_reversed[col]]
        # For the rows
        game_env_reversed = game_env_reversed.T
        for row in del_rows:
            game_env_reversed[row] = [0.0 for i in game_env_reversed[row]]

        # Copy the game_env_reversed to the self.game_env
        self.game_env = game_env_reversed.copy()

        reward: float = (len(del_rows) + len(del_cols)) * self.reward_score["line"] 
        return reward
    
    def check_on_full_lines_rows(self, env):
        del_rows: list = []
        for index_row, row in enumerate(env):
            min: int = np.min(row)
            if min > 0: del_rows.append(index_row)
        return del_rows
    
    def check_if_user_can_place_the_shapes(self):
        can_set_shapes: int = False
        for index_row, row in enumerate(self.game_env):
            for index_col, col in enumerate(row):
                for shape in self.shapes_queue:
                    can_set_shapes = self.check_space_available(shape, index_row, index_col, False)
                    if can_set_shapes>0:
                        return True
        return False

    def check_space_available(self, shape: Shape, row: int, col: int, change_game_grid: bool = True) -> int:
        # Check if the space is available
        space_available: bool = True
        reward: int = 0
        game_env = self.game_env.copy()
        for r in range(shape.nr_rows):
            # Check cols
            for c in range(shape.nr_cols):
                # Check if the cell is not out of the grid
                if r + row >= 10 or c + col >= 10:
                    space_available = False
                    break
                nr_cols_available = self.game_env[r+row, c+col]
                # Check if it is an empty block
                if shape.shape[r][c] != -1:
                    reward += 1
                    if nr_cols_available>0 and shape.shape[r][c]>0:
                        space_available = False
                        break
                    if change_game_grid:
                        game_env[r+row, c+col] = shape.shape[r][c]
        # Replace the game_env with the new one if their is enough space
        if space_available == True:
            self.game_env = game_env
            reward *= self.reward_score["yes"]
        else:
            reward = -10
        self.total_reward += reward
        return reward
    
    def remove_shape(self, shape: Shape) -> None:
        self.shapes_queue.remove(shape)
    
    def get_random_shapes(self, n: int = 1) -> None:
        for i in range(n):
            if len(self.shapes_queue)<self.shapes_queue_max:
                self.shapes_queue.append(random.choice(self.shapes))

    def step(self, shape: Shape, row: int, col: int):
        done: bool = False
        # Set the shape in the game env
        reward: int = self.check_space_available(shape, row, col)
        # Add a new random chape to the queue
        if reward>0:
            self.remove_shape(shape)
            self.get_random_shapes()

        # TODO: Add the iteration-number to the return
        # TODO: Create a function that checks if there is enough space for the shapes
        # TODO:Check error als shape buiten game_grid komt
        reward += self.check_on_full_lines()
        self.check_if_user_can_place_the_shapes()
        self.game_view.create_screenshot(self.game_env, self.shapes_queue)
        return reward, self.game_env, done, self.shapes_queue
        
    def render(self) -> None:
        game_env = json.dumps(self.game_env.tolist())
        shape_queue: list = [{"nr_rows": sq.nr_rows, "nr_cols": sq.nr_cols, "shape": sq.shape} for sq in self.shapes_queue]
        self.game_view.change_game_view(game_env, self.total_reward, shape_queue)

"""         display.clear_output(wait=True)
        print(self.game_env)
        print(self.total_reward)
        print(self.shapes_queue) """

a = Game()
eel.sleep(3)
a.render()
eel.sleep(3)
"""eel.sleep(3)
a.step(a.shapes_queue[0], 0, 1)

a.render()
eel.sleep(3)
a.step(a.shapes_queue[0], 5, 5)
a.render()"""


while a.game_play:
    input_shape: int = int(input("Select a shape (0, 1, 2): "))
    input_place_row: int = int(input("Row: "))
    input_place_col: int = int(input("Col: "))
    a.step(a.shapes_queue[input_shape], input_place_row, input_place_col)
    a.render()
    eel.sleep(1)



while True:
    eel.sleep(1)