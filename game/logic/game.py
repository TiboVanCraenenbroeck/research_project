from typing import List
import numpy as np
from IPython import display
from time import sleep, time, strftime, gmtime
import random, eel, json
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from models.shape import Shape
from logic.game_view import GameView
class Game:
    def __init__(self, shape_queue_max: int = 3, create_screenshot_check: bool = True, port: int = 80, env_size: int = 10):
        self.env_size = env_size
        # 10x10
        self.shapes: List[Shape] = []

        self.make_shapes()

        self.reward_score: dict = {"no": -1, "yes": 1, "line": 3}

        self.shapes_queue_max: int = shape_queue_max
        self.create_screenshot_check = create_screenshot_check
        self.game_view = GameView(self.create_screenshot_check, port, self.env_size)
        self.reset()
    
    def reset(self):
        self.game_play = True
        self.game_env = np.zeros((self.env_size,self.env_size))
        self.total_reward: int = 0 # 3 = 600 | 2 = 300 | 1 = 100
        self.shapes_queue: List[Shape] = []
        self.get_random_shapes(self.shapes_queue_max)
        self.game_view.reset()
        uid: str = ""
        if self.create_screenshot_check:
            uid = self.game_view.create_screenshot(self.game_env, self.shapes_queue)
        return uid
    
    def make_shapes(self):
        # x x x x x
        if self.env_size >4:
            self.shapes.append(Shape(1, 5, [[1, 1, 1, 1, 1]]))
            self.shapes.append(Shape(5, 1, [[1], [1], [1], [1], [1]]))
            # x x x x
            self.shapes.append(Shape(1, 4, [[2, 2, 2, 2]]))
            self.shapes.append(Shape(4, 1, [[2], [2], [2], [2]]))
        
            """ 
            xxx
            xxx
            xxx
            """
            self.shapes.append(Shape(3, 3, [[7, 7, 7], [7, 7, 7], [7, 7, 7]]))

            """ 
            x         |    x x x     |    x x x    |        x
            x         |        x     |    x        |        x
            x x x     |        x     |    x        |    x x x
            """
            self.shapes.append(Shape(3, 3, [[9, -1, -1], [9, -1, -1], [9, 9, 9]]))
            self.shapes.append(Shape(3, 3, [[9, 9, 9], [-1, -1, 9], [-1, -1, 9]]))
            self.shapes.append(Shape(3, 3, [[9, 9, 9], [9, -1, -1], [9, -1, -1]]))
            self.shapes.append(Shape(3, 3, [[-1, -1, 9], [-1, -1, 9], [9, 9, 9]]))

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


        """ xx """
        self.shapes.append(Shape(1, 2, [[8, 8]]))
        self.shapes.append(Shape(2, 1, [[8], [8]])) 

    
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
    
    def check_if_user_can_place_the_shapes(self, return_position=False):
        can_set_shapes: bool = False
        for index_row, row in enumerate(self.game_env):
            for index_col, col in enumerate(row):
                for i, shape in enumerate(self.shapes_queue):
                    can_set_shapes = self.check_space_available(shape, index_row, index_col, False)
                    if can_set_shapes>0:
                        if return_position:
                            return False, i, index_row, index_col
                        else:
                            return False
        return True

    def check_space_available(self, shape: Shape, row: int, col: int, change_game_grid: bool = True) -> int:
        # Check if the space is available
        space_available: bool = True
        reward: int = 0
        game_env = self.game_env.copy()
        for r in range(shape.nr_rows):
            # Check cols
            for c in range(shape.nr_cols):
                # Check if the cell is not out of the grid
                if r + row >= self.env_size or c + col >= self.env_size:
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
            reward = self.reward_score["no"]
        if change_game_grid:
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

        # TODO: Check function that checks if there is enough space for the shapes
        full_lines: int = self.check_on_full_lines()
        reward += full_lines
        done = self.check_if_user_can_place_the_shapes()
        uid: str = ""
        if self.create_screenshot_check:
            uid = self.game_view.create_screenshot(self.game_env, self.shapes_queue)
        return reward, full_lines, self.game_env, done, self.shapes_queue, uid
        
    def render(self) -> None:
        game_env = json.dumps(self.game_env.tolist())
        shape_queue: list = [{"nr_rows": sq.nr_rows, "nr_cols": sq.nr_cols, "shape": sq.shape} for sq in self.shapes_queue]
        # Check if the user will see the changes in the app
        if self.game_view.render:
            self.game_view.change_game_view(game_env, self.total_reward, shape_queue)


# TODO: Render function on and off --> To slow
# TODO: Naam teruggeven bij het maken van een nieuwe map
# TODO: game_state --> Als er geen acties meer mogelijk zijn --> Deze op false zetten