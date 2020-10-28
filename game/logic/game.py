from typing import List
import numpy as np
from IPython import display
from time import sleep

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
from models.shape import Shape
class Game:
    def __init__(self):
        # 10x10
        self.game_env = np.zeros((10, 10))
        self.shapes: List[Shape] = []

        self.total_reward: int = 0 # 3 = 600 | 2 = 300 | 1 = 100
        self.reward_score: dict = {"no": -10, "yes": 10, "line": 20}
    
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
        self.shapes.append(Shape(2, 2, [[4], [4, 4]]))
        self.shapes.append(Shape(2, 2, [[4, 4], [4]]))
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

    
    def check_space_available(self, shape: Shape, row: int, col: int) -> bool:
        # Check if the space is available
        space_available: bool = True
        reward: int = 0
        game_env = self.game_env.copy()
        for r in range(shape.nr_rows):
            # Check cols
            for c in range(shape.nr_cols):
                nr_cols_available = self.game_env[r+row, c+col]
                # Check if it is an empty block
                if shape.shape[r][c] != -1:
                    reward += 1
                    if nr_cols_available>0 and shape.shape[r][c]>0:
                        space_available = False
                        break
                    game_env[r+row, c+col] = shape.shape[r][c]
        # Replace the game_env with the new one if their is enough space
        if space_available == True:
            self.game_env = game_env
            reward *= self.reward_score["yes"]
        else:
            reward = -10
        self.total_reward += reward
        return reward
    
    def get_random_shape(self):
        pass

    def step(self):
        pass
    
    def add_shape_to_game_env(self, shape: Shape, row: int, col:int) -> int:
        points: int = -10
        # Check if the space is available
        if self.check_space_available(shape, row, col):
            # Add the shape in the grid
            pass
        return points
        
    def render(self):
        display.clear_output(wait=True)
        print(self.game_env)
        print(self.total_reward)

a = Game()
a.make_shapes()
a.render()

a.check_space_available(a.shapes[0], 0, 1)