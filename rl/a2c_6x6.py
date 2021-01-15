import math
import tensorflow as tf
from tensorflow.python.keras.backend import conv2d
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import sys
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import backend as K
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

from game.logic.game import Game

class A2C:
    def __init__(self, number_actions: int, learning_rate_actor:int, learning_rate_critic:int, gamma, port, path:str = None):
        self.number_actions = number_actions
        self.action_space = [i for i in range(self.number_actions)]
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.gamma = gamma

        self.actor_nn, self.actor_learn_nn, self.critic_nn = self.build_nn()
        self.stats: dict = {"total_rewards" : [], "chosen_action": [], "nr_full_lines" : [], "nr_actions_true": [], "nr_actions_false": []}
        if not path:
            self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
            self.base_path = f"{BASE_DIR}/start_models/a2c/{self.agent_name}"     
            Path(self.base_path).mkdir(parents=True, exist_ok=True)
        else:
            self.agent_name = path
            self.base_path = f"{BASE_DIR}/start_models/a2c/{self.agent_name}"

            self.actor_nn = keras.models.load_model(f"{self.base_path}/actor_nn.h5")
            self.actor_learn_nn = keras.models.load_model(f"{self.base_path}/actor_learn_nn.h5")
            self.critic_nn = keras.models.load_model(f"{self.base_path}/critic_nn.h5")
            with open(f'{BASE_DIR}/history_data/REINFORCE/game_history_data_{self.agent_name}.pkl', 'rb') as f:
                self.stats = pickle.load(f)

        self.reset_memory()

        self.env = Game(1, False, port, 6)
    
    def save_model(self):
        self.actor_nn.save(f"{self.base_path}/actor_nn.h5")
        self.actor_learn_nn.save(f"{self.base_path}/actor_learn_nn.h5")
        self.critic_nn.save(f"{self.base_path}/critic_nn.h5")
    
    def reset_memory(self):
        self.states, self.next_states, self.shapes, self.next_shapes, self.rewards, self.actions, self.done = [], [], [], [], [], [], []

    def add_to_memory(self, state, next_state, shape, next_shape, action, reward, done) -> None:
        self.states.append(state)
        self.next_states.append(next_state)

        self.shapes.append(shape)
        self.next_shapes.append(next_shape)
        # Action one-hot
        action_onehot = np.zeros([self.number_actions])
        action_onehot[action] = 1
        self.actions.append(action_onehot)

        self.rewards.append(reward)
        self.done.append(done)

    def build_nn(self):
        input_game_grid = Input(shape=(6, 6, 1))
        input_shape = Input(shape=(5, 5, 1))
        input_delta = Input(shape=[1])
        
        def custom_actor_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*input_delta)

        model_game_grid = Conv2D(8, (3, 3), activation="relu")(input_game_grid)
        model_game_grid = Flatten()(model_game_grid)
        model_game_grid = Dense(128, activation="relu")(model_game_grid)
        """ model_game_grid = Dense(128, activation="relu")(model_game_grid)
        model_game_grid = Dense(128, activation="relu")(model_game_grid) """
        model_game_grid = Model(inputs=[input_game_grid], outputs=[model_game_grid])
        
        model_shape = Conv2D(8, (3, 3), activation="relu")(input_shape)
        model_shape = Flatten()(model_shape)
        model_shape = Dense(72, activation="relu")(model_shape)
        """ model_shape = Dense(72, activation="relu")(model_shape)
        model_shape = Dense(72, activation="relu")(model_shape) """
        model_shape = Model(inputs=[input_shape], outputs=[model_shape])

        model = concatenate([model_game_grid.output, model_shape.output])

        model = Dense(200, activation="relu")(model)
        model = Dense(100, activation="relu")(model)
        model = Dense(50, activation="relu")(model)

        actor = Dense(self.number_actions, activation="softmax")(model)
        critic = Dense(1, activation="linear")(model)

        actor_learn = Model(inputs=[input_game_grid, input_shape, input_delta], outputs=[actor])
        actor = Model(inputs=[input_game_grid, input_shape], outputs=[actor])
        critic = Model(inputs=[input_game_grid, input_shape], outputs=[critic])

        actor_learn.compile(loss=custom_actor_loss, optimizer=Adam(lr=self.learning_rate_actor))
        critic.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate_critic))

        return actor, actor_learn, critic
    
    def train_nn(self):
        if len(self.states)>0:
            states = np.array(self.states, dtype=np.float)
            next_states = np.array(self.next_states, dtype=np.float)
            
            shapes = np.array(self.shapes, dtype=np.float)
            next_shapes = np.array(self.next_shapes, dtype=np.float)

            value = self.critic_nn.predict([states, shapes])
            next_values = self.critic_nn.predict([next_states, next_shapes])

            targets = [self.rewards[i] + self.gamma * next_values[i] * (1-int(done)) for i, done in enumerate(self.done)]
            deltas = [target - value[i] for i, target in enumerate(targets)] # TODO: Zet de ongeldige acties op 0!!!

            targets = np.array(targets, dtype=np.float)
            deltas = np.array(deltas, dtype=np.float)

            actions = np.array(self.actions)
            
            self.actor_learn_nn.fit([states, shapes, deltas], actions, epochs=1, verbose=0)
            self.critic_nn.fit([states, shapes], targets, epochs=1, verbose=0)
            self.reset_memory()     


    def create_standard_shape(self, shape):
        standard_shape = np.zeros((5, 5))
        for index_row, row in enumerate(shape.shape):
                for index_col, col in enumerate(row):
                    standard_shape[index_row, index_col] = 1 if col>0 else 0
        standard_shape = np.array(standard_shape, dtype=np.float)
        standard_shape = np.resize(standard_shape, (5, 5, 1))
        standard_shape /= 10
        return standard_shape
    
    def compute_action(self, state, shapes_queue):
        state = np.array([state], dtype=np.float)
        shapes_queue = np.array([shapes_queue], dtype=np.float)
        input = [state, shapes_queue]
        probs = self.actor_nn.predict(input)[0].flatten()

        action = np.random.choice(self.action_space,p=probs)

        return action, probs
    
    def number_to_arc(self, number: int):
        a = 0 if number<=35 else 1 if number<=71 else 2
        #number = number if a==0 else number - 36 if a==1 else number - 72
        r = math.floor(number/6)
        c = number if r == 0 else number - (6 * r)
        return a, r, c
    
    def reward_function(self, reward, full_lines, negative_rewards, done):
        done_old = done
        done: bool = False
        if reward <= 0:
            negative_rewards += 1
            reward = -1
            if negative_rewards >= 3:
                done = True
        elif reward > 0 and negative_rewards > 0:
            negative_rewards = 0
            reward = 1
        else:
            reward = 1
        reward += full_lines * 10
        done = done_old if done_old else done
        return reward, negative_rewards, done


    def train(self, episodes: int):
        for episode in range(episodes):
            # Get start-time
            start_time = time.time()

            # Reset the env
            uid: str = self.env.reset()
            state = self.env.game_env
            state[state>0] = 1
            state[state<=0] = 0
            state = np.resize(state,(6, 6, 1))
            shapes_queue = self.env.shapes_queue[0]
            shapes_queue = self.create_standard_shape(shapes_queue)
            self.env.render()

            # Game stats
            done: bool = False
            negative_rewards: int = 0
            total_reward_episode: float = 0
            total_actions = []
            nr_full_lines = 0
            nr_actions_true: int = 0
            nr_actions_false: int = 0

            # Iterations
            while not done:
                action, prob = self.compute_action(state, shapes_queue)
                action_shape, action_row, action_col = self.number_to_arc(action)
                reward, full_lines, state_new, done, shapes_queue_new, uid = self.env.step(self.env.shapes_queue[action_shape], action_row, action_col)
                shapes_queue_new = self.create_standard_shape(shapes_queue_new[0])
                state_new[state_new>0] = 1
                state_new[state_new<=0] = 0
                state_new = np.resize(state_new,(6, 6, 1))

                reward, negative_rewards, done = self.reward_function(reward, full_lines, negative_rewards, done)

                # Add the data to the memory
                self.add_to_memory(state, state_new, shapes_queue, shapes_queue_new, action, reward, done)
                state = state_new
                shapes_queue = shapes_queue_new
                self.env.render()

                # Stats
                total_reward_episode += reward
                nr_full_lines += full_lines
                total_actions.append(action)
                if reward>0:
                    nr_actions_true += 1
                else:
                    nr_actions_false += 1

            
            self.train_nn()
            self.stats["total_rewards"].append(total_reward_episode)
            self.stats["chosen_action"].append(total_actions)
            self.stats["nr_full_lines"].append(nr_full_lines)
            self.stats["nr_actions_true"].append(nr_actions_true)
            self.stats["nr_actions_false"].append(nr_actions_false)
            if episode % 10000 == 0:
                try:
                    self.save_model()
                    with open(f'{BASE_DIR}/history_data/REINFORCE/game_history_data_{self.agent_name}.pkl', 'wb') as f:
                        pickle.dump(self.stats, f)
                    print(f"Episode: {episode} - total reward: {total_reward_episode} | Duration: {time.time()-start_time}")
                except:
                    print("Saved failed!")



lr_actor = 1e-2
lr_critic = 1e-2
""" lr_actor = 1e-3
lr_critic = 1e-3 """
gamma: float = 0.99
agent = A2C(36, lr_actor, lr_critic, gamma, 8084)
agent.train(1000000)