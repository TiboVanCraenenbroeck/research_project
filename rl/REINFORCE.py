import os
import sys
import numpy as np
import cv2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.python.framework.ops import disable_eager_execution
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
disable_eager_execution()

import eel
from datetime import datetime
from pathlib import Path
import pickle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)

from game.logic.game import Game

class REINFORCE:
    def __init__(self, path=None):
        self.env = Game()
        eel.sleep(1)
        self.state_shape = (100)  # the state space
        self.action_shape = 300  # the action space
        self.gamma = 0.3  # decay rate of past observations
        self.alpha = 1e-4  # learning rate of gradient
        self.learning_rate = 0.0001  # learning of deep learning model
        
        self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        self.model_name = f"model_{self.agent_name}.h5"
        if not path:
            self.model = self.build_policy_network()  # build model
            self.save_model()
        else:
            self.model = self.load_model(path)  # import model

        # record observations
        self.uids = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.actions = []
        self.stats: dict = {"total_rewards" : [], "chosen_action": []}
    
    def save_model(self):
        Path(f"{BASE_DIR}/start_models/REINFORCE").mkdir(parents=True, exist_ok=True)
        self.model.save(f"{BASE_DIR}/start_models/REINFORCE/{self.model_name}")

    def build_policy_network(self):
        # BUILD MODEL ####################################################################
        input_game_grid = Input(shape=(78, 78, 3))
        input_shape_queue_0 = Input(shape=(48, 48, 3))
        input_shape_queue_1 = Input(shape=(48, 48, 3))
        input_shape_queue_2 = Input(shape=(48, 48, 3))

        model_game_grid = Conv2D(3, (3, 3))(input_game_grid)
        model_game_grid = Conv2D(3, (3, 3))(model_game_grid)
        model_game_grid = Flatten()(model_game_grid)
        model_game_grid = Model(inputs=input_game_grid,outputs=model_game_grid, name="Model_GameGrid")

        model_shape_queue_0 = Conv2D(3, (3, 3), activation="relu")(input_shape_queue_0)
        model_shape_queue_0 = Conv2D(3, (3, 3), activation="relu")(model_shape_queue_0)
        model_shape_queue_0 = Flatten()(model_shape_queue_0)
        model_shape_queue_0 = Model(inputs=input_shape_queue_0, outputs=model_shape_queue_0, name="Model_ShapeQueue_0")

        model_shape_queue_1 = Conv2D(3, (3, 3), activation="relu")(input_shape_queue_1)
        model_shape_queue_1 = Conv2D(3, (3, 3), activation="relu")(model_shape_queue_1)
        model_shape_queue_1 = Flatten()(model_shape_queue_1)
        model_shape_queue_1 = Model(inputs=input_shape_queue_1, outputs=model_shape_queue_1, name="Model_ShapeQueue_1")

        model_shape_queue_2 = Conv2D(3, (3, 3), activation="relu")(input_shape_queue_2)
        model_shape_queue_2 = Conv2D(3, (3, 3), activation="relu")(model_shape_queue_2)
        model_shape_queue_2 = Flatten()(model_shape_queue_2)
        model_shape_queue_2 = Model(inputs=input_shape_queue_2, outputs=model_shape_queue_2, name="Model_ShapeQueue_2")

        combined = concatenate([model_game_grid.output, model_shape_queue_0.output,model_shape_queue_1.output, model_shape_queue_2.output])

        model_output = Dense(1024, activation="relu")(combined)
        model_output = Dense(1024, activation="relu")(model_output)
        model_output = Dense(512, activation="relu")(model_output)
        model_output = Dense(512, activation="relu")(model_output)
        model_output = Dense(self.action_shape, activation="softmax")(model_output)
        model_output = Model(inputs=[model_game_grid.input, model_shape_queue_0.input, model_shape_queue_1.input, model_shape_queue_2.input], outputs=model_output, name="Model_Output")

        model_output.compile(loss="categorical_crossentropy",
                             optimizer=Adam(lr=self.learning_rate))

        """ model_shape_queue_0 = Flatten()(input_shape_queue_0)

        model_shape_queue_0 = Dense(100,kernel_initializer=initializers.he_normal())(model_shape_queue_0)
        model_shape_queue_0 = LeakyReLU()(model_shape_queue_0) """
        #model_shape_queue_0 = Model(inputs=input_shape_queue_0, outputs=model_shape_queue_0, name="Model_ShapeQueue_0")
        
        """ model_shape_queue_1 = Conv2D(representations_conv, (1, 1), kernel_initializer=initializers.he_normal())(input_shape_queue_1)
        model_shape_queue_1 = LeakyReLU()(model_shape_queue_1) """

        """ model_shape_queue_1 = Flatten()(input_shape_queue_1)

        model_shape_queue_1 = Dense(100,kernel_initializer=initializers.he_normal())(model_shape_queue_1)
        model_shape_queue_1 = LeakyReLU()(model_shape_queue_1) """
        #model_shape_queue_1 = Model(inputs=input_shape_queue_1, outputs=model_shape_queue_1, name="Model_ShapeQueue_1")
        
        """ model_shape_queue_2 = Conv2D(representations_conv, (1, 1), kernel_initializer=initializers.he_normal())(input_shape_queue_2)
        model_shape_queue_2 = LeakyReLU()(model_shape_queue_2) """

        """ model_shape_queue_2 = Flatten()(input_shape_queue_2)

        model_shape_queue_2 = Dense(100,kernel_initializer=initializers.he_normal())(model_shape_queue_2)
        model_shape_queue_2 = LeakyReLU()(model_shape_queue_2)  """
        #model_shape_queue_2 = Model(inputs=input_shape_queue_2, outputs=model_shape_queue_2, name="Model_ShapeQueue_2")


        model_output = concatenate([input_game_grid, input_shape_queue_0, input_shape_queue_1, input_shape_queue_2])
        model_output = Conv2D(4, (2, 2), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)
        
        model_output = Conv2D(4, (2, 2), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)
        
        model_output = Conv2D(4, (2, 2), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)
        
        model_output = Flatten()(model_output)
        
        model_output = Dense(self.action_shape, activation="softmax",kernel_initializer=initializers.he_normal())(model_output)
        model_output = Model(inputs=[input_game_grid, input_shape_queue_0, input_shape_queue_1, input_shape_queue_2], outputs=model_output, name="Model_Output")

        model_output.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        print(model_output.summary())
        return model_output

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def remember(self,state, shapes_queue,uid, action, action_prob, reward, action_shape):
        # STORE EACH STATE, ACTION AND REWARD into the episodic memory #############################
        encoded_action = self.hot_encode_action(action)
        self.uids.append(uid)
        self.gradients.append(encoded_action-action_prob)
        self.rewards.append(reward)
        self.probs.append(action_prob)
        self.actions.append(action)

    def compute_action(self, uid):
        # COMPUTE THE ACTION FROM THE SOFTMAX PROBABILITIES
        # Get the images from the iteration
        img_game_grid = cv2.imread(f"{self.env.game_view.base_dictionary}/game_gird/game_gird_{uid}.jpg")
        img_shape_queue_0 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}0.jpg")
        img_shape_queue_1 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}1.jpg")
        img_shape_queue_2 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}2.jpg")

        # Add the imgs to a list + create a numpy array + concatenate it together
        imgs_game_grid = np.array([img_game_grid], dtype=np.float)
        imgs_shape_queue_0 = np.array([img_shape_queue_0], dtype=np.float)
        imgs_shape_queue_1 = np.array([img_shape_queue_1], dtype=np.float)
        imgs_shape_queue_2 = np.array([img_shape_queue_2], dtype=np.float)

        # Scale it (0, 1)
        imgs_game_grid /=255
        imgs_shape_queue_0 /=255
        imgs_shape_queue_1 /=255
        imgs_shape_queue_2 /=255
        # Concatenate the lists
        input: list = [imgs_game_grid, imgs_shape_queue_0, imgs_shape_queue_1, imgs_shape_queue_2]
        # Get action probability
        action_probability_distribution = self.model.predict(input).flatten()
        # Norm action probability distribution
        action_probability_distribution /= np.sum(
            action_probability_distribution)
        # Sample action
        action = np.random.choice(
            self.action_shape, 1, p=action_probability_distribution)[0]

        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_total_return = 0

        negative_discount_reward_index = []
        # iterate the rewards backwards and and calc the total return
        for i, reward in enumerate(rewards[::-1]):
            cumulative_total_return = (
                cumulative_total_return*self.gamma)+reward
            discounted_rewards.insert(0, cumulative_total_return)
            if reward<=0:
                negative_discount_reward_index.append(i)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards)/(std_rewards+1e-7)  # avoiding zero div

        for i in negative_discount_reward_index:
            norm_discounted_rewards[i][0] = 0            
            self.probs[i][self.actions[i]] = 0
        return norm_discounted_rewards

    def train_policy_network(self):
        # Get the imgs of the episode
        imgs_game_grid: list = []
        imgs_shape_queue0: list = []
        imgs_shape_queue1: list = []
        imgs_shape_queue2: list = []

        for iteration in self.uids:
            # Get the img
            img_game_grid = cv2.imread(f"{self.env.game_view.base_dictionary}/game_gird/game_gird_{iteration}.jpg")
            img_shape_queue_0 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}0.jpg")
            img_shape_queue_1 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}1.jpg")
            img_shape_queue_2 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}2.jpg")
            # Add the img to the list
            imgs_game_grid.append(img_game_grid)
            imgs_shape_queue0.append(img_shape_queue_0)
            imgs_shape_queue1.append(img_shape_queue_1)
            imgs_shape_queue2.append(img_shape_queue_2)
        # Change it to a numpy array
        imgs_game_grid = np.array(imgs_game_grid, dtype=np.float)
        imgs_shape_queue0 = np.array(imgs_shape_queue0, dtype=np.float)
        imgs_shape_queue1 = np.array(imgs_shape_queue1, dtype=np.float)
        imgs_shape_queue2 = np.array(imgs_shape_queue2, dtype=np.float)
        # Scale it (0, 1)
        imgs_game_grid /=255
        imgs_shape_queue0 /=255
        imgs_shape_queue1 /=255
        imgs_shape_queue2 /=255
        # get y_train
        # Met hoeveel de probabiliteit zou moeten veranderen
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        # Rewards gaan uitspreiden --> Vorige acties hebben misschien een aandeel in deze reward (einde)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        gradients = self.alpha*np.vstack([gradients])+self.probs
        y_train = gradients
        history = self.model.train_on_batch([imgs_game_grid, imgs_shape_queue0, imgs_shape_queue1, imgs_shape_queue2], y_train)

        self.uids, self.probs, self.gradients, self.rewards = [], [], [], []
        self.save_model()
        return history

    def train(self, episodes):
        self.stats: dict = {"total_rewards" : [], "chosen_action": [], "nr_full_lines" : [], "nr_actions_true": [], "nr_actions_false": []}
        n_n = 100
        for episode in range(episodes):
            start_time = time.time()
            uid: str = self.env.reset()
            self.env.render()
            done: bool = False
            total_reward_episode: float = 0
            negative_rewards: int = 0
            total_actions = []
            nr_full_lines = 0
            nr_actions_true: int = 0
            nr_actions_false: int = 0
            while not done:
                start_time_1 = time.time()
                action, prob = self.compute_action(uid)
                action_shape, action_row, action_col = self.number_to_arc(
                    action)
                reward, game_env, done, shapes_queue, uid = self.env.step(
                    self.env.shapes_queue[action_shape], action_row, action_col)
                self.remember(uid, action, prob, reward)
                total_reward_episode += reward
                self.env.render()
                # Stop the episode if the agent makes 10 mistakes
                if reward < 0:
                    negative_rewards += 1
                    reward = 0
                    nr_actions_false += 1
                    if negative_rewards >= 100:
                        done = True
                elif reward > 0 and negative_rewards>0:
                    negative_rewards = 0
                    reward = 2
                    nr_actions_true += 1
                else:
                    reward = 2
                    nr_actions_true += 1
                reward += full_lines * 10
                nr_full_lines += full_lines
                self.remember(state, shapes_queue, uid, action, prob, reward, action_shape)
                state = state_new

            history = self.train_policy_network()
            self.stats["total_rewards"].append(total_reward_episode)
            self.stats["chosen_action"].append(total_actions)
            self.stats["nr_full_lines"].append(nr_full_lines)
            self.stats["nr_actions_true"].append(nr_actions_true)
            self.stats["nr_actions_false"].append(nr_actions_false)
            if episode % 10000 == 0:
                #n_n = 0 if n_n <= 0 else n_n-1
                self.save_model()
                with open(f'{BASE_DIR}/history_data/REINFORCE/game_history_data_{self.agent_name}.pkl', 'wb') as f:
                    pickle.dump(self.stats, f)
                print(f"Episode: {episode} - total reward: {total_reward_episode} | Duration: {time.time()-start_time}")
            with open(f'{BASE_DIR}/history_data/REINFORCE/game_history_data_{self.agent_name}.pkl', 'wb') as f:
                pickle.dump(self.total_rewards, f)

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def number_to_arc(self, number: int):
        number /= 100
        number_str = f"{number:.2f}".replace('.', '')
        number_list = list(map(int, number_str))
        return number_list[0], number_list[1], number_list[2]


N_EPISODES = 1000

agent = REINFORCE()

agent.train(N_EPISODES)
