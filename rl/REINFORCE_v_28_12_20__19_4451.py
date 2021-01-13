from game.logic.game import Game
import pickle
from pathlib import Path
from datetime import datetime
import eel
import time
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate
from tensorflow import keras
import cv2
import numpy as np
import sys
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

disable_eager_execution()


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)


class REINFORCE:
    def __init__(self, path=None):
        self.env = Game(1, False)
        eel.sleep(1)
        self.state_shape = (100)  # the state space
        self.action_shape = 100  # the action space
        self.gamma = 0.6  # decay rate of past observations
        self.alpha = 1e-5  # learning rate of gradient
        self.learning_rate = 0.0001  # learning of deep learning model

        self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        self.model_name = f"model_{self.agent_name}.h5"
        Path(f"{BASE_DIR}/start_models/REINFORCE").mkdir(parents=True, exist_ok=True)
        Path(f"{BASE_DIR}/history_data/REINFORCE").mkdir(parents=True, exist_ok=True)
        if not path:
            self.model = self.build_policy_network()  # build model
            self.save_model()
        else:
            self.model = self.load_model(path)  # import model

        # record observations
        self.states = []
        self.shapes_queues = []
        self.uids = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_rewards = []

    def save_model(self):
        self.model.save(f"{BASE_DIR}/start_models/REINFORCE/{self.model_name}")

    def create_standard_shape(self, shape):
        standard_shape = np.zeros((5, 5))
        for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = col
        return standard_shape

    def build_policy_network(self):
        # BUILD MODEL ####################################################################
        input_game_grid = Input(shape=(10, 10, 1))
        input_shape_queue_0 = Input(shape=(5, 5, 1))
        """ input_shape_queue_1 = Input(shape=(48, 48, 3))
        input_shape_queue_2 = Input(shape=(48, 48, 3)) """

        model_game_grid = Conv2D(8, (3, 3))(input_game_grid)
        model_game_grid = LeakyReLU()(model_game_grid)

        model_game_grid = Flatten()(model_game_grid)

        model_game_grid = Dense(512)(model_game_grid)
        model_game_grid = LeakyReLU()(model_game_grid)

        model_game_grid = Dense(128)(model_game_grid)
        model_game_grid = LeakyReLU()(model_game_grid)

        model_game_grid = Dense(100)(model_game_grid)
        model_game_grid = LeakyReLU()(model_game_grid)
        model_game_grid = Model(inputs=input_game_grid,
                                outputs=model_game_grid, name="Model_GameGrid")

        model_shape_queue_0 = Conv2D(8, (3, 3))(input_shape_queue_0)
        model_shape_queue_0 = LeakyReLU()(model_shape_queue_0)

        model_shape_queue_0 = Flatten()(model_shape_queue_0)
        model_shape_queue_0 = Dense(72)(model_shape_queue_0)
        model_shape_queue_0 = LeakyReLU()(model_shape_queue_0)

        model_shape_queue_0 = Dense(72)(model_shape_queue_0)
        model_shape_queue_0 = LeakyReLU()(model_shape_queue_0)

        model_shape_queue_0 = Dense(25)(model_shape_queue_0)
        model_shape_queue_0 = LeakyReLU()(model_shape_queue_0)

        model_shape_queue_0 = Model(
            inputs=input_shape_queue_0, outputs=model_shape_queue_0, name="Model_ShapeQueue_0")

        """ model_shape_queue_1 = Conv2D(4, (3, 3), activation="relu")(input_shape_queue_1)
        model_shape_queue_1 = Flatten()(model_shape_queue_1)
        model_shape_queue_1 = Model(inputs=input_shape_queue_1, outputs=model_shape_queue_1, name="Model_ShapeQueue_1")

        model_shape_queue_2 = Conv2D(4, (3, 3), activation="relu")(input_shape_queue_2)
        model_shape_queue_2 = Flatten()(model_shape_queue_2)
        model_shape_queue_2 = Model(inputs=input_shape_queue_2, outputs=model_shape_queue_2, name="Model_ShapeQueue_2") """

        combined = concatenate(
            [model_game_grid.output, model_shape_queue_0.output])

        model_output = Dense(125)(combined)
        model_output = LeakyReLU()(model_output)

        model_output = Dense(125)(model_output)
        model_output = LeakyReLU()(model_output)

        model_output = Dense(125)(model_output)
        model_output = LeakyReLU()(model_output)

        model_output = Dense(
            self.action_shape, activation="softmax")(model_output)
        model_output = Model(inputs=[
                             model_game_grid.input, model_shape_queue_0.input], outputs=model_output, name="Model_Output")

        model_output.compile(loss="categorical_crossentropy",
                             optimizer=Adam(lr=self.learning_rate))
        print(model_output.summary())
        return model_output

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def remember(self, state, shapes_queue, uid, action, action_prob, reward):
        # STORE EACH STATE, ACTION AND REWARD into the episodic memory #############################
        encoded_action = self.hot_encode_action(action)
        self.states.append(state)
        self.shapes_queues.append(shapes_queue)
        self.uids.append(uid)
        self.gradients.append(encoded_action-action_prob)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def compute_action(self, state, shape_queue):
        # COMPUTE THE ACTION FROM THE SOFTMAX PROBABILITIES
        shape_queue = np.array([shape_queue], dtype=np.float)
        state = np.array([state], dtype=np.float)

        # Get the images from the iteration
        """ img_game_grid = cv2.imread(f"{self.env.game_view.base_dictionary}/game_gird/game_gird_{uid}.jpg")
        img_shape_queue_0 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}0.jpg") """
        """ img_shape_queue_1 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}1.jpg")
        img_shape_queue_2 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{uid}2.jpg") """

        # Add the imgs to a list + create a numpy array + concatenate it together
        """ imgs_game_grid = np.array([img_game_grid], dtype=np.float)
        imgs_shape_queue_0 = np.array([img_shape_queue_0], dtype=np.float) """
        """ imgs_shape_queue_1 = np.array([img_shape_queue_1], dtype=np.float)
        imgs_shape_queue_2 = np.array([img_shape_queue_2], dtype=np.float) """

        # Scale it (0, 1)
        """ imgs_game_grid /=255
        imgs_shape_queue_0 /=255 """
        """ imgs_shape_queue_1 /=255
        imgs_shape_queue_2 /=255 """
        # Concatenate the lists
        input: list = [state, shape_queue]
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
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (
                cumulative_total_return*self.gamma)+reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards)/(std_rewards+1e-7)  # avoiding zero div

        return norm_discounted_rewards

    def train_policy_network(self):
        # Get the imgs of the episode
        imgs_game_grid: list = []
        imgs_shape_queue0: list = []
        """ imgs_shape_queue1: list = []
        imgs_shape_queue2: list = [] """

        for iteration in self.uids:
            # Get the img
            """ img_game_grid = cv2.imread(f"{self.env.game_view.base_dictionary}/game_gird/game_gird_{iteration}.jpg")
            img_shape_queue_0 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}0.jpg") """
            """ img_shape_queue_1 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}1.jpg")
            img_shape_queue_2 = cv2.imread(f"{self.env.game_view.base_dictionary}/queue_shapes/queue_shapes_{iteration}2.jpg") """
            # Add the img to the list
            """ imgs_game_grid.append(img_game_grid)
            imgs_shape_queue0.append(img_shape_queue_0) """
            """ imgs_shape_queue1.append(img_shape_queue_1)
            imgs_shape_queue2.append(img_shape_queue_2) """
        # Change it to a numpy array
        """ imgs_game_grid = np.array(imgs_game_grid, dtype=np.float)
        imgs_shape_queue0 = np.array(imgs_shape_queue0, dtype=np.float) """
        """ imgs_shape_queue1 = np.array(imgs_shape_queue1, dtype=np.float)
        imgs_shape_queue2 = np.array(imgs_shape_queue2, dtype=np.float) """
        # Scale it (0, 1)
        """ imgs_game_grid /=255
        imgs_shape_queue0 /=255 """
        """ imgs_shape_queue1 /=255
        imgs_shape_queue2 /=255 """

        states = np.array(self.states, dtype=np.float)
        shape_queues = np.array(self.shapes_queues, dtype=np.float)

        # get y_train
        # Met hoeveel de probabiliteit zou moeten veranderen
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        # Rewards gaan uitspreiden --> Vorige acties hebben misschien een aandeel in deze reward (einde)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        gradients = self.alpha*np.vstack([gradients])+self.probs
        y_train = gradients
        history = self.model.train_on_batch([states, shape_queues], y_train)

        self.states, self.shapes_queues, self.probs, self.gradients, self.rewards = [], [], [], [], []
        return history

    def change_state_shape_queue(self, state, shape_queue):
        state = np.array(state, dtype=np.float)
        shape_queue = np.array(shape_queue, dtype=np.float)

        shape_queue = np.resize(shape_queue, (5, 5, 1))
        state = np.resize(state, (10, 10, 1))

        state /= 9
        shape_queue /= 9
        return state, shape_queue

    def train(self, episodes):
        self.total_rewards: list = []
        for episode in range(episodes):
            start_time = time.time()
            uid: str = self.env.reset()
            state = self.env.game_env
            shapes_queue = self.env.shapes_queue[0]
            shapes_queue = self.create_standard_shape(shapes_queue)
            state, shapes_queue = self.change_state_shape_queue(
                state, shapes_queue)

            self.env.render()
            done: bool = False
            total_reward_episode: float = 0
            negative_rewards: int = 0
            while not done:
                action, prob = self.compute_action(state, shapes_queue)
                action_shape, action_row, action_col = self.number_to_arc(
                    action)
                reward, state_new, done, shapes_queue_new, uid = self.env.step(
                    self.env.shapes_queue[action_shape], action_row, action_col)
                shapes_queue_new = self.create_standard_shape(
                    shapes_queue_new[0])

                state_new, shapes_queue_new = self.change_state_shape_queue(
                    state_new, shapes_queue_new)

                self.remember(state, shapes_queue, uid, action, prob, reward)
                state = state_new
                shapes_queue = shapes_queue_new
                total_reward_episode += reward
                self.env.render()
                # Stop the episode if the agent makes 10 mistakes
                if reward <= 0:
                    negative_rewards += 1
                    if negative_rewards >= 1:
                        done = True
                elif reward > 0 and negative_rewards > 3:
                    negative_rewards = 0

            history = self.train_policy_network()
            self.total_rewards.append(total_reward_episode)
            if episode % 25 == 0:
                self.save_model()
                print(
                    f"Episode: {episode} - total reward: {total_reward_episode} | Duration: {time.time()-start_time}")
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


N_EPISODES = 1000000

agent = REINFORCE()

agent.train(N_EPISODES)