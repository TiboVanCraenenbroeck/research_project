import tensorflow as tf
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from datetime import datetime
import eel
import time
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import class_weight
from tensorflow.keras import initializers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add, Dropout, Reshape
from tensorflow import keras
import cv2
import sys
import os
import numpy as np
import random
import collections
import math

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from game.logic.game import Game



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)


class REINFORCE:
    def __init__(self, path=None, replay_memory_size: int = 1000, port: int = 8080):
        self.env = Game(3, False, port, 6)
        eel.sleep(1)
        self.state_shape = (100)  # the state space
        self.action_shape = 36  # the action space
        self.action_shape_total = 108
        self.gamma = 0.9  # decay rate of past observations
        self.alpha = 1e-3  # learning rate of gradient
        self.learning_rate = 0.01  # learning of deep learning model

        self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        self.model_name = f"model_{self.agent_name}.h5"
        self.model_chance_space_availible_name = f"model_chance_space_availible_{self.agent_name}.h5"
        Path(f"{BASE_DIR}/start_models/REINFORCE").mkdir(parents=True, exist_ok=True)
        Path(f"{BASE_DIR}/history_data/REINFORCE").mkdir(parents=True, exist_ok=True)

        self.tensorboard = TensorBoard(
            log_dir='./logs/{}'.format(self.agent_name))

        self.space_availible_shapes_true = collections.deque(
            maxlen=replay_memory_size)
        self.space_availible_shapes_false = collections.deque(
            maxlen=replay_memory_size)
        self.batch_size: int = 64
        self.nr_iterations_update_weights = 5000
        self.policy_nn_can_update = False
        self.policy_nn_can_train = False
        self.train_chance_space_available_nn_after = 1
        self.best_accuracy_updating_policy_nn: float = 0.91

        if not path:
            self.model_chance_space_availible = self.build_nn_chance_space_availible(
                -1, True)
            self.model = self.build_policy_network()  # build model
            self.save_model()
        else:
            self.model = self.build_policy_network()  # import model
            self.model = keras.models.load_model(path)

        # record observations
        self.states = []
        self.shapes = []
        self.uids = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.actions = []
        self.stats: dict = {"total_rewards": [], "chosen_action": []}

    def save_model(self):
        self.model.save(f"{BASE_DIR}/start_models/REINFORCE/{self.model_name}")
        self.model_chance_space_availible.save(
            f"{BASE_DIR}/start_models/REINFORCE/{self.model_chance_space_availible_name}")

    def build_nn_chance_space_availible(self, nr, trainable: bool = False):
        input_shape_state = Input(shape=(5, 5, 1), name=f"input_state_{nr}")
        input_shape_shape = Input(shape=(5, 5, 1), name=f"input_shape_{nr}")
        dropout = 0.5

        model = add([input_shape_state, input_shape_shape], name=f"add_{nr}")

        model = Conv2D(4, (2, 2), activation="relu",
                       name=f"conv2d_{nr}")(model)
        model = Dropout(dropout, name=f"dropout_{nr}")(model)
        model = Flatten(name=f"flatten_{nr}")(model)

        model = Dense(1, activation="sigmoid", name=f"dense_{nr}")(model)

        model = Model(inputs=[input_shape_state,
                              input_shape_shape], outputs=model)
        model.trainable = trainable
        if trainable:
            model.compile(loss="binary_crossentropy",
                          optimizer=Adam(lr=0.01), metrics=['accuracy'])
            return model
        return model, input_shape_state, input_shape_shape

    def create_standard_shape(self, shape):
        standard_shape = np.zeros((5, 5))
        for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col > 0 else 0
        return standard_shape

    def build_policy_network(self):
        # BUILD MODEL ####################################################################
        models = []
        models_input = []
        models_output = []
        for i in range(36):
            model_single, input_shape_state, input_shape_shape = self.build_nn_chance_space_availible(
                i)
            models.append(model_single)
            models_input.append(input_shape_state)
            models_input.append(input_shape_shape)
            models_output.append(model_single.output)

        model_output = concatenate(models_output)

        model_output = Reshape((6, 6, 1))(model_output)
        
        model_output = Conv2D(4, (3, 3), activation="relu", name=f"conv2d_output")(model_output)
        model_output = Flatten(name=f"flatten_output")(model_output)

        #model_output = Dense(100, activation="relu", name="dense_output_0")(model_output)

        model_output = Dense(36, activation="softmax",
                             name="dense_output")(model_output)

        model_output = Model(inputs=models_input, outputs=model_output)

        model_output.compile(loss="categorical_crossentropy",
                             optimizer=Adam(lr=self.learning_rate))

        return model_output

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def remember(self, state, shape, uid, action, action_prob, reward, action_shape):
        # STORE EACH STATE, ACTION AND REWARD into the episodic memory #############################
        encoded_action = self.hot_encode_action(action)
        self.states.append(state)
        self.shapes.append(shape)

        self.uids.append(uid)
        action_r = 0 if action_shape == 0 else 36 if action_shape == 1 else 72
        action_prob = action_prob[action_r:action_r + 36]
        self.gradients.append(encoded_action-action_prob)
        self.rewards.append(reward)
        self.probs.append(action_prob)
        self.actions.append(action)

    def add_to_memory(self, sa: bool, X_train_state, X_train_shape, y_train):
        step = (X_train_state, X_train_shape, y_train)
        if sa:
            self.space_availible_shapes_true.append(step)
        else:
            self.space_availible_shapes_false.append(step)

    def compute_action(self, states, shape_queue_0, shape_queue_1, shape_queue_2):
        shape_queue = [shape_queue_0, shape_queue_1, shape_queue_2]
        action_probability_distribution = np.array([])
        for shape in shape_queue:
            input_nn = [np.array([output])
                        for state in states for output in (state, shape)]

            action_probability_distribution_shape = self.model.predict(
                input_nn).flatten()
            action_probability_distribution = np.concatenate(
                (action_probability_distribution, action_probability_distribution_shape), axis=None)

        # Norm action probability distribution
        action_probability_distribution /= np.sum(
            action_probability_distribution)

        # Sample action
        action = np.random.choice(
            self.action_shape_total, 1, p=action_probability_distribution)[0]

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
            if reward <= 0:
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

    def train_policy_network(self, nr_iterations_update_weights_left):
        # Train the policy nn only when the accuracy of the chance-nn >= 0.7
        history = None
        if self.policy_nn_can_train:
            X_train = [np.array(output, dtype=np.float) for i in range(36) for output in (
                np.asarray(list(zip(*self.states))[i], dtype=float), self.shapes)]

            # get y_train
            # Met hoeveel de probabiliteit zou moeten veranderen
            gradients = np.vstack(self.gradients)
            rewards = np.vstack(self.rewards)
            # Rewards gaan uitspreiden --> Vorige acties hebben misschien een aandeel in deze reward (einde)
            discounted_rewards = self.get_discounted_rewards(rewards)
            gradients *= discounted_rewards
            gradients = self.alpha*np.vstack([gradients])+self.probs

            y_train = gradients
            history = self.model.train_on_batch(X_train, y_train)
            #history = self.model.fit(X_train, y_train, epochs=1)
        self.states, self.shapes, self.probs, self.gradients, self.rewards, self.actions = [
        ], [], [], [], [], []
        return history

    def train_chance_space_available_nn(self, nr_iterations_update_weights_left):
        # Train only the nn after n itterations
        if len(self.space_availible_shapes_true) >= self.batch_size and len(self.space_availible_shapes_false) >= self.batch_size and nr_iterations_update_weights_left % self.train_chance_space_available_nn_after == 0:
            samples_true = random.sample(
                self.space_availible_shapes_true, self.batch_size)
            samples_false = random.sample(
                self.space_availible_shapes_false, self.batch_size)
            mini_batches = samples_true + samples_false
            X_train_state = np.asarray(
                list(zip(*mini_batches))[0], dtype=float)
            X_train_shape = np.asarray(
                list(zip(*mini_batches))[1], dtype=float)
            y_train = np.asarray(list(zip(*mini_batches))[2], dtype=float)

            self.model_chance_space_availible.fit(
                [X_train_state, X_train_shape], y_train, epochs=1, verbose=0)

    def update_chance_space_availible_policy_nn(self):
        # Check if the accuracy-score >=0.7
        if len(self.space_availible_shapes_true) >= self.batch_size and len(self.space_availible_shapes_false) >= self.batch_size:
            samples_true = random.sample(
                self.space_availible_shapes_true, self.batch_size)
            samples_false = random.sample(
                self.space_availible_shapes_false, self.batch_size)
            mini_batches = samples_true + samples_false
            X_test_state = np.asarray(list(zip(*mini_batches))[0], dtype=float)
            X_test_shape = np.asarray(list(zip(*mini_batches))[1], dtype=float)
            y_test = np.asarray(list(zip(*mini_batches))[2], dtype=float)

            y_pred = self.model_chance_space_availible.predict(
                [X_test_state, X_test_shape])
            y_pred_1 = [0 if pre < 0.5 else 1 for pre in y_pred]
            ac_score = accuracy_score(y_test, y_pred_1)
            self.policy_nn_can_update = True if ac_score > self.best_accuracy_updating_policy_nn else False
            if self.policy_nn_can_update:
                self.policy_nn_can_train = True
                self.train_chance_space_available_nn_after = 1000
                self.batch_size = 128
                updated_weights = ["input_state_", "input_shape_",
                                   "add_", "dropout_", "conv2d_", "flatten_", "dense_"]
                layer_dict = {
                    layer.name: layer for layer in self.model_chance_space_availible.layers}
                model_layer_names = [layer.name for layer in self.model.layers]

                for uw in updated_weights:
                    for i in range(36):
                        index_input = model_layer_names.index(f"{uw}{i}")
                        weights = layer_dict[f"{uw}-1"].get_weights()
                        self.model.layers[index_input].set_weights(weights)

    def change_state_shape_queue(self, shape_queue):
        shape_queue = [np.array(shape, dtype=np.float)
                       for shape in shape_queue]

        shape_queue = [np.resize(shape, (5, 5, 1)) for shape in shape_queue]

        shape_queue_0 = shape_queue[0]
        shape_queue_1 = shape_queue[1]
        shape_queue_2 = shape_queue[2]

        shape_queue_0 /= 10
        shape_queue_1 /= 10
        shape_queue_2 /= 10
        return shape_queue_0, shape_queue_1, shape_queue_2

    def convert_state_to_5_5(self, state):
        state[state > 0] = 1
        new_states: list = []
        for i_row, row in enumerate(state):
            for i_col, col in enumerate(row):
                state_5_5 = state[i_row:i_row+5, i_col:i_col+5]
                standard_state = np.full((5, 5), -2)
                standard_state[0:state_5_5.shape[0],
                               0:state_5_5.shape[1]] = state_5_5
                standard_state = np.array(standard_state, dtype=np.float)
                standard_state /= 10
                standard_state = np.resize(standard_state, (5, 5, 1))
                new_states.append(standard_state)
        return new_states

    def reward_function(self, reward, full_lines, negative_rewards, done):
        done_old = done
        done: bool = False
        if reward <= 0:
            negative_rewards += 1
            reward = 0
            if negative_rewards >= 10:
                done = True
        elif reward > 0 and negative_rewards > 0:
            negative_rewards = 0
            reward = 2
        else:
            reward = 2
        reward += full_lines * 10
        done = done_old if done_old else done
        return reward, negative_rewards, done

    def train(self, episodes):
        self.stats: dict = {"total_rewards": [], "chosen_action": [
        ], "nr_full_lines": [], "nr_actions_true": [], "nr_actions_false": []}
        nr_iterations_update_weights_left = 0
        printed_can_train = False
        e_give_answer: int = 20000
        for episode in range(episodes):
            start_time = time.time()
            uid: str = self.env.reset()
            state = self.env.game_env
            shapes_queue = self.env.shapes_queue
            shapes_queue = [self.create_standard_shape(
                shape) for shape in shapes_queue]
            shapes_queue_0, shapes_queue_1, shapes_queue_2 = self.change_state_shape_queue(
                shapes_queue)
            shapes_queue = [shapes_queue_0, shapes_queue_1, shapes_queue_2]
            states_5_5 = self.convert_state_to_5_5(state)
            self.env.render()
            done: bool = False
            total_reward_episode: float = 0
            negative_rewards: int = 0
            total_actions = []
            nr_full_lines = 0
            nr_actions_true: int = 0
            nr_actions_false: int = 0
            reward = 1
            negative_rewards = 0
            e_give_answer_episode: int = 0
            if not printed_can_train and self.policy_nn_can_train:
                print("Can train policy-nn")
                printed_can_train = True
            while not done:
                action, prob = self.compute_action(
                    states_5_5, shapes_queue_0, shapes_queue_1, shapes_queue_2)
                action_shape, action_row, action_col, action_r_c = self.number_to_arc(
                    action)
                action = action_r_c
                if e_give_answer>=0 and 9<=negative_rewards<=10 and printed_can_train and e_give_answer_episode<=75:
                    e_give_answer -= 1
                    e_give_answer_episode += 1
                    _, action_shape, action_row, action_col = self.env.check_if_user_can_place_the_shapes(True)
                    action = (action_row * 6) + action_col
                reward, full_lines, state_new, done, shapes_queue_new, uid = self.env.step(
                    self.env.shapes_queue[action_shape], action_row, action_col)
                shapes_queue_new = [self.create_standard_shape(
                    shape) for shape in shapes_queue_new]

                shapes_queue_0_new, shapes_queue_1_new, shapes_queue_2_new = self.change_state_shape_queue(
                    shapes_queue_new)
                states_5_5_new = self.convert_state_to_5_5(state_new)
                # Stop the episode if the agent makes 10 mistakes
                reward, negative_rewards, done = self.reward_function(
                    reward, full_lines, negative_rewards, done)
                nr_full_lines += full_lines
                # Add to memory for the policynetwork
                self.remember(
                    states_5_5, shapes_queue[action_shape], uid, action, prob, reward, action_shape)
                # Add to memory for the shape-nn
                sa = True if reward > 0 else False
                sa_nr = 1 if reward > 0 else 0
                self.add_to_memory(
                    sa, states_5_5[action], shapes_queue[action_shape], sa_nr)
                self.train_chance_space_available_nn(
                    nr_iterations_update_weights_left)
                # Updating the nn with the weights of the chance_space_available_nn
                if nr_iterations_update_weights_left % self.nr_iterations_update_weights == 0:
                    self.update_chance_space_availible_policy_nn()
                nr_iterations_update_weights_left += 1

                # Replace the old state and shapes with the new
                states_5_5 = states_5_5_new

                shapes_queue_0, shapes_queue_1, shapes_queue_2 = shapes_queue_0_new, shapes_queue_1_new, shapes_queue_2_new
                shapes_queue = [shapes_queue_0, shapes_queue_1, shapes_queue_2]
                total_reward_episode += reward
                total_actions.append(action)
                if sa:
                    nr_actions_true += 1
                else:
                    nr_actions_false += 1
                self.env.render()
            history = self.train_policy_network(
                nr_iterations_update_weights_left)
            self.stats["total_rewards"].append(total_reward_episode)
            self.stats["chosen_action"].append(total_actions)
            self.stats["nr_full_lines"].append(nr_full_lines)
            self.stats["nr_actions_true"].append(nr_actions_true)
            self.stats["nr_actions_false"].append(nr_actions_false)
            if episode % 200 == 0:
                #n_n = 0 if n_n <= 0 else n_n-1
                self.save_model()
                with open(f'{BASE_DIR}/history_data/REINFORCE/game_history_data_{self.agent_name}.pkl', 'wb') as f:
                    pickle.dump(self.stats, f)
                print(
                    f"Episode: {episode} - total reward: {total_reward_episode} | Duration: {time.time()-start_time} | e_give_answer_left: {e_give_answer}")

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_shape)
        action_encoded[action] = 1

        return action_encoded

    def number_to_arc(self, number: int):
        a = 0 if number<=35 else 1 if number<=71 else 2
        number = number if a==0 else number - 36 if a==1 else number - 72
        r = math.floor(number/6)
        c = number if r == 0 else number - (6 * r)
        return a, r, c, number


N_EPISODES = 1000000

agent = REINFORCE(port=8081)

agent.train(N_EPISODES)
