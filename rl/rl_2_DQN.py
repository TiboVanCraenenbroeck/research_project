from game.logic.game import Game
import copy
import random
import collections
import pickle
from pathlib import Path
from datetime import datetime
import eel
import time
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras import initializers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add
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


class DQNAgent:
    def __init__(self, replay_memory_size: int, batch_size: int, actions: list, retrain_target_nn_nr_episodes: int, exploration_rate: float, min_exploration_rate: float, discount: float, decay: float):
        self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        #self.agent_name = f"v_01_01_21__09_4052"
        self.batch_size = batch_size
        self.decay: float = decay
        self.discount: float = discount
        self.replay_memory = collections.deque(maxlen=replay_memory_size)

        self.possible_actions: list = actions

        self.retrain_target_nn_nr_episodes: int = retrain_target_nn_nr_episodes
        self.retrain_target_nn_counter: int = 0

        self.policy_nn_model = self.build_nn()

        self.target_nn_model = self.build_nn()
        self.target_nn_update_weights()

        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate

        self.exploration: bool = True
        self.episode_nr: int = 0
        self.exploration_number: float = 1.0

        self.total_max_reward: list = []
        self.nr_episodes: int = 1000000

        self.base_path: str = f"C:/Users/Tibovc/Desktop/MCT/Research Project/research_project/start_models/DQN/{self.agent_name}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def build_nn(self):
        input_game_grid = Input(shape=(10, 10, 1))
        input_shape_queue_0 = Input(shape=(10, 10, 1))
        input_shape_queue_1 = Input(shape=(10, 10, 1))
        input_shape_queue_2 = Input(shape=(10, 10, 1))

        model_output = concatenate(
            [input_game_grid, input_shape_queue_0, input_shape_queue_1, input_shape_queue_2])
        model_output = Conv2D(
            5, (1, 1), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)

        model_output = Conv2D(
            5, (1, 1), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)

        """ model_output = Conv2D(8, (2, 2), kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output) """

        model_output = Flatten()(model_output)

        model_output = Dense(
            300, kernel_initializer=initializers.he_normal())(model_output)
        model_output = LeakyReLU()(model_output)

        model_output = Dense(len(self.possible_actions),
                             activation="linear")(model_output)
        model_output = Model(inputs=[input_game_grid, input_shape_queue_0, input_shape_queue_1,
                                     input_shape_queue_2], outputs=model_output, name="Model_Output")

        model_output.compile(optimizer=Adam(lr=0.001),
                             loss="mse", metrics=["MeanSquaredError"])
        return model_output

    def train(self):
        self.policy_nn_train()
        # Check for retraining the target nn
        if self.retrain_target_nn_counter >= self.retrain_target_nn_nr_episodes:
            self.target_nn_update_weights()
            self.retrain_target_nn_counter = 0
        self.retrain_target_nn_counter += 1
        self.change_exploration_rate()

    def policy_nn_train(self):
        # Train only when there are enough samples in the replay-memory
        if len(self.replay_memory) >= self.batch_size:
            mini_batches = self.sample_from_replay_memory()
            mini_batches_states = np.asarray(
                list(zip(*mini_batches))[0], dtype=float)
            mini_batches_shapes_0 = np.asarray(
                list(zip(*mini_batches))[1], dtype=float)
            mini_batches_shapes_1 = np.asarray(
                list(zip(*mini_batches))[2], dtype=float)
            mini_batches_shapes_2 = np.asarray(
                list(zip(*mini_batches))[3], dtype=float)
            mini_batches_next_states = np.asarray(
                list(zip(*mini_batches))[6], dtype=float)
            mini_batches_next_shapes_0 = np.asarray(
                list(zip(*mini_batches))[7], dtype=float)
            mini_batches_next_shapes_1 = np.asarray(
                list(zip(*mini_batches))[8], dtype=float)
            mini_batches_next_shapes_2 = np.asarray(
                list(zip(*mini_batches))[9], dtype=float)
            mini_batches_actions = np.asarray(
                list(zip(*mini_batches))[4], dtype=int)
            mini_batches_rewards = np.asarray(
                list(zip(*mini_batches))[5], dtype=float)
            mini_batches_done = np.asarray(
                list(zip(*mini_batches))[10], dtype=bool)

            q_values_current_state = self.policy_nn_predict(
                mini_batches_states, mini_batches_shapes_0, mini_batches_shapes_1, mini_batches_shapes_2)
            y_train = q_values_current_state

            q_values_next_state = self.target_nn_predict(
                mini_batches_next_states, mini_batches_next_shapes_0, mini_batches_next_shapes_1, mini_batches_next_shapes_2)

            q_value_max_next_state = np.max(q_values_next_state, axis=1)

            for i, done in enumerate(mini_batches_done):
                if done:
                    y_train[i, mini_batches_actions[i]
                            ] = mini_batches_rewards[i]
                else:
                    y_train[i, mini_batches_actions[i]] = mini_batches_rewards[i] + \
                        self.discount * q_value_max_next_state[i]

            self.policy_nn_model.fit([mini_batches_states, mini_batches_shapes_0, mini_batches_shapes_1, mini_batches_shapes_2],
                                     y_train, batch_size=self.batch_size, verbose=0, use_multiprocessing=True, epochs=1)

    def policy_nn_predict(self, state, shape_0, shape_1, shape_2):
        self.state = state
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.q_policy = self.policy_nn_model.predict(
            [state, shape_0, shape_1, shape_2])
        return self.q_policy

    def target_nn_update_weights(self):
        self.target_nn_model.set_weights(self.target_nn_model.get_weights())

    def target_nn_predict(self, state, shape_0, shape_1, shape_2):
        self.state = state
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.q_target = self.target_nn_model.predict(
            [state, shape_0, shape_1, shape_2])
        return self.q_target

    def add_step_to_replay_memory(self, state, shape_0, shape_1, shape_2, action, reward, new_state, new_shape_0, new_shape_1, new_shape_2, done):
        step = (state, shape_0, shape_1, shape_2, action, reward,
                new_state, new_shape_0, new_shape_1, new_shape_2, done)
        self.replay_memory.append(step)

    def sample_from_replay_memory(self):
        # Check if the replay-memory is greater than batch size (or equal)
        if len(self.replay_memory) >= self.batch_size:
            return random.sample(self.replay_memory, self.batch_size)
        return None

    def change_exploration(self, new_reward: float):
        exploration: bool = False
        if self.exploration == False:
            if new_reward <= -1500 and self.reward <= -1500:
                self.exploration = True
                exploration = True
                self.exploration_rate = 0.5
                self.episode_nr = 0
                self.decay = 0.5/(200*1)
                self.exploration_number = 0.5
        self.reward = new_reward
        return exploration

    def change_exploration_rate(self):
        if self.exploration == True:
            self.exploration_rate = self.exploration_number - \
                (self.episode_nr * self.decay)
            self.episode_nr += 1
            if self.exploration_rate <= self.min_exploration_rate:
                self.exploration = False

        """if self.exploration_rate>self.min_exploration_rate:
            self.exploration_rate = 1 - (episode_nr * self.decay)
            #self.exploration_rate = self.exploration_rate * self.decay"""

    def compute_action(self, state, shapes_queue_0, shapes_queue_1, shapes_queue_2):
        self.state = state
        #self.shape = shape
        if np.random.uniform() <= self.exploration_rate:
            self.action = np.random.choice(self.possible_actions, 1)[0]
        else:
            state = np.array([state])
            shapes_queue_0 = np.array([shapes_queue_0])
            shapes_queue_1 = np.array([shapes_queue_1])
            shapes_queue_2 = np.array([shapes_queue_2])

            q_values = self.policy_nn_predict(
                state, shapes_queue_0, shapes_queue_1, shapes_queue_2)
            self.action = np.argmax(q_values[0])
        return self.action

    def save_agent(self, agent):

        agent_t = copy.copy(agent)
        agent_t.policy_nn_model.save(
            f'{self.base_path}/policy_model_agent_{self.agent_name}.h5')
        agent_t.target_nn_model.save(
            f'{self.base_path}/target_model_agent_{self.agent_name}.h5')

        del agent_t.policy_nn_model
        del agent_t.target_nn_model
        with open(f'{self.base_path}/agent_{self.agent_name}.pickle', 'wb') as f:
            pickle.dump(agent_t, f)

    def get_agent(self, agent):
        # Get the agent
        with open(f'{self.base_path}/agent_{self.agent_name}.pickle', 'rb') as f:
            agent = pickle.load(f)
        # Get the weights
        agent.policy_nn_model = self.build_nn()
        agent.target_nn_model = self.build_nn()
        agent.policy_nn_model.load_weights(
            f'{self.base_path}/policy_model_agent_{self.agent_name}.h5')
        agent.target_nn_model.load_weights(
            f'{self.base_path}/target_model_agent_{self.agent_name}.h5')
        return agent


def change_state_shape_queue(state, shape_queue):
    state = np.array(state, dtype=np.float)
    shape_queue = [np.array(shape, dtype=np.float) for shape in shape_queue]

    shape_queue = [np.resize(shape, (10, 10, 1)) for shape in shape_queue]
    state = np.resize(state, (10, 10, 1))

    state /= 9
    shape_queue_0 = shape_queue[0]
    shape_queue_1 = shape_queue[1]
    shape_queue_2 = shape_queue[2]

    shape_queue_0 /= 9
    shape_queue_1 /= 9
    shape_queue_2 /= 9
    return state, shape_queue_0, shape_queue_1, shape_queue_2


def create_standard_shape(shape):
    standard_shape = np.zeros((10, 10))
    for index_row, row in enumerate(shape.shape):
        for index_col, col in enumerate(row):
            standard_shape[index_row, index_col] = 0 if col == -1 else 1
    return standard_shape


def number_to_arc(number: int):
    number /= 100
    number_str = f"{number:.2f}".replace('.', '')
    number_list = list(map(int, number_str))
    return number_list[0], number_list[1], number_list[2]


env = Game(3, False, 8081)
eel.sleep(1)

replay_memory_size: int = 500000
batch_size: int = 64
discount: float = 0.95
possible_actions = [i for i in range(300)]
nr_episodes_update_target_nn: int = 5000

epsilon: float = 1.0
min_epsilon: float = 0.01

#decay: float = (0.01/0.999)**(1/(200*200))
decay: float = 1/(200000)
agent = DQNAgent(replay_memory_size, batch_size, possible_actions,
                 nr_episodes_update_target_nn, epsilon, min_epsilon, discount, decay)
#agent = agent.get_agent(agent)

stats: dict = {"total_rewards": [], "chosen_action": [], "nr_full_lines": []}
total_max_reward: list = []
#total_max_reward: list = pickle.load( open( "C:/Users/Tibovc/Desktop/MCT/Research Project/research_project/start_models/DQN/v_01_01_21__09_4052/total_max_reward.pkl", "rb" ))

for i in range(agent.nr_episodes):
    uid = env.reset()
    state = env.game_env
    shapes_queue = env.shapes_queue
    shapes_queue = [create_standard_shape(shape) for shape in shapes_queue]
    state, shapes_queue_0, shapes_queue_1, shapes_queue_2 = change_state_shape_queue(
        state, shapes_queue)

    env.render()

    done: bool = False
    total_reward_epsilon: int = 0
    total_steps: int = 0
    negative_rewards: int = 0
    total_actions = []
    nr_full_lines = 0

    start_time = time.time()

    while not done:
        total_steps += 1
        action = agent.compute_action(
            state, shapes_queue_0, shapes_queue_1, shapes_queue_2)
        action_shape, action_row, action_col = number_to_arc(action)
        total_actions.append(action)
        reward, full_lines, new_state, done, new_shape, uid = env.step(
            env.shapes_queue[action_shape], action_row, action_col)

        new_shape = [create_standard_shape(shape) for shape in new_shape]
        new_state, new_shape_0, new_shape_1, new_shape_2 = change_state_shape_queue(
            new_state, new_shape)

        if reward <= 0:
            negative_rewards += 1
            reward = -0.1
            if negative_rewards >= 3:
                done = True
        elif reward > 0 and negative_rewards > 0:
            negative_rewards = 0
            reward = 0.2
        else:
            reward = 0.2
        reward += full_lines * 10
        nr_full_lines += full_lines
        total_reward_epsilon += reward

        agent.add_step_to_replay_memory(state, shapes_queue_0, shapes_queue_1, shapes_queue_2,
                                        action, reward, new_state, new_shape_0, new_shape_1, new_shape_2, done)
        agent.train()

        state = new_state
        shapes_queue_0, shapes_queue_1, shapes_queue_2 = new_shape_0, new_shape_1, new_shape_2

        stats["total_rewards"].append(total_reward_epsilon)
        stats["chosen_action"].append(total_actions)
        stats["nr_full_lines"].append(nr_full_lines)
        env.render()
    if i % 5000 == 0:
        print("Saved!")
        with open(f'{agent.base_path}/stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        agent.save_agent(agent)
        print(f"Episode: {i} - total reward: {total_reward_epsilon} - Exploration-rate: {agent.exploration_rate} | Duration: {time.time()-start_time}")
