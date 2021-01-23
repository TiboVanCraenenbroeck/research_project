import random
import tensorflow as tf
try
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import math
import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, concatenate, add, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import initializers
import time
import eel
from datetime import datetime
from pathlib import Path
import pickle
import collections


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
from game.logic.game import Game

class Node:
    def __init__(self, prior):
        self.hidden_state = None
        self.prior = prior
        self.value_sum = 0
        self.visit_count = 0
        self.reward = 0
        self.children = {} # {action1: node1, action2: node2, ...}
        self.to_play = -1

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
class ReplayBuffer:
    def __init__(self, window_size, batch_size, num_unroll_steps):
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, bs=None):
        games = [self.sample_game() for _ in range(self.batch_size if bs is None else bs)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        def xtend(g,x,s):
            # pick the last (fake) action
            while len(x) < s:
                x.append(-1)
            return x
        test = [(g.make_image(i), xtend(g,g.history[i:i + self.num_unroll_steps], self.num_unroll_steps),
                    g.make_target(i, self.num_unroll_steps))
                    for (g, i) in game_pos]
        return test

    def sample_game(self):
        return random.choice(self.buffer)


    def sample_position(self, game):
        # have to do -num_unroll_steps to allow enough actions
        return random.randint(0, len(game.history)-1)

class GamePlay:
    def __init__(self, env: Game, discount=0.95):
        self.env = env
        _ = env.reset()
        self.observations = []
        self.history = []
        self.rewards = []
        self.policies = []
        self.actions_true = 0
        self.actions_false = 0
        self.discount = discount
        self.done = False
        self.total_reward = 0
        self.negative_rewards = 0
        self.full_lines = 0
        self.start()
    
    def start(self):
        state = self.env.game_env
        state[state > 0] = 1
        state[state <= 0] = 0
        state = np.resize(state, (4, 4, 1))
        shapes_queue = self.env.shapes_queue[0]
        shapes_queue = self.create_standard_shape(shapes_queue)
        observation = [state, shapes_queue]
        self.observation = observation

    def terminal(self):
        return self.done

    def create_standard_shape(self, shape):
        standard_shape = np.zeros((2, 2))
        for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col > 0 else 0
        standard_shape = np.array(standard_shape, dtype=np.float)
        standard_shape = np.resize(standard_shape, (2, 2, 1))
        standard_shape /= 10
        return standard_shape

    def number_to_arc(self, number: int):
        a = 0 if number <= 15 else 1 if number <= 31 else 2
        #number = number if a==0 else number - 36 if a==1 else number - 72
        r = math.floor(number/4)
        c = number if r == 0 else number - (4 * r)
        return a, r, c
    
    def reward_function(self, reward, full_lines, done):
        done_old = done
        done: bool = False
        if reward <= 0:
            self.negative_rewards += 1
            reward = -1
            if self.negative_rewards >= 3:
                done = True
        elif reward > 0 and self.negative_rewards > 0:
            self.negative_rewards = 0
            reward = 3
        else:
            reward = 3
        reward += full_lines * 2
        done = done_old if done_old else done
        return reward, done

    def apply(self, a_1, p=None):
        self.observations.append(np.copy(self.observation))
        action_shape, action_row, action_col = self.number_to_arc(a_1)
        r_1, full_lines, state_new, done, shapes_queue_new, uid = self.env.step(self.env.shapes_queue[action_shape], action_row, action_col)
        shapes_queue_new = self.create_standard_shape(shapes_queue_new[0])
        state_new[state_new > 0] = 1
        state_new[state_new <= 0] = 0
        state_new = np.resize(state_new, (4, 4, 1))

        observation = [state_new, shapes_queue_new]
        self.observation = observation
        r_1, done = self.reward_function(r_1, full_lines, done)
        self.history.append(a_1)
        self.rewards.append(r_1)
        self.total_reward += r_1
        self.policies.append(p)
        self.full_lines += 1
        if self.negative_rewards==0:
            self.actions_true += 1
        else:
            self.actions_false += 1

        self.done = done

    def act_with_policy(self, policy):
        act = np.random.choice(list(range(len(policy))), p=policy)
        self.apply(act, policy)

    def make_image(self, i):
        return self.observations[i]

    def make_target(self, state_index, num_unroll_steps):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            value = 0
            for i, reward in enumerate(self.rewards[current_index:]):
                value += reward * self.discount**i
            
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < len(self.policies):
                targets.append((value, last_reward, self.policies[current_index]))
            else:
                # no policy, what does cross entropy do? hopefully not learn
                targets.append((0, last_reward, np.array([0]*len(self.policies[0]))))
        return targets

class MCTS:
    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.95
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

    def ucb_score(self, parent: Node, child: Node, min_max_stats=None) -> float:
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            if min_max_stats is not None:
                value_score = child.reward + self.discount * min_max_stats.normalize(child.value())
            else:
                value_score = child.reward + self.discount * child.value()
        else:
            value_score = 0

        #print(prior_score, value_score)
        return prior_score + value_score

    def select_child(self, node: Node, min_max_stats=None):
        out = [(self.ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items()]
        smax = max([x[0] for x in out])
        # this max is why it favors 1's over 0's
        _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
        return action, child
    
    def mcts_search(self, m, observation, num_simulations=10, minimax=True):
        # init the root node
        root = Node(0)
        root.hidden_state = m.ht(observation)
        """ if minimax:
            root.to_play = observation[0][-1] """
        hidden_state_np = np.array([root.hidden_state])
        policy, value = m.ft(hidden_state_np)

        # expand the children of the root node
        for i in range(policy.shape[0]):
            root.children[i] = Node(policy[i])
            root.children[i].to_play = -root.to_play

        # add exploration noise at the root
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(actions))
        frac = self.root_exploration_fraction
        for a, n in zip(actions, noise):
            root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac

        # run_mcts
        #min_max_stats = MinMaxStats()
        for _ in range(num_simulations):
            history = []
            node = root
            search_path = [node]

            # traverse down the tree according to the ucb_score 
            while node.expanded():
                #action, node = select_child(node, min_max_stats)
                action, node = self.select_child(node)
                history.append(action)
                search_path.append(node)

            # now we are at a leaf which is not "expanded", run the dynamics model
            parent = search_path[-2]
            node.hidden_state, node.reward = m.gt(parent.hidden_state, history[-1])

            # use the model to estimate the policy and value, use policy as prior
            hidden_state_np = np.array([node.hidden_state])
            policy, value = m.ft(hidden_state_np)
            #print(history, value)

            # create all the children of the newly expanded node
            for i in range(policy.shape[0]):
                node.children[i] = Node(prior=policy[i])
                node.children[i].to_play = -node.to_play

            # update the state with "backpropagate"
            for bnode in reversed(search_path):
                if minimax:
                    bnode.value_sum += value if root.to_play == bnode.to_play else -value
                else:
                    bnode.value_sum += value
                bnode.visit_count += 1
                #min_max_stats.update(node.value())
                value = bnode.reward + self.discount * value

        # output the final policy
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        visit_counts = [x[1] for x in sorted(visit_counts)]
        av = np.array(visit_counts).astype(np.float64)
        policy = self.softmax(av)
        return policy, root

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

class MuZero:
    def __init__(self, learning_rate: float, replay_memory_size: int, batch_size:int, K: int, port: int = 8080):
        self.learning_rate = learning_rate
        self.a_dim = 16
        self.with_policy = True

        self.start(replay_memory_size, batch_size, K)

        self.nn_hidden_state, self.nn_prediction, self.nn_dynamic = self.build_nn()
        self.make_model()
        self.env = Game(1, False, port, 4)
        self.mcts = MCTS()
    
    def save_model(self):
        self.nn_hidden_state.save(f"{self.base_path}/h.h5")
        self.nn_prediction.save(f"{self.base_path}/f.h5")
        self.nn_dynamic.save(f"{self.base_path}/g.h5")
        self.model.save(f"{self.base_path}/base_model.h5")

        history = {"tot_reward": self.tot_reward, "game_history": self.game_history, "len_actions_taken": self.len_actions_taken, "actions_true": self.actions_true, "actions_false": self.actions_false, "full_lines": self.full_lines}
        with open(f'{self.base_path}/game_history.pkl', 'wb') as f:
            pickle.dump(history, f)

    def start(self, replay_memory_size, batch_size, K):
        self.agent_name = f"v_{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        self.base_path = f"{BASE_DIR}/start_models/muzero/{self.agent_name}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        # Vars
        self.dim_hidden_state = 20
        self.dim_prediction_function_policy: int = 16
        self.dim_dynamic_function_current_action: int = 16

        self.number_layers_prediction_function: int = 3
        self.number_layers_dynamic_function: int = 3

        self.number_neurons_prediction_function: int = 40
        self.number_neurons_dynamic_function: int = 72

        self.batch_size = batch_size

        self.K = K

        self.replay_memory = collections.deque(maxlen=replay_memory_size)
        self.losses = []
    
    def build_nn(self):
        # Input
        input_game_grid = Input(shape=(4, 4, 1,))
        input_shape = Input(shape=(2, 2, 1))

        # Hidden state
        #model_game_grid = Conv2D(8, (3, 3), activation="relu")(input_game_grid)
        model_game_grid = Flatten()(input_game_grid)
        """ model_game_grid = Dense(32, activation="relu")(model_game_grid)
        model_game_grid = Dense(32, activation="relu")(model_game_grid) """
        model_game_grid = Model(inputs=[input_game_grid], outputs=[model_game_grid])

        #model_shape = Conv2D(8, (2, 2), activation="relu")(input_shape)
        model_shape = Flatten()(input_shape)
        """ model_shape = Dense(8, activation="relu")(model_shape)
        model_shape = Dense(8, activation="relu")(model_shape) """
        model_shape = Model(inputs=[input_shape], outputs=[model_shape])

        model_hidden_state = concatenate([model_game_grid.output, model_shape.output])

        model_hidden_state = Dense(self.dim_hidden_state)(model_hidden_state)
        model_hidden_state = Model(inputs=[input_game_grid, input_shape], outputs=[model_hidden_state])

        # Policy/value model --> Prediction function
        input_s0 = Input(shape=(self.dim_hidden_state))

        model_prediction = input_s0

        for i in range(self.number_layers_prediction_function):
            model_prediction = Dense(self.number_neurons_prediction_function, activation="relu")(model_prediction)
            """ if i != self.number_layers_prediction_function-1:
                model_prediction = BatchNormalization()(model_prediction) """

        model_prediction_policy = Dense(self.dim_prediction_function_policy, activation="softmax")(model_prediction)
        model_prediction_value = Dense(1, activation="linear")(model_prediction)

        model_prediction = Model(inputs=[input_s0], outputs=[model_prediction_policy, model_prediction_value])

        # Dynamic function
        input_current_action = Input(shape=(self.dim_dynamic_function_current_action))

        model_dynamic = concatenate([input_s0, input_current_action])

        for i in range(self.number_layers_dynamic_function):
            model_dynamic = Dense(self.number_neurons_dynamic_function, activation="relu")(model_dynamic)
            """ if i != self.number_layers_dynamic_function:
                model_dynamic = BatchNormalization()(model_dynamic) """

        model_dynamic_s1 = Dense(self.dim_hidden_state)(model_dynamic)
        model_dynamic_reward = Dense(1, activation="linear")(model_dynamic)

        model_dynamic = Model(inputs=[input_s0, input_current_action], outputs=[model_dynamic_s1, model_dynamic_reward])

        return model_hidden_state, model_prediction, model_dynamic

    def make_model(self):
        inputs, outputs, loss = [], [], []

        def softmax_ce_logits(y_true, y_pred):
            return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        input_game_grid = Input(shape=(4, 4))
        input_shape = Input(shape=(2, 2))
        
        model_hidden_state = self.nn_hidden_state([input_game_grid, input_shape])

        model_policy, model_value = self.nn_prediction([model_hidden_state])
        outputs += [model_policy, model_value]
        loss += ["mse", softmax_ce_logits]

        for k in range(self.K):
            input_current_action = Input(shape=(self.dim_dynamic_function_current_action), name=f"l_{k}")
            inputs.append(input_current_action)
            model_state_1, model_reward = self.nn_dynamic([model_hidden_state, input_current_action])

            policy_s1, value_s1 = self.nn_prediction([model_state_1])
            outputs += [model_reward, policy_s1, value_s1]
            loss += ["mse", softmax_ce_logits, "mse"]

            model_hidden_state = model_state_1

        self.model = Model(inputs=[input_game_grid, input_shape] + inputs, outputs=outputs)
        self.model.compile(loss=loss, optimizer=Adam(self.learning_rate))
    
    def ht(self, o_0):
        o_0_state = np.array([o_0[0]])
        o_0_shape = np.array([o_0[1]])
        return self.nn_hidden_state.predict([o_0_state, o_0_shape])[0]
        #return self.nn_hidden_state.predict(np.array(o_0)[None])[0]

    def ft(self, s_k):
        if self.with_policy:
            p_k, v_k = self.nn_prediction.predict(s_k)
            return np.exp(p_k[0]), v_k[0][0]
        else:
            v_k = self.nn_prediction.predict(s_k[None])
            return np.array([1/self.a_dim]*self.a_dim), v_k[0][0]

    def to_one_hot(self, x,n):
        ret = np.zeros([n])
        if x >= 0:
            ret[x] = 1.0
        return ret

    def gt(self, s_km1, a_k):
        s_k, r_k = self.nn_dynamic.predict([s_km1[None], self.to_one_hot(a_k, self.a_dim)[None]])
        return s_k[0], r_k[0][0]
    
    def play_game(self, env, m):
        game = GamePlay(env, discount=0.997)
        while not game.terminal():
            cc = random.random()
            if cc < 0.05:
                policy = [1/m.a_dim]*m.a_dim
            else:
                policy, _ = self.mcts.mcts_search(m, game.observation, 50)
            game.act_with_policy(policy)
            self.env.render()
        return game

    def bstack(self, bb):
        ret = [[x] for x in bb[0]]
        for i in range(1, len(bb)):
            for j in range(len(bb[i])):
                ret[j].append(bb[i][j])
        return [np.array(x) for x in ret]

    def reformat_batch(self, batch, a_dim, remove_policy=False):
        X,Y = [], []
        for o,a,outs in batch:
            x = [o[0]] + [o[1]] + [self.to_one_hot(x, a_dim) for x in a]
            y = []
            for ll in [list(x) for x in outs]:
                y += ll
            X.append(x)
            Y.append(y)
        X = self.bstack(X)
        Y = self.bstack(Y)
        if remove_policy:
            nY = [Y[0]]
            for i in range(3, len(Y), 3):
                nY.append(Y[i])
                nY.append(Y[i+1])
            Y = nY
        else:
            Y = [Y[0]] + Y[2:]
        return X,Y
    
    def train_on_batch(self, batch):
        X,Y = self.reformat_batch(batch, self.a_dim, not self.with_policy)
        X[0] = np.reshape(X[0],(len(X[0]), 4, 4))
        X[1] = np.reshape(X[1],(len(X[0]), 2, 2))

        for i in range((self.K+1)):
            Y[i*3], Y[i*3+1] = Y[i*3+1], Y[i*3]

        l = self.model.train_on_batch(X,Y)
        self.losses.append(l)
        return l

    def train_v1(self, window_size, train_episodes, nr_train_on_batch_samples):
        self.replay_buffer = ReplayBuffer(window_size, self.batch_size, self.K)
        self.tot_reward = []
        self.game_history = []
        self.len_actions_taken = []
        self.actions_true = []
        self.actions_false = []
        self.full_lines = []

        for train_episode in range(train_episodes):
            game = self.play_game(self.env, self)
            self.replay_buffer.save_game(game)
            for i in range(nr_train_on_batch_samples):
                self.train_on_batch(self.replay_buffer.sample_batch())
            reward = sum(game.rewards)
            self.tot_reward.append(reward)
            game_history = collections.Counter(game.history)
            self.game_history.append(game_history)
            len_actions_taken = len(game.history)
            self.len_actions_taken.append(len_actions_taken)
            self.actions_true.append(game.actions_true)
            self.actions_false.append(game.actions_false)
            self.full_lines.append(game.full_lines)
            print(train_episode, len_actions_taken, reward, game_history, self.losses[-1][0])

            if train_episode%250 == 0:
                self.save_model()
    

    def create_standard_shape(self, shape):
        standard_shape = np.zeros((2, 2))
        for index_row, row in enumerate(shape.shape):
            for index_col, col in enumerate(row):
                standard_shape[index_row, index_col] = 1 if col > 0 else 0
        standard_shape = np.array(standard_shape, dtype=np.float)
        standard_shape = np.resize(standard_shape, (2, 2, 1))
        standard_shape /= 10
        return standard_shape

    def number_to_arc(self, number: int):
        a = 0 if number <= 15 else 1 if number <= 31 else 2
        #number = number if a==0 else number - 36 if a==1 else number - 72
        r = math.floor(number/4)
        c = number if r == 0 else number - (4 * r)
        return a, r, c

    def play(self):
        _ = self.env.reset()

        state = self.env.game_env
        state[state > 0] = 1
        state[state <= 0] = 0
        state = np.resize(state, (4, 4, 1))
        shapes_queue = self.env.shapes_queue[0]
        shapes_queue = self.create_standard_shape(shapes_queue)
        observation = [state, shapes_queue]
        done = False
        while not done:
            p_0, _ = self.mcts.mcts_search(self, observation, 50)
            a_1 = np.random.choice(list(range(len(p_0))), p=p_0)
            hidden_state = self.ht(observation)
            hidden_state_np = np.array([hidden_state])
            _, v_0 = self.ft(hidden_state_np)

            action_shape, action_row, action_col = self.number_to_arc(a_1)
            r, full_lines, state_new, done, shapes_queue_new, uid = self.env.step(self.env.shapes_queue[action_shape], action_row, action_col)
            shapes_queue_new = self.create_standard_shape(shapes_queue_new[0])
            state_new[state_new > 0] = 1
            state_new[state_new <= 0] = 0
            state_new = np.resize(state_new, (4, 4, 1))
            observation = [state_new, shapes_queue_new]
            
            self.env.render()
            print(a_1, v_0, r)
        
        print("Done!")



learning_rate = 0.0001
replay_memory_size = 500
batch_size = 128
test = MuZero(learning_rate, replay_memory_size, batch_size, 3, 8083)

window_size: int = 200
train_episodes: int = 10000
train_from_batch: int = 21
test.train_v1(window_size, train_episodes, train_from_batch)
test.play()