
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


import tensorflow as tf

if tf.__version__.startswith("1."):    
    raise RuntimeError("Error!! You are using tensorflow-v1")

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, Input

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import gym

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=2, \
                layer1_size=1024, layer2_size=512, input_dims=4):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = self.build_network()

        self.action_space = [i for i in range(self.n_actions)]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

    def build_network(self):

        input = Input(shape=(self.input_dims, ))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation="relu")(input)
        dense2 = Dense(self.fc2_dims, activation="relu")(dense1)
        probs = Dense(self.n_actions, activation="softmax")(dense2)
        values = Dense(1, activation="linear")(dense2)

        actor = Model(inputs=[input, delta], outputs=[probs])

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss="mse")

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):

        state = observation[np.newaxis, :]
        probs = self.policy.predict(state)[0]

        action = np.random.choice(self.action_space, p=probs)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        self.critic.fit(state, target, verbose=0)

        with tf.GradientTape() as tape:
            y_pred = self.actor(state)
            out = K.clip(y_pred, 1e-8, 1-1e-8)  
            log_lik = actions * K.log(out)            
            myloss = K.sum(-log_lik*delta)
        grads = tape.gradient(myloss, self.actor.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

if __name__ == "__main__":
    agent=Agent(alpha=0.0001, beta=0.0005)

    ENV_SEED = 1024  ## Reproducibility of the game
    NP_SEED = 1024  ## Reproducibility of numpy random
    env = gym.make('CartPole-v0')
    env = env.unwrapped    # use unwrapped version, otherwise episodes will terminate after 200 steps
    env.seed(ENV_SEED)  
    np.random.seed(NP_SEED)


    ### The Discrete space allows a fixed range of non-negative numbers, so in this case valid actions are either 0 or 1. 
    print(env.action_space)
    ### The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers. 
    print(env.observation_space)
    ### We can also check the Box’s bounds:
    print(env.observation_space.high)
    print(env.observation_space.low)

    score_history = []
    num_episodes = 2000

    for i in range(num_episodes):
        done = False

        score = 0
        observation = env.reset()

        while not done:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info= env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score+=reward

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print("episode ", i, "score %.2f average score %.2f" % (score, avg_score))