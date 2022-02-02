
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np
import random as pr
import tensorflow as tf

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': True}
)

env = gym.make("FrozenLake-v3")

input_size = env.observation_space.n
output_size = env.action_space.n

X = tf.Tensor(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))
Qpred = tf.matmul(X, W)

Y = tf.Tensor(shape=[1, output_size], dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=0.7).minimize(loss)
