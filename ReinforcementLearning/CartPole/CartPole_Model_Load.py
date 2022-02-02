from tensorflow.keras.models import load_model
import os
import gym
import numpy as np

load_dir = os.getcwd()
model_name = 'keras_dqn_trained_model.h5'
model_path = os.path.join(load_dir, model_name)
model = load_model(model_path)

env = gym.make('CartPole-v1')
num_state = env.observation_space.shape[0]

for i in range(1000):
    state = env.reset()
    done = False

    while not done:
        env.render()
        state = np.array(state).reshape(1, num_state)
        q_value = model.predict(state)
        action = np.argmax(q_value)
        state, reward, done, info = env.step(action)
