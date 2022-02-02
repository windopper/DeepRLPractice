from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tqdm import tqdm
import numpy as np
import gym
import os
env = gym.make('CartPole-v1')

num_state = env.observation_space.shape[0]
num_action = env.action_space.n

model = Sequential()
model.add(Dense(32, input_dim=num_state, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_action))
model.compile(
    loss='mse',
    optimizer='adam'
)

num_iteration = 500
min_timestep_per_batch = 2500

epsilon = .3
gamma = .95
batch_size = 32

for i in tqdm(range(num_iteration)):
    timesteps_this_batch = 0
    memory = []
    while True:
        state = env.reset()
        done = False
        while not done:
            env.render()
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                q_value = model.predict(state.reshape(1, num_state))
                action = np.argmax(q_value[0])

            next_state, reward, done, info = env.step(action)

            memory.append((state, action, reward, next_state, done))
            state = next_state

        timesteps_this_batch += len(memory)

        if timesteps_this_batch > min_timestep_per_batch:
            break


    # Replay
    for state, action, reward, next_state, done in memory:
        if done:
            target = reward
        else:
            target = reward + gamma * (np.max(model.predict(next_state.reshape(1, num_state))[0]))
        q_value = model.predict(state.reshape(1, num_state))
        q_value[0][action] = target
        model.fit(state.reshape(1, num_state), q_value, epochs=1, batch_size=32, verbose=0)

env.close()

# save_dir = os.getcwd()
# model_name = 'keras_dqn_trained_model.h5'
#
# # Save model and weights
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)

