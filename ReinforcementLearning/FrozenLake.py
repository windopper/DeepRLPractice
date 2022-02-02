import random

import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np
import random as pr

def rargmax(vector):
    m = np.amax(vector) # 벡터 요소 중 최대 요소를 구함
    indices = np.nonzero(vector == m)[0] # vector 변수와 어레이 최대값이 같은 인덱스들을 배열로 구함
    return pr.choice(indices) # 인덱스들의 리스트를 랜덤으로 뽑음

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

env = gym.make("FrozenLake-v3")
Q = np.zeros([env.observation_space.n, env.action_space.n])
dis = .99
lr = 0.85

num_episodes = 2000
state = env.reset()

# lists for containing total reward and steps per episode
rList = []

# Q-Table algorithm
for i in range(num_episodes):

    e = 1. / ((i // 100) + 1)  # decaying E-greedy

    state = env.reset()
    rAll = 0
    done = False

    while not done:

        # 두개의 방법이 존재 노이즈 방법이 이 경우에서 더 높은 성공 확률을 보임

        # 액션 값에 따른 기대 보상에 노이즈를 추가하여 다음 액션을 정함
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

        # # e-greedy 에 의하여 선택되는 action
        # if np.random.rand(1) < e:
        #     action = env.action_space.sample()
        # else:
        #     action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        # 모든 가능한 Action을 기반으로 next_state의 Q 최댓값을 구하여 현재 보상에 더함
        # decay rate를 사용하여 미래에 받게 되는 보상의 비율을 조정함

        # 이전 리워드를 반영하면서 '고집'이 세짐. 랜덤
        Q[state, action] = (1-lr) * Q[state, action] + lr * (reward + dis * np.max(Q[new_state, :]))

        #
        # Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward # 받은 보상
        state = new_state # state update

    # print(rAll)
    rList.append(rAll)

print("Success Rate : " + str(sum(rList)/num_episodes))
print(Q)

