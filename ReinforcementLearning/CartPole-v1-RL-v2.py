import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym
import tensorflow as tf

# from replaybuffer import ReplayBuffer


env = gym.make('CartPole-v1')

env.action_space

print(env.observation_space.shape[0])
print(env.action_space.sample())

class DQN(Model):
    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(action_n, activation='linear')

    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        q = self.q(x)
        return q

class DQNagent(object):
    def __init__(self, env):

        # 하이퍼 파라미터들
        self.GAMMA = 0.95 # 할인계수 : 에이전트가 현재 보상을 미래 보상보다 얼마나 더 가치있게 여기는지
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000 # 메모리 저장
        self.DQN_LEARNING_RATE = 0.001 # 학습률
        self.TAU = 0.001
        self.EPSILON = 1.0 # 학습 시작시 에이전트가 무작위로 행동할 확률
        self.EPSILON_DECAY = 0.995 # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
        self.EPSILON_MIN = 0.01 # 학습 막바지에 에이전트가 무작위로 행동할 확률
        self.env = env

        self.state_dim = env.observation_space.shape[0] # 상태 차원 가져오기
        self.action_n = env.action_space.n # 에이전트가 취할 수 있는 행동 수

        # q-network 제작하기
        self.dqn = DQN(self.action_n)
        self.target_dqn = DQN(self.action_n)

        self.dqn.build(input_shape=(None, self.state_dim))
        self.target_dqn.build(input_shape=(None, self.state_dim))

        self.dqn.summary()

        # 최적화 함수
        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        # 리플레이 버퍼 초기화
        # self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 결과 저장
        self.save_epi_reward = []

        # 액션 취하기
        def choose_action(self, state):
            if np.random.random() <= self.EPSILON:
                return self.env.action_space.sample()
            else:
                qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
                return np.argmax(qs.numpy())

        def update_target_network(self, TAU):
            phi = self.dqn.get_weights()
            target_phi = self.target_dqn.get_weights()


