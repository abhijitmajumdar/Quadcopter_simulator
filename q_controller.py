from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

STATE_SIZE = 12
ACTION_SIZE = 4
LEARNING_RATE = 0.001

class Q_Controller:
    def __init__(self):
        self.prev_action = [0 for x in range(ACTION_SIZE)]
        self.prev_state = [0 for x in range(STATE_SIZE)]
        self.memory = deque(maxlen=2000)
        self.model = self.create_model()
        # bug in tensorflow with asynchronous events makes this predict call necessary
        self.model.predict(np.zeros(12).reshape(1,12))
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.99

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=STATE_SIZE, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(ACTION_SIZE, activation='sigmoid'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def get_motor_speeds(self, reward, data):
        self.remember(self.prev_state, self.prev_action, reward, data)
        self.prev_state = data
        if np.random.rand() <= self.epsilon:
            action = [np.random.rand() for x in range(ACTION_SIZE)]
            self.prev_action = action
            return action
        action = self.model.predict(np.array(data).reshape(1,12))[0]
        self.prev_action = action
        return action

    def replay(self):
        minibatch = random.sample(self.memory, len(self.memory) // 10)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * self.model.predict(np.array(next_state).reshape(1,12))[0]
            target_f = self.model.predict(np.array(state).reshape(1,12))
            target_f[0] = target_f[0] + target
            self.model.fit(np.array(state).reshape(1,12), np.array(target_f).reshape(1,4), epochs=10, verbose=0)
        self.epsilon *= self.epsilon_decay
