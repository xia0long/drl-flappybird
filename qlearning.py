import random
import numpy as np
import pandas as pd

from flappy_bird import FlappyBird

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[str(observation), :]
            # print(state_action)
            if state_action['flap'] > state_action['do nothing']:
                action = 'flap'
            else:
                action = 'do nothing'
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[str(s), a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[str(s_), :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if str(state) not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=str(state)
                )
            )
        # print(len(self.q_table.index))

fb = FlappyBird()
qt = QLearningTable(['flap', 'do nothing'])
observation = fb.next_frame(random.randint(0, 1))[-1]

max_score = 0
for i in range(2000000):
    action = qt.choose_action(list(observation))
    reward, terminal, score, observation_ = fb.next_frame(action)[1:]
    if terminal:
        observation_ = 'terminal'

    qt.learn(observation, action, reward, observation_)

    observation = observation_
    
    if terminal:
        max_score = max(max_score, score)
        
    if i % 1000 == 0:
        print('Epodie: {}, Score: {}, States: {}'.format(i, max_score, qt.q_table.index.__len__()))