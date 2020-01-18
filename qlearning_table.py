import random
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.6, reward_decay=0.8, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[str(observation), :]
            if state_action['flap'] > state_action['do nothing']:
                action = 'flap'
            elif state_action['flap'] == state_action['do nothing']:
                action = np.random.choice(self.actions)
            else:
                action = 'do nothing'
        else:
            # print('random action.')
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
        # print(self.q_table.loc[str(s), a])

    def check_state_exist(self, state):
        if str(state) not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=str(state)
            ))

