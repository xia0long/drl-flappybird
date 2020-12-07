import random
import numpy as np

class QLearningTable:
    def __init__(self, actions=[0, 1], learning_rate=0.6, reward_decay=0.8, e_greedy=1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table[observation]
            action = 0 if state_action[0] > state_action[1] else  1
        else:
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_pridict = self.q_table[s][a] # a=0 means flap, a=1 means do nothing
        if s_ != 'terminal':
            q_target = r + self.gamma * max(self.q_table[s_])
        else:
            q_target = r  # next state is terminal
        self.q_table[s][a] += self.lr * (q_target - q_pridict)

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
