import os
import json
import random
from flappy_bird import FlappyBird
from qlearning_table import QLearningTable

fb = FlappyBird()
qt = QLearningTable()
observation = fb.next_frame(random.choice([0, 1]))[-1]

max_score = 0
step = 0
episode = 0
terminal = False
while 1:
    qt.check_state_exist(observation)
    text = "{} : {:.3f},{:.3f}".format(observation, qt.q_table[observation][0], qt.q_table[observation][1])
    action = qt.choose_action(observation)
    reward, terminal, score, observation_ = fb.next_frame(action, text)
    if score > max_score:
        max_score = score

    if terminal:
        observation_ = 'terminal'
        qt.learn(observation, action, reward, observation_)
        # observation = fb.next_frame(random.choice([0, 1]))[-1]
        episode += 1
        if (episode+1) % 10 == 0:
            print('Episode: {}, Max_score: {}, States: {}'.format(episode, max_score, qt.q_table.__len__()))
            # print(qt.q_table)
            json.dump(qt.q_table, open('qtable.json', 'w'), sort_keys=True, indent=4)

    else:
        qt.learn(observation, action, reward, observation_)
        observation = observation_
    
    if terminal:
        observation = fb.next_frame(random.choice([0, 1]))[-1]
    step += 1
