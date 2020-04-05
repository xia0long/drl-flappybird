import os
import json
import random
import logging
import sys
sys.path.append("..")

from game.flappy_bird import FlappyBird
from qtable import QLearningTable

def set_up_logger(filename, log_level=logging.INFO):
    logger = logging.getLogger('train')
    hdlr = logging.FileHandler(filename)
    formatter = logging.Formatter('[%(levelname)s] [train] [%(asctime)s] : %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(log_level)
    return logger

logger = set_up_logger(filename='train.log')
fb = FlappyBird()
qt = QLearningTable()

step = 0
score = 0
episode = 0
observation = fb.next_frame(random.choice([0, 1]))[-1]

while 1:
    qt.check_state_exist(observation)
    text = "score: {} {} : {:.3f},{:.3f}".format(score, observation, *qt.q_table[observation])
    action = qt.choose_action(observation)
    reward, terminal, score, observation_ = fb.next_frame(action, text)

    if terminal:
        observation_ = 'terminal'
        qt.learn(observation, action, reward, observation_)
        episode += 1
        logger.info('Episode: {}, Score: {}, States: {}'.format(episode, score, qt.q_table.__len__()))
    else:
        qt.learn(observation, action, reward, observation_)
        observation = observation_
    
    if terminal:
        fb.__init__()
        observation = fb.next_frame(random.choice([0, 1]))[-1]
    
    if (step+1) % 1000 == 0:
        json.dump(qt.q_table, open('qtable.json', 'w'), sort_keys=True, indent=4)
    
    step += 1



