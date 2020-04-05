import os
import json
import random
import logging
import sys
sys.path.append("..")
from game.flappy_bird import FlappyBird
from qltable import QLearningTable

def set_up_logger(log_level=logging.INFO):
    stream_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('[%(levelname)s] [train] [%(asctime)s] : %(message)s')
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(log_level)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = 0
    if not logger.handlers:
        logger.addHandler(stream_handler)
    return logger


logger = set_up_logger()
fb = FlappyBird()
qt = QLearningTable()
observation = fb.next_frame(random.choice([0, 1]))[-1]

score = 0
max_score = 0
episode = 0
while 1:
    qt.check_state_exist(observation)
    text = "score: {} {} : {:.3f},{:.3f}".format(score, observation, *qt.q_table[observation])
    action = qt.choose_action(observation)
    reward, terminal, score, observation_ = fb.next_frame(action, text)
    if score > max_score:
        max_score = score

    if terminal:
        observation_ = 'terminal'
        qt.learn(observation, action, reward, observation_)
        episode += 1
        if (episode) % 10 == 0:
            logger.info('Episode: {}, Max_score: {}, States: {}'.format(episode, max_score, qt.q_table.__len__()))
            json.dump(qt.q_table, open('qtable.json', 'w'), sort_keys=True, indent=4)
    else:
        qt.learn(observation, action, reward, observation_)
        observation = observation_
    
    if terminal:
        observation = fb.next_frame(random.choice([0, 1]))[-1]



