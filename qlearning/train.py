import os
import json
import random
import argparse
import sys
sys.path.append("..")

from tensorboardX import SummaryWriter

from game.flappy_bird import FlappyBird
from qtable import QLearningTable

def get_args():
    parser = argparse.ArgumentParser(
            """Use q learning to play FlappyBird""")
    parser.add_argument("--num_steps", type=int, default=400000)
    parser.add_argument("--log_path", type=str, default="log")
    parser.add_argument("--saved_path", type=str, default="qtable")
    
    args = parser.parse_args()
    return args

args = get_args()
fb = FlappyBird()
qt = QLearningTable()
writer = SummaryWriter(args.log_path)

step = 0
score = 0
episode = 0
observation = fb.next_frame(random.choice([0, 1]))[-1]

while step < args.num_steps:
    qt.check_state_exist(observation)
    text = "score: {} {} : {:.3f},{:.3f}".format(score, observation, *qt.q_table[observation])
    action = qt.choose_action(observation)
    reward, terminal, score, observation_ = fb.next_frame(action, text)[1:]

    if terminal:
        observation_ = 'terminal'
        qt.learn(observation, action, reward, observation_)
        episode += 1
        # print('Episode: {}, Score: {}, States: {}'.format(episode, score, qt.q_table.__len__()))
    else:
        qt.learn(observation, action, reward, observation_)
        observation = observation_
    
    writer.add_scalar('Train/score', score, episode)

    if (step+1) % 10000 == 0:
        json.dump(qt.q_table, open('qtable/qtable_{}.json'.format(step+1), 'w'), sort_keys=True, indent=4)
    step += 1

json.dump(qt.q_table, open('qtable/q_table.json'.format(step+1), 'w'), sort_keys=True, indent=4)
