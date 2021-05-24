import json
import random
import sys
sys.path.append("..")
import argparse

from game.flappy_bird import FlappyBird
from qtable import QLearningTable

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Q-Learning to play Flappy Bird""")
    parser.add_argument("--qtable",
                        type=str,
                        default="qtable/q_table.json",
                        help="Specify the Q table.")
    
    args = parser.parse_args()
    return args

args = get_args()
fb = FlappyBird()
qlt = QLearningTable(e_greedy=1)

qlt.q_table = json.load(open(args.qtable, "r"))
observation = fb.next_frame(random.choice([0, 1]))[-1]

while 1:
    qlt.check_state_exist(observation)
    action = qlt.choose_action(observation)
    reward, terminal, score, observation_ = fb.next_frame(action, text='')[1:]
    observation = observation_

    if terminal:
        fb.__init__()
        observation = fb.next_frame(random.choice([0, 1]))[-1]
