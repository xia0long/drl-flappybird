import argparse

import pandas as pd

from qlearning_table import QLearningTable

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Q-Learning to play Flappy Bird""")
    parser.add_argument("--qtable", type=str, default=84)
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

args = get_args()
fb = FlappyBird()
qt = QLearningTable(['flap', 'do nothing'])
qt.q_table = pd.read_pickle(args.qtable)
observation = fb.next_frame(random.randint(0, 1))[-1]

while 1:
    action = qt.choose_action(list(observation))
    reward, terminal, score, observation_ = fb.next_frame(action)[1:]
    