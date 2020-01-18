import os
import random
from flappy_bird import FlappyBird
from qlearning_table import QLearningTable

fb = FlappyBird()
qt = QLearningTable(['flap', 'do nothing'])
observation = fb.next_frame(random.randint(0, 1))[-1]

max_score = 0
episode = 1
while 1:
    if episode % 4 != 0:
        # print(episode)
        fb.next_frame(0)
        episode += 1
        continue
    action = qt.choose_action(list(observation))
    reward, terminal, score, observation_ = fb.next_frame(action)[1:]
    if terminal:
        observation_ = 'terminal'
        max_score = max(max_score, score)

    qt.learn(observation, action, reward, observation_)
    observation = observation_
        
    if episode % 1000 == 0:
        print('Episode: {}, Score: {}, States: {}'.format(episode, max_score, qt.q_table.index.__len__()))
        print(qt.q_table)

    if episode % 500000 == 0:
        file_path = 'trained_tables/{}.pkl'.format(episode)
        if not os.path.exists(file_path):
            os.mknod(file_path)
        qt.q_table.to_pickle(file_path)

    episode += 1