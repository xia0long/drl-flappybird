#!/usr/bin/env python3
import os
import argparse
import sys
sys.path.append("..")
from random import random, randint, sample

import cv2
import numpy as np
import torch
import torch.nn as nn

from game.flappy_bird import FlappyBird
from networks import Net as Network

def get_args():
    parser = argparse.ArgumentParser("""Use Deep Q-learning to play Flappy Bird.""")
    parser.add_argument('--seed', type=int, default=456, help='seed for initializing training. ')
    parser.add_argument("--img_size", type=int, default=84, help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--replay_memory_size", type=int, default=50000, help="Number of epoches between testing phases")

    args = parser.parse_args()
    return args

def image_pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    return image[None, :, :].astype(np.float32)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    fb = FlappyBird()

    image, reward, terminal, score = fb.next_frame(randint(0, 1))[:-1]

    img = image_pre_processing(image, args.img_size, args.img_size)
    img = torch.from_numpy(img)
    img.to(device)
    state = torch.cat(tuple(img for _ in range(4)))[None, :, :, :]

    replay_memory = []
    episode = 0

    while 1:
        prediction = model(state)[0]
        epsilon = args.initial_epsilon
        if np.random.uniform() > epsilon:
            action = torch.argmax(prediction)
        else:
            action = randint(0, 1)

        next_image, reward, terminal = fb.next_frame(action, '')[:3]

        if terminal:
            fb.__init__()
            next_image, reward, terminal = fb.next_frame(action, '')[:3]

        next_img = image_pre_processing(next_image, args.img_size, args.img_size)
        next_img = torch.from_numpy(next_img)
        next_state = torch.cat((state[0,1:,:,:], next_img))[None,:,:,:]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > args.replay_memory_size:
            del replay_memory[0]

        batch = sample(replay_memory, min(len(replay_memory), args.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch =  zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32)
        )
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.float32)[:, None]
        )
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        state_batch.to(device)
        action_batch.to(device)
        reward_batch.to(device)
        next_state_batch.to(device)

        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + args.gamma * torch.max(prediction) for reward, terminal, prediction in \
                zip(reward_batch, terminal_batch, next_prediction_batch))
        )
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        episode += 1

if __name__ == "__main__":
    
    args  = get_args()
    train(args)

