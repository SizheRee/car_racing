"""
This module trains a DQN agent to play the CarRacing-v2 game using OpenAI Gym. The agent learns to maximize its reward by
exploring the game environment and updating its Q-values using a neural network. The training process is logged using
TensorboardX and the best performing model is saved to disk.

Usage:
    python train.py [--cuda] [--env ENV_NAME]

Arguments: 
    --cuda DEVICE               Enable CUDA for training on a GPU (default: -1)
    --env ENV_NAME  Name of the OpenAI Gym environment to use (default: CarRacing-v2)

Hyperparameters:
    GAMMA                       Discount factor for future rewards (default: 0.99)
    BATCH_SIZE                  Number of samples to use in each training batch (default: 32)
    REPLAY_SIZE                 Maximum size of the experience replay buffer (default: 1000)
    LEARNING_RATE               Learning rate for the neural network optimizer (default: 1e-4)
    SYNC_TARGET_FRAMES          Number of frames to wait before updating the target network (default: 500)
    REPLAY_START_SIZE           Minimum number of samples in the replay buffer before training begins (default: 1000)
    EPSILON_DECAY_LAST_FRAME    Number of frames over which to decay the exploration rate (default: 2000)
    EPSILON_START               Starting exploration rate (default: 0.9)
    EPSILON_FINAL               Final exploration rate (default: 0.01)
    MEAN_REWARD_BOUND           Threshold for the mean reward over the last 100 episodes to consider the game solved (default: 600)
"""
#!/usr/bin/env python3
import gym

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model
import utils
import envs

import logging
import os
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename='log/car-racing.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_ENV_NAME = "CarRacing-v2"
MEAN_REWARD_BOUND = 500  # 奖励阈值

GAMMA = 0.99 #  
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 50000 #150000
EPSILON_START = 0.9
EPSILON_FINAL = 0.01

OBSERVATION_SHAPE = (4,96,96)
ACTIONSPACE_SIZE = 5

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, dtype=np.float32, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, dtype=np.float32, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=-1, help="Specify CUDA device number, default=-1 (CPU)")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--save_dir", default="models", help="Directory in which to save the model weights, default=models")
    parser.add_argument("--checkpoint", help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None

    device = torch.device("cuda:%d"%args.cuda if args.cuda>-1 else "cpu")
    logger.debug("device: %s", device), print("DEBUG device:", device)
    
    env = envs.GoodWrapper(gym.make('CarRacing-v2',continuous=False, render_mode="rgb_array"))
    net = dqn_model.DQN(OBSERVATION_SHAPE, ACTIONSPACE_SIZE).to(device)
    tgt_net = dqn_model.DQN(OBSERVATION_SHAPE, ACTIONSPACE_SIZE).to(device)
    buffer = utils.ExperienceBuffer(REPLAY_SIZE)
    agent = envs.Agent(env, buffer)
    
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint))
        tgt_net.load_state_dict(net.state_dict())
        logger.info("Loaded checkpoint from %s" % args.checkpoint)
        print("DEBUG: Loaded checkpoint from %s" % args.checkpoint)
        
    writer = SummaryWriter(comment="-" + args.env)
    logger.info(net), print(net)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            logger.info("GAME DONE frame_idx %s reward: %s epsilon: %s", frame_idx, reward, epsilon)
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, m_reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon, speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                best_m_reward = m_reward
                if m_reward >= 300 and m_reward % 50 == 0: # 只有在300分以上才保存模型
                    torch.save(net.state_dict(), 
                            os.path.join(args.save_dir, args.env + "-best_%.0f.dat" % m_reward))
            if m_reward > MEAN_REWARD_BOUND: # 达到阈值，停止训练
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
