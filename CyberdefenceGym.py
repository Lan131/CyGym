
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import os
import imageio
import igraph as ig
from collections import deque
import numpy as np
import random
from CyberDefenseEnv import CyberDefenseEnv

def run_game(env, episodes, steps_per_episode):
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        print(f"Episode {episode+1}")

        for step in range(steps_per_episode):
            env.mode = 'defender' if step % 2 == 0 else 'attacker'
            action = env.sample_action()
            obs, reward, done, info, _ = env.step(action)
            print(f"{env.mode.capitalize()} Action: {action}, Reward: {reward}, Done: {done}")
            total_reward += reward
            
            if step %10 == 0 :
                env.generate_viz()

            if done:
                print("Game over")
                break
            


        print(f"Total reward: {total_reward}")

# Test the environment
env = CyberDefenseEnv()
run_game(env, episodes=1, steps_per_episode=50)

