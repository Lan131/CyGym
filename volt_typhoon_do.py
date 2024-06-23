import gym
import matplotlib.pyplot as plt
import os
import imageio
import igraph as ig
import random
import numpy as np
from collections import deque
from gym import spaces
from CyberDefenseEnv import CyberDefenseEnv, calculate_max_compromise_proportion
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv
import pickle
from CDSimulatorComponents import App, Device, OperatingSystem, Workload, os_encoder
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import csv
import logging
from do_agent import DoubleOracle
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
#logging.basicConfig(level=logging.DEBUG)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def plot_graph(g, devices, filename):
    layout = g.layout("kk")  # Kamada-Kawai layout

    # Get the positions
    positions = {i: layout[i] for i in range(len(layout))}

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the vertices
    for vertex in g.vs:
        x, y = positions[vertex.index]
        device_id = vertex["name"]
        device = devices[device_id]

        if device.attacker_owned:
            ax.scatter(x, y, color="black", marker="s", s=500)  # Black square for attacker hubs
        if device.isCompromised and not device.attacker_owned:
            ax.scatter(x, y, color="red", marker="*", s=400)  # Red star for compromised
        if not device.isCompromised and not device.attacker_owned:
            ax.scatter(x, y, color="blue", marker="o", s=100)  # Blue circle for non-compromised
        ax.text(x, y, s=str(vertex["name"]), color="black", fontsize=12, ha='right')

    # Plot the edges
    for edge in g.es:
        start, end = edge.tuple
        x_start, y_start = positions[start]
        x_end, y_end = positions[end]
        ax.plot([x_start, x_end], [y_start, y_end], color="gray")

    ax.set_title("Graph Plot")
    ax.axis('off')

    plt.savefig(filename)
    plt.close(fig)

def calculate_returns(rewards):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R
        returns.insert(0, R)
    return returns

def smooth_rewards(rewards, window=10):
    smoothed_rewards = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed_rewards.append(np.mean(rewards[start:i+1]))
    return smoothed_rewards

def initialize_reward_dict(steps_per_episode, episodes):
    max_steps = steps_per_episode * episodes + 1
    return {i: [] for i in range(0, max_steps, 100)}

def run_game(env, episodes, steps_per_episode, seed, csv_writer, test_rewards_defender, test_rewards_attacker, baseline):
    set_seed(seed)
    filenames = []

    double_oracle = DoubleOracle(env, 1, steps_per_episode, seed)


    step_count = 0
    defender_returns = []
    attacker_returns = []

    for episode in range(episodes):
        obs = env.reset()
        env.tech = "DO"  # Ensure tech is set after reset

        print(f"Episode {episode + 1}")

        # Burn-in phase
        for burn_in_step in range(100):
            env.mode = 'defender'
            action = env.sample_action()
            if burn_in_step % 7 == 0:
                action = list(action)
                action[0] = 1
                action = tuple(action)
            else:
                action = env.sample_action()
            obs, reward, done, info, Log = env.step(action)

        print(f"Burn-in for Episode {episode + 1} complete")

        burn_in_logs = env.simulator.logger.get_logs()
        try:
            train_logs, test_logs = train_test_split(burn_in_logs, test_size=0.2, random_state=42)
            env.simulator.detector.train(train_logs)
            evaluation_results = env.simulator.detector.evaluate(test_logs)
            env.simulator.logger.clear_logs()
        except:
            env.simulator.detector.train()

        defender_episode_rewards = []
        attacker_episode_rewards = []

        # Run Double Oracle algorithm to update strategies
        if episode % 2 == 0:
            print("Set double oracle mode to defender")
            double_oracle.env.mode = 'defender'
        else:
            print("Set double oracle mode to attack")
            double_oracle.env.mode = 'attacker'
        double_oracle.run()


        # Logging the test results
        test_episode_rewards_defender = []
        test_episode_rewards_attacker = []
       

        for i in range(10):  # 10 test runs
            obs = env.reset()
            env.tech = "DO"  # Ensure tech is set after reset
            defender_step_rewards = []
            attacker_step_rewards = []
            episode_compromise_ratios = []
            

            for test_step in range(30):
                
                if test_step % 2 == 0:
                    env.mode = 'defender'
                    print(double_oracle.defender_strategy_probabilities)
                    defender_strategy = double_oracle.sample_strategy(double_oracle.defender_strategies, double_oracle.defender_strategy_probabilities)
                    action = double_oracle.sample_action(defender_strategy, test_step)
                    obs, reward, done, info, Log = env.step(action)
                    defender_step_rewards.append(reward)
                else:
                    env.mode = 'attacker'
                  
                    attacker_strategy = double_oracle.sample_strategy(double_oracle.attacker_strategies, double_oracle.attacker_strategy_probabilities)
                    action = double_oracle.sample_action(attacker_strategy, test_step)
                    obs, reward, done, info, Log = env.step(action)
                    attacker_step_rewards.append(reward)

                

            test_episode_rewards_defender.append(np.sum(defender_step_rewards))
            test_episode_rewards_attacker.append(np.sum(attacker_step_rewards))
            

            # Write the attacker test rewards to the CSV within the loop for each step
            for test_step, reward in enumerate(attacker_step_rewards):
                csv_writer.writerow([seed, step_count, reward, 'attacker', baseline])

            # Write the defender test rewards to the CSV within the loop for each step
            for test_step, reward in enumerate(defender_step_rewards):
                csv_writer.writerow([seed, step_count, reward, 'defender', baseline])

        if step_count not in test_rewards_defender:
            test_rewards_defender[step_count] = []
        if step_count not in test_rewards_attacker:
            test_rewards_attacker[step_count] = []

        test_rewards_defender[step_count].append(np.mean(test_episode_rewards_defender))
        test_rewards_attacker[step_count].append(np.mean(test_episode_rewards_attacker))

        step_count += 1

        print(f"Baseline {baseline}, Step {step_count}: Defender Utility: {np.mean(test_episode_rewards_defender)}, Attacker Utility: {np.mean(test_episode_rewards_attacker)}, Defender std: {np.std(test_episode_rewards_defender)}, Attacker Std: {np.std(test_episode_rewards_attacker)}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Cyber Defense simulation.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--steps_per_episode', type=int, default=100, help='Number of steps per episode')
    parser.add_argument('--num_of_device', type=int, default=70, help='Number of devices in the environment')

    args = parser.parse_args()

    episodes = args.episodes
    steps_per_episode = args.steps_per_episode
    num_of_device = args.num_of_device

    seeds = [3]  # Add more seeds if necessary
    baselines = ["Nash", "Preset", "No Defense"]

    test_rewards_defender = {baseline: initialize_reward_dict(steps_per_episode, episodes) for baseline in baselines}
    test_rewards_attacker = {baseline: initialize_reward_dict(steps_per_episode, episodes) for baseline in baselines}
    #test_rewards_defender = {baseline: initialize_reward_dict(30, episodes) for baseline in baselines}
    #test_rewards_attacker = {baseline: initialize_reward_dict(30, episodes) for baseline in baselines}
    for baseline in baselines:
        for i in seeds:
            test_rewards_defender[baseline][0].append(0)
            test_rewards_attacker[baseline][0].append(0)

    with open('attacker_test_rewards.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|')
        csv_writer.writerow(['Seed', 'Step number', 'Reward', 'Role', 'Baseline'])

        for baseline in baselines:
            for seed in seeds:
                print(f"Running with seed {seed} and baseline {baseline}")
                set_seed(seed)

                env = Volt_Typhoon_CyberDefenseEnv()
                env.numOfDevice = num_of_device
                env.base_line = baseline
                env.tech = "DO"  # Make sure to set technique here
                env.mode = "defender" #initialize defender

                env.initialize_environment()

                run_game(
                    env, episodes, steps_per_episode, seed, csv_writer, test_rewards_defender[baseline], test_rewards_attacker[baseline], baseline
                )

    steps = sorted(test_rewards_defender[baselines[0]].keys())

    colors = {'Nash': 'blue', 'Preset': 'black', 'No Defense': 'orange'}

    plt.figure(figsize=(10, 5))
    for baseline in baselines:
        avg_avg_rewards_defender = smooth_rewards([np.mean(test_rewards_defender[baseline][k]) for k in sorted(test_rewards_defender[baseline])])
        std_avg_rewards_defender = smooth_rewards([np.std(test_rewards_defender[baseline][k]) for k in sorted(test_rewards_defender[baseline])])
        std_avg_rewards_defender = np.array(std_avg_rewards_defender) / np.sqrt(10*len(seeds))
        plt.plot(steps, avg_avg_rewards_defender, label=f'Defender Average Test Utility - {baseline}', color=colors[baseline])
        plt.fill_between(steps, np.array(avg_avg_rewards_defender) - np.array(std_avg_rewards_defender), np.array(avg_avg_rewards_defender) + np.array(std_avg_rewards_defender), alpha=0.1, color=colors[baseline])

    plt.xlabel('Episodes')
    plt.ylabel('Average Test Utility')
    plt.title('Average Test Utilities Over Episodes (Defender)')
    plt.legend()
    plt.grid()
    plt.savefig('average_test_utilities_defender.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    for baseline in baselines:
        avg_avg_rewards_attacker = smooth_rewards([np.mean(test_rewards_attacker[baseline][k]) for k in sorted(test_rewards_attacker[baseline])])
        std_avg_rewards_attacker = smooth_rewards([np.std(test_rewards_attacker[baseline][k]) for k in sorted(test_rewards_attacker[baseline])])
        std_avg_rewards_attacker = np.array(std_avg_rewards_attacker) / np.sqrt(10*len(seeds))

        plt.plot(steps, avg_avg_rewards_attacker, label=f'Attacker Average Test Utility - {baseline}', color=colors[baseline])
        plt.fill_between(steps, np.array(avg_avg_rewards_attacker) - np.array(std_avg_rewards_attacker), np.array(avg_avg_rewards_attacker) + np.array(std_avg_rewards_attacker), alpha=0.1, color=colors[baseline])

    plt.xlabel('Episodes')
    plt.yscale('linear')
    plt.ylabel('Average Test Utility')
    plt.title('Average Test Utilities Over Episodes (Attacker)')
    plt.legend()
    plt.grid()
    plt.savefig('average_test_utilities_attacker.png')
    plt.close()

    print("Simulation complete")
