import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from scipy.optimize import linprog
import pickle
from torch.optim.lr_scheduler import CyclicLR

class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seed, device):
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_dim, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, action_dim).to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, device):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_dim + action_dim, 128).to(self.device)
        self.fc2 = nn.Linear(128, 128).to(self.device)
        self.fc3 = nn.Linear(128, 1).to(self.device)

    def forward(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_ddpg(actor, critic, target_actor, target_critic, replay_buffer, actor_optimizer, critic_optimizer,actor_scheduler,critic_scheduler, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        
        return
    



    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # Convert lists to numpy arrays first
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
   
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        next_actions = target_actor(next_states)
        target_q_values = target_critic(next_states, next_actions)
        target_q_values = rewards + (gamma * target_q_values * (1 - dones))

    q_values = critic(states, actions)
    critic_loss = nn.MSELoss()(q_values, target_q_values)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()


    
    actor_loss = -critic(states, actor(states)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    actor_scheduler.step()   
    critic_scheduler.step() 
    tau = 0.005
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class Strategy:
    def __init__(self, actions):
        self.actions = actions
        self.payoffs = []
        self.probabilities = np.array([1.0]) 


    def add_payoff(self, payoff):
        self.payoffs.append(payoff)

    def average_payoff(self):
        return np.mean(self.payoffs) if self.payoffs else 0

    def __repr__(self):
        return f"Actions: {self.actions}, Average Payoff: {self.average_payoff()}"

class DoubleOracle:
    def __init__(self, env, num_episodes, steps_per_episode, seed):
        self.env = env
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.seed = seed
        self.defender_strategies = [self.defense_strategy()]
        self.attacker_strategies = [self.init_attack_strategy()]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.payoff_matrix = np.zeros((1, 1))
        self.defender_strategy_probabilities = np.array([1.0])
        self.attacker_strategy_probabilities = np.array([1.0])

        defender_state_dim = self.env._get_defender_state().shape[0]
        attacker_state_dim = self.env._get_attacker_state().shape[0]
        self.defender_ddpg = self.init_ddpg(defender_state_dim, self.env.get_num_action_types(mode="defender"))
        self.attacker_ddpg = self.init_ddpg(attacker_state_dim, self.env.get_num_action_types(mode="attacker"))

    def init_ddpg(self, state_dim, num_action_types):
        num_device_indices = self.env.numOfDevice
        num_exploit_indices = self.env.get_num_exploit_indices()
        num_app_indices = self.env.get_num_app_indices()

        action_dim = num_action_types + num_device_indices + num_exploit_indices + num_app_indices

        actor = Actor(state_dim, action_dim, self.seed, self.device)
        critic = Critic(state_dim, action_dim, self.seed, self.device)
        target_actor = Actor(state_dim, action_dim, self.seed, self.device)
        target_critic = Critic(state_dim, action_dim, self.seed, self.device)
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())
        actor_optimizer = optim.Adam(actor.parameters(), lr=1e-2)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2)
        replay_buffer = ReplayBuffer(10000, self.seed)
        actor_scheduler = CyclicLR(actor_optimizer, base_lr=1e-4, max_lr=1e-1, step_size_up=100, mode='triangular')
        critic_scheduler = CyclicLR(critic_optimizer, base_lr=1e-4, max_lr=1e-1, step_size_up=100, mode='triangular')

        return {
            'actor': actor,
            'critic': critic,
            'target_actor': target_actor,
            'target_critic': target_critic,
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'replay_buffer': replay_buffer,
            'actor_scheduler': actor_scheduler,
            'critic_scheduler': critic_scheduler
        }

    def solve_nash_equilibrium(self, defender=True):
        defender_payoff_matrix, attacker_payoff_matrix = self.build_payoff_matrices()
        if defender:
            print("updating defender_strategy_probabilities")
            print("Before Nash")
            print(self.defender_strategy_probabilities)
            self.defender_strategy_probabilities, _ = self.find_mixed_ne(defender_payoff_matrix, attacker_payoff_matrix)
            print("After nash")
            print(self.defender_strategy_probabilities)
            return self.defender_strategy_probabilities, None
        else:
            print("updating attacker_strategy_probabilities")
            _, self.attacker_strategy_probabilities = self.find_mixed_ne(attacker_payoff_matrix, defender_payoff_matrix)
            return None, self.attacker_strategy_probabilities

    def defense_strategy(self):
        actions = [(0, *self.env.sample_action()[1:]) for _ in range(self.steps_per_episode)]
        return Strategy(actions)

    def init_attack_strategy(self):
        actions = [(0, *self.env.sample_action()[1:]) for _ in range(self.steps_per_episode)]
        return Strategy(actions)

    def one_hot_encode(self, value, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[value] = 1
        return one_hot

    def encode_action(self, action, num_action_types, num_device_indices, num_exploit_indices, num_app_indices):
        action_type, device_indices, exploit_indices, app_index = action
        action_type_one_hot = self.one_hot_encode(action_type, num_action_types)
        device_indices_one_hot = sum([self.one_hot_encode(idx, num_device_indices) for idx in device_indices], np.zeros(num_device_indices))
        exploit_indices_one_hot = sum([self.one_hot_encode(idx, num_exploit_indices) for idx in exploit_indices], np.zeros(num_exploit_indices))
        app_index_one_hot = self.one_hot_encode(app_index, num_app_indices)
        return np.concatenate([action_type_one_hot, device_indices_one_hot, exploit_indices_one_hot, app_index_one_hot])

    def decode_action(self, action_vector, num_action_types, num_device_indices, num_exploit_indices, num_app_indices):
        action_type = np.argmax(action_vector[:num_action_types])
        device_start = num_action_types
        device_end = device_start + num_device_indices
        device_values = action_vector[device_start:device_end]
        device_indices = np.where(device_values >= 0)[0]

        exploit_start = device_end
        exploit_end = exploit_start + num_exploit_indices
        exploit_values = action_vector[exploit_start:exploit_end]
        exploit_indices = np.where(exploit_values >= 0)[0]

        app_start = exploit_end
        app_values = action_vector[app_start:]
        app_index = np.argmax(app_values)

        return (action_type, device_indices, exploit_indices, app_index)

    def run(self):
        if self.env.mode == 'defender':  # Update defender strategy
            new_defender_strategy = self.ddpg_best_response(self.attacker_strategies, self.defender_strategy_probabilities, 'defender')
            if new_defender_strategy:
                self.update_profile(new_defender_strategy, 'defender')
            defender_eq, _ = self.solve_nash_equilibrium(defender=True)
        else:  # Update attacker strategy
            new_attacker_strategy = self.ddpg_best_response(self.defender_strategies, self.attacker_strategy_probabilities, 'attacker')
            if new_attacker_strategy:
                self.update_profile(new_attacker_strategy, 'attacker')
            _, attacker_eq = self.solve_nash_equilibrium(defender=False)

    def ddpg_best_response(self, opponent_strategies, opponent_equilibrium, role):
        if role == 'defender':
            ddpg = self.defender_ddpg
            state = self.env._get_defender_state()
            num_action_types = self.env.get_num_action_types(mode="defender")
        else:
            ddpg = self.attacker_ddpg
            state = self.env._get_attacker_state()
            num_action_types = self.env.get_num_action_types(mode="attacker")

        num_device_indices = self.env.numOfDevice
        num_exploit_indices = self.env.get_num_exploit_indices()
        num_app_indices = self.env.get_num_app_indices()

        action_dim = num_action_types + num_device_indices + num_exploit_indices + num_app_indices

        gamma = 0.99
        batch_size = 16
        decay_rate = 0.631592

        total_reward = 0
        actions = []  # Store actions taken during the episode
        epsilon = 0.99  # Starting value of epsilon for exploration
        
        for step in range(self.steps_per_episode):
            if np.random.random() >= epsilon:
                action_vector = ddpg['actor'](torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)).cpu().detach().numpy()[0]
            else:
                action_vector = np.random.random(action_dim)
                action_vector = action_vector / np.sum(action_vector)

            action = self.decode_action(action_vector, num_action_types, num_device_indices, num_exploit_indices, num_app_indices)

            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward

            actions.append(action)  # Store each action taken
            ddpg['replay_buffer'].push(state, action_vector, reward, next_state, done)
            state = next_state

            train_ddpg(ddpg['actor'], ddpg['critic'], ddpg['target_actor'], ddpg['target_critic'], ddpg['replay_buffer'], ddpg['actor_optimizer'], ddpg['critic_optimizer'], ddpg['actor_scheduler'], ddpg['critic_scheduler'], batch_size, gamma, self.device)

            epsilon = min(.1, epsilon * decay_rate)
            if self.steps_per_episode % 100 == 0:
                epsilon = 0.1

        strategy = Strategy(actions)
        strategy.add_payoff(total_reward)

        return strategy

    def build_payoff_matrices(self):
        num_defender_strategies = len(self.defender_strategies)
        num_attacker_strategies = len(self.attacker_strategies)

        defender_matrix = np.zeros((num_defender_strategies, num_attacker_strategies))
        attacker_matrix = np.zeros((num_attacker_strategies, num_defender_strategies))

        for i, defender_strategy in enumerate(self.defender_strategies):
            for j, attacker_strategy in enumerate(self.attacker_strategies):
                avg_defender_payoff, avg_attacker_payoff = self.simulate_game(defender_strategy, attacker_strategy)
                defender_matrix[i, j] = avg_defender_payoff
                attacker_matrix[j, i] = avg_attacker_payoff

        return defender_matrix, attacker_matrix

    def simulate_game(self, defender_strategy, attacker_strategy):
        total_defender_payoff = 0.0
        total_attacker_payoff = 0.0
        num_simulations = 1

        with open('checkpoint_estimate_payoff.pkl', 'wb') as f:
            pickle.dump({'simulator': self.env.simulator}, f)
        
        for _ in range(num_simulations):
            self.env.reset()
            defender_payoff = 0.0
            attacker_payoff = 0.0

            for step in range(100):
                if step % 2 == 0:
                    action = defender_strategy.actions[step % len(defender_strategy.actions)]
                else:
                    action = attacker_strategy.actions[step % len(attacker_strategy.actions)]

                obs, reward, done, _, _ = self.env.step(action)
                if step % 2 == 0:
                    defender_payoff += reward
                else:
                    attacker_payoff += reward

            total_defender_payoff += defender_payoff
            total_attacker_payoff += attacker_payoff

            with open('checkpoint_estimate_payoff.pkl', 'rb') as f:
                checkpoint = pickle.load(f)
                self.env.simulator = checkpoint['simulator']

        avg_defender_payoff = total_defender_payoff / num_simulations
        avg_attacker_payoff = total_attacker_payoff / num_simulations

        return avg_defender_payoff, avg_attacker_payoff

            
    def find_mixed_ne(self, defender_payoff_matrix, attacker_payoff_matrix):
        """
        Finds the mixed strategy Nash equilibrium for both defender and attacker in a non-zero-sum game.
    
        Args:
            defender_payoff_matrix (numpy.ndarray): Payoff matrix for the defender.
            attacker_payoff_matrix (numpy.ndarray): Payoff matrix for the attacker.
    
        Returns:
            tuple: Tuple containing the mixed strategy Nash equilibrium for defender and attacker.
    
        Notes:
            This function computes the mixed strategy Nash equilibrium for both players in a non-zero-sum game. It uses 
            Linear Programming (LP) to find the probabilities of each player's strategies that maximize their expected 
            payoff.
    
            For each player:
            - If a pure strategy exists that maximizes their payoff against all possible strategies of the opponent, it is
              chosen.
            - Otherwise, it formulates an LP problem to maximize expected payoffs under uncertainty.
    
            Maximin Strategy:
            - If the LP solver fails for a player, it falls back to treating the game as zero sum.
    
            Random Strategy:
            - If both LP and maximin strategies fail, random probabilities are returned.
    

        """
        def maximin_strategy(payoff_matrix):
            num_strategies = payoff_matrix.shape[0]
            c = np.ones(num_strategies)
            A_ub = -payoff_matrix.T
            b_ub = -np.ones(payoff_matrix.shape[1])
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
            if res.success:
                strategy = res.x / res.x.sum()  # Normalize
                return strategy
            else:
                raise ValueError("Failed to find maximin strategy")
    
        def random_strategy(num_strategies):
            strategy = np.random.random(num_strategies)
            return strategy / strategy.sum()  # Normalize
    
        def is_pure_strategy(payoff_matrix):
            num_strategies = payoff_matrix.shape[0]
            for i in range(num_strategies):
                if all(np.all(payoff_matrix[i] >= payoff_matrix[j]) for j in range(num_strategies) if i != j):
                    return i
            return None
    
        num_defender_strategies = defender_payoff_matrix.shape[0]
        num_attacker_strategies = attacker_payoff_matrix.shape[1]
    
        defender_payoff_matrix = np.array(defender_payoff_matrix)
        attacker_payoff_matrix = np.array(attacker_payoff_matrix)
    
        # Adding epsilon to avoid zero-probability
        epsilon = 1e-5
    
        # Check for pure strategy for defender
        defender_pure_strategy = is_pure_strategy(defender_payoff_matrix)
        if defender_pure_strategy is not None:
            defender_strategy_probabilities = np.zeros(num_defender_strategies)
            defender_strategy_probabilities[defender_pure_strategy] = 1.0
        else:
            c_defender = np.zeros(num_defender_strategies + 1)
            c_defender[0] = -1  # Maximize v (negative sign to switch from minimization to maximization)
            A_eq_defender = np.zeros((num_attacker_strategies, num_defender_strategies + 1))
            A_eq_defender[:, 0] = -1  # -v
            A_eq_defender[:, 1:num_defender_strategies + 1] = np.pad(defender_payoff_matrix.T + epsilon,
                                                                     ((0, max(0, num_attacker_strategies - defender_payoff_matrix.shape[1])), (0, 0)), 'constant')
            b_eq_defender = np.zeros(num_attacker_strategies)
    
            res_defender = linprog(c=c_defender, A_eq=A_eq_defender, b_eq=b_eq_defender,
                                   bounds=[(0, None)] + [(0, None)] * num_defender_strategies, method='highs')
            if res_defender.success and not np.isnan(res_defender.x[1:]).any():
                defender_strategy_probabilities = res_defender.x[1:]
                defender_strategy_probabilities = np.nan_to_num(defender_strategy_probabilities)
                if np.sum(defender_strategy_probabilities) > 0:
                    defender_strategy_probabilities /= np.sum(defender_strategy_probabilities)  # Normalize
                else:
                    print("Sum of defender probabilities is zero, using fallback strategy.")
                    defender_strategy_probabilities = random_strategy(num_defender_strategies)
            else:
                print("Defender LP failed. Falling back to maximin or random strategy.")
                try:
                    defender_strategy_probabilities = maximin_strategy(defender_payoff_matrix)
                    print("Using maximin strategy for defender.")
                except ValueError:
                    defender_strategy_probabilities = random_strategy(num_defender_strategies)
                    print("Using random strategy for defender.")
    
        # Check for pure strategy for attacker
        attacker_pure_strategy = is_pure_strategy(attacker_payoff_matrix.T)
        if attacker_pure_strategy is not None:
            attacker_strategy_probabilities = np.zeros(num_attacker_strategies)
            attacker_strategy_probabilities[attacker_pure_strategy] = 1.0
        else:
            c_attacker = np.zeros(num_attacker_strategies + 1)
            c_attacker[0] = -1  # Maximize v (negative sign to switch from minimization to maximization)
            A_eq_attacker = np.zeros((num_defender_strategies, num_attacker_strategies + 1))
            A_eq_attacker[:, 0] = -1  # -v
            A_eq_attacker[:, 1:num_attacker_strategies + 1] = np.pad(attacker_payoff_matrix + epsilon,
                                                                   ((0, max(0, num_defender_strategies - attacker_payoff_matrix.shape[0])), (0, 0)), 'constant')
            b_eq_attacker = np.zeros(num_defender_strategies)
    
            res_attacker = linprog(c=c_attacker, A_eq=A_eq_attacker, b_eq=b_eq_attacker,
                                   bounds=[(0, None)] + [(0, None)] * num_attacker_strategies, method='highs')
            if res_attacker.success and not np.isnan(res_attacker.x[1:]).any():
                attacker_strategy_probabilities = res_attacker.x[1:]
                attacker_strategy_probabilities = np.nan_to_num(attacker_strategy_probabilities)
                if np.sum(attacker_strategy_probabilities) > 0:
                    attacker_strategy_probabilities /= np.sum(attacker_strategy_probabilities)  # Normalize
                else:
                    print("Sum of attacker probabilities is zero, using fallback strategy.")
                    attacker_strategy_probabilities = random_strategy(num_attacker_strategies)
            else:
                print("Attacker LP failed. Falling back to maximin or random strategy.")
                try:
                    attacker_strategy_probabilities = maximin_strategy(attacker_payoff_matrix.T)
                    print("Using maximin strategy for attacker.")
                except ValueError:
                    attacker_strategy_probabilities = random_strategy(num_attacker_strategies)
                    print("Using random strategy for attacker.")
    
        # Normalize probabilities to ensure they sum to 1 and handle NaNs
        defender_strategy_probabilities = np.nan_to_num(defender_strategy_probabilities).flatten()
        attacker_strategy_probabilities = np.nan_to_num(attacker_strategy_probabilities).flatten()
        defender_strategy_probabilities /= np.sum(defender_strategy_probabilities)
        attacker_strategy_probabilities /= np.sum(attacker_strategy_probabilities)
    
        return defender_strategy_probabilities, attacker_strategy_probabilities

    



    def update_profile(self, new_strategy, role):
        if self.env.mode == 'defender':
           
            self.defender_strategies.append(new_strategy)
            
        else:
            self.attacker_strategies.append(new_strategy)

        self.payoff_matrix = self.update_payoff_matrix(self.payoff_matrix, new_strategy, role)

    def update_payoff_matrix(self, payoff_matrix, new_strategy, role):
        if role == 'defender':
            new_row = np.zeros((1, payoff_matrix.shape[1]))
            for j, attacker_strategy in enumerate(self.attacker_strategies):
                new_row[0, j] = self.get_payoff(new_strategy, attacker_strategy)
            payoff_matrix = np.vstack([payoff_matrix, new_row])
        elif role == 'attacker':
            new_column = np.zeros((payoff_matrix.shape[0], 1))
            for i, defender_strategy in enumerate(self.defender_strategies):
                new_column[i, 0] = self.get_payoff(defender_strategy, new_strategy)
            payoff_matrix = np.hstack([payoff_matrix, new_column])
        return payoff_matrix

    def get_payoff(self, defender_strategy, attacker_strategy):
        total_defender_payoff, total_attacker_payoff = self.simulate_game(defender_strategy, attacker_strategy)
        return total_defender_payoff

    def sample_strategy(self, strategy_set, probabilities):
        probabilities = np.asarray(probabilities).flatten()
        probabilities /= np.sum(probabilities)
        print("Probabilities:", probabilities)
        print("Strategy set length:", len(strategy_set))
        assert len(strategy_set) == len(probabilities), "Strategy set and probabilities must have the same length"
        
        strategy_index = np.random.choice(len(strategy_set), p=probabilities)
        return strategy_set[strategy_index]


    
    def sample_action(self,strategy, step):
        return strategy.actions[step % len(strategy.actions)]
    
