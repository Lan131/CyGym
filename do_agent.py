from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from scipy.optimize import linprog
import pickle
import nashpy as nash
import logging
import copy
import torch.nn.utils as utils
import warnings
import math
from typing import Optional, Dict, List, Tuple
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv

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
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_dim, 256).to(self.device)
        self.fc2 = nn.Linear(256, 256).to(self.device)
        self.fc3 = nn.Linear(256, action_dim).to(self.device)
        

    def forward(self, state):
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CommitteeStrategy:
    """
    A “committee” that holds one Strategy per z (exploit id) and at action-time
    asks each expert for its Q(s,a) and picks the best.
    """
    def __init__(self,
                 oracle: DoubleOracle,
                 experts: List[Tuple[Optional[int], Strategy]],
                 role: str):
        """
        oracle      -- your DoubleOracle instance (to get decode/encode, dims, seed, device)
        experts     -- list of (z, Strategy) pairs returned by train_exploit_committee
        role        -- 'attacker' or 'defender'
        """
        self.oracle  = oracle
        self.experts = experts
        self.role    = role
        self.device  = oracle.device
        self.seed    = oracle.seed

        # dims for decode/encode
        self.n_types = (oracle.n_att_types if role=='attacker'
                        else oracle.n_def_types)
        self.D       = oracle.D_init
        self.E       = oracle.E_init
        self.A       = oracle.A_init

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, int]:
        """
        Given a raw env-state vector, pick the best (atype, exps, devs, app)
        by asking each expert for its Q-value.
        """
        s_tensor = torch.tensor(state, dtype=torch.float32)\
                         .unsqueeze(0).to(self.device)

        best_Q = -float('inf')
        best_a = None

        for z, strat in self.experts:
            # 1) rebuild this expert's actor & critic
            actor  = strat.load_actor(Actor,
                                      state_dim  = state.shape[0],
                                      action_dim = self.n_types + self.D + self.E + self.A,
                                      seed       = self.seed,
                                      device     = self.device)
            critic = strat.load_critic(Critic,
                                       state_dim  = state.shape[0],
                                       action_dim = self.n_types + self.D + self.E + self.A,
                                       seed       = self.seed,
                                       device     = self.device)

            # 2) compute its continuous proposal + discrete decoding
            with torch.no_grad():
                vec = actor(s_tensor).cpu().numpy()[0]
            a_z = self.oracle.decode_action(
                vec,
                num_action_types   = self.n_types,
                num_device_indices = self.D,
                num_exploit_indices= self.E,
                num_app_indices    = self.A,
                exploit_override   = z
            )

            # 3) re-encode that discrete action into one-hot
            onehot = self.oracle.encode_action(
                a_z,
                num_action_types   = self.n_types,
                num_device_indices = self.D,
                num_exploit_indices= self.E,
                num_app_indices    = self.A
            )
            onehot_t = torch.tensor(onehot, dtype=torch.float32, device=self.device)\
                               .unsqueeze(0)

            # 4) Q(s,a)
            with torch.no_grad():
                q_val = critic(s_tensor, onehot_t).item()

            if q_val > best_Q:
                best_Q = q_val
                best_a = a_z

        return best_a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, device):
        super(Critic, self).__init__()
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #random.seed(seed)
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


def train_ddpg(actor,
            critic,
            target_actor,
            target_critic,
            replay_buffer,
            actor_optimizer,
            critic_optimizer,
            batch_size,
            gamma,
            device):
    reward_scale = 1
    max_grad_norm = .5

    if len(replay_buffer) < batch_size:
        return

    # sample batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # to-tensor & stack
    states      = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    actions     = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions]).to(device)
    next_states = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states]).to(device)
    dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    #if self.zero_day:
    #    rewards = rewards.clamp(-100.0, +100.0)
    #else:
    rewards = rewards.clamp(-10.0, +10.0)

    # --- ENSURE 2‑D SHAPES ---
    # If actions/state somehow picked up an extra singleton dim, drop it.
    if actions.dim() == 3 and actions.size(1) == 1:
        actions = actions.squeeze(1)
    if states.dim() == 3 and states.size(1) == 1:
        states = states.squeeze(1)
    if next_states.dim() == 3 and next_states.size(1) == 1:
        next_states = next_states.squeeze(1)

    # compute TD target
    with torch.no_grad():
        next_actions    = target_actor(next_states)
        # flatten next_actions too
        if next_actions.dim() == 3 and next_actions.size(1) == 1:
            next_actions = next_actions.squeeze(1)
        target_q_values = target_critic(next_states, next_actions)
        td_target       = rewards + gamma * (1 - dones) * target_q_values
        #td_target = td_target.clamp(-50.0, +50.0)

    # critic update
    current_q   = critic(states, actions)
    loss_critic = nn.SmoothL1Loss()(current_q, td_target)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()

    # actor update
    actor_optimizer.zero_grad()
    # when calling actor(states), ensure its output is 2‑D too
    pred_actions = actor(states)
    if pred_actions.dim() == 3 and pred_actions.size(1) == 1:
        pred_actions = pred_actions.squeeze(1)
    loss_actor = -critic(states, pred_actions).mean()
    loss_actor.backward()
    utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_optimizer.step()

    # soft updates
    tau = 1e-2
    for tgt, src in zip(target_actor.parameters(), actor.parameters()):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)
    for tgt, src in zip(target_critic.parameters(), critic.parameters()):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)


class Strategy:
    def __init__(self,
                 actor_state_dict: Optional[Dict[str,torch.Tensor]] = None,
                 critic_state_dict: Optional[Dict[str,torch.Tensor]] = None,
                 actions: Optional[List[Tuple]] = None,
                 baseline_name: Optional[str] = None,
                 actor_dims: Optional[Tuple[int,int]] = None,
                 critic_dims: Optional[Tuple[int,int]] = None,
                 type_mapping: Optional[Dict[int,Tuple[Dict,Dict]]]=None,):
        """
        • If this is a fixed‐action strategy, pass `actions=[(atype, devs, exps, ai), …]`
          and leave both ` actor_state_dict ` and ` critic_state_dict ` = None.
        • If this is a parametric strategy, pass in both its trained
          `actor_state_dict` and *that same* `critic_state_dict`.
        • If it’s a “baseline” (e.g. “No Defense” or “Preset”), set
          `baseline_name="No Defense"` (and leave `actor_state_dict=None`, `actions=None`).
        """
        # Parametric‐network strategy if and only if both actor+critic dicts are given:
        self.actor_state_dict  = actor_state_dict
        self.critic_state_dict = critic_state_dict
        self.actions           = actions
        self.baseline_name     = baseline_name
        self.actor_dims        = actor_dims
        self.critic_dims       = critic_dims
        self.type_mapping = type_mapping
        self.payoffs = []

    def is_parametric(self):
        return (self.actor_state_dict is not None and
                self.critic_state_dict is not None)

    def add_payoff(self, payoff):
        self.payoffs.append(payoff)

    def average_payoff(self):
        return float(np.mean(self.payoffs)) if self.payoffs else 0.0

    def load_actor(self, ActorClass, state_dim, action_dim, seed, device):
        if self.actor_state_dict is None:
            return None
        actor = ActorClass(state_dim, action_dim, seed, device)
        actor.load_state_dict(self.actor_state_dict)
        actor.to(device).eval()
        return actor

    def load_critic(self, CriticClass, state_dim, action_dim, seed, device):
        if self.critic_state_dict is None:
            return None
        critic = CriticClass(state_dim, action_dim, seed, device)
        critic.load_state_dict(self.critic_state_dict)
        critic.to(device).eval()
        return critic


    def __repr__(self):
        kind = "Parametric" if self.is_parametric() else "Fixed"
        return f"<Strategy {kind} avg_payoff={self.average_payoff():.3f}>"


class DoubleOracle:
    def __init__(self,
                 env,
                 num_episodes,
                 steps_per_episode,
                 seed,
                 baseline,
                 dynamic_neighbor_search,
                 BR_type,
                 zero_day):
        self.env = env
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.seed = seed
        self.baseline = baseline
        self.dynamic_neighbor_search = dynamic_neighbor_search
        self.BR_type = BR_type
        self.zero_day = zero_day
        self.N_MC = 6 #number of monte carlo simulations for constructing payoff matries and such



        self.defender_strategies = [ self.defense_strategy() ]
        self.attacker_strategies = [ self.init_attack_strategy() ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.payoff_matrix = np.zeros((1, 1))
        self.attacker_payoff_matrix   = np.zeros((1,1)) 

        # Some hyperparameters
        self.coord_K = 5
        self.coord_tau = 0.5
        self.coord_noise_std = 0.1


        self.epsilon = 0.05
        self.K = 1

        self.noise_std = .1
        self.coord_tau = 0.5
        self.tau       = self.coord_tau

        self.defender_equilibrium = None
        self.attacker_equilibrium = None

        # ── NOW initialize the DDPG structures ────────────────────────────
        defender_state_dim = self.env._get_defender_state().shape[0]
        attacker_state_dim = self.env._get_attacker_state().shape[0]
        n_def_actions = self.env.get_num_action_types(mode="defender")
        n_att_actions = self.env.get_num_action_types(mode="attacker")

        # Create defender DDPG dict
        self.defender_ddpg = self.init_ddpg(defender_state_dim, n_def_actions)
        # Create attacker DDPG dict
        self.attacker_ddpg = self.init_ddpg(attacker_state_dim, n_att_actions)
        self.n_def_types = n_def_actions
        self.n_att_types = n_att_actions
        self.D_init      = self.env.Max_network_size
        self.E_init      = self.env.MaxExploits
        self.A_init      = self.env.get_num_app_indices()

        # ── Only now can we save out copies of their actor+critic state_dicts ──

        # Prepare lists of “saved” actor/critic dictionaries
        self.saved_defender_actors  = []
        self.saved_defender_critics = []
        self.saved_attacker_actors  = []
        self.saved_attacker_critics = []

        # Append the freshly‐initialized actor+critic
        self.saved_defender_actors.append(
            copy.deepcopy(self.defender_ddpg['actor'].state_dict())
        )
        self.saved_defender_critics.append(
            copy.deepcopy(self.defender_ddpg['critic'].state_dict())
        )

        self.saved_attacker_actors.append(
            copy.deepcopy(self.attacker_ddpg['actor'].state_dict())
        )
        self.saved_attacker_critics.append(
            copy.deepcopy(self.attacker_ddpg['critic'].state_dict())
        )




    def decode_action(
        self,
        action_vector: np.ndarray,
        num_action_types: int,
        num_device_indices: int,
        num_exploit_indices: int,
        num_app_indices: int,
        *,
        state_tensor: torch.Tensor = None,
        actor: nn.Module = None,
        critic: nn.Module = None,
        exploit_override: Optional[int] = None
    ):
        """
        Decode a continuous action_vector into a discrete tuple:
        (action_type, exploit_indices, device_indices, app_index)

        If BR_type=="Cord_asc" and we have state+actor+critic, we call
        greedy_device_coord_ascent (which now accepts exploit_override).
        Otherwise fallback to block‐wise argmax.
        """
        # Coordinate‐ascent branch
        if (
            self.BR_type == "Cord_asc"
            and state_tensor is not None
            and actor is not None
            and critic is not None
        ):
            return self.greedy_device_coord_ascent(
                n_types         = num_action_types,
                D               = num_device_indices,
                E               = num_exploit_indices,
                A               = num_app_indices,
                state_tensor    = state_tensor,
                raw_action      = action_vector,
                actor           = actor,
                critic          = critic,
                exploit_override= exploit_override
            )

        # ── fallback: block‐wise argmax ─────────────────────────────
        # 1) action‐type (ε‐greedy)
        at_slice = action_vector[:num_action_types]
        if at_slice.size > 0:
            if np.random.rand() < self.epsilon:
                action_type = random.randint(0, num_action_types - 1)
            else:
                action_type = int(np.argmax(at_slice))
        else:
            action_type = 0

        # 2) device indices: all logits ≥0
        d0, d1 = num_action_types, num_action_types + num_device_indices
        dev_vals = action_vector[d0:d1]
        device_indices = np.where(dev_vals  > 0)[0] if dev_vals.size > 0 else np.array([], dtype=int)

        # 3) exploit index: pick exactly one via argmax
        e0, e1 = d1, d1 + num_exploit_indices
        exp_vals = action_vector[e0:e1]
        if exp_vals.size > 0:
            best = int(np.argmax(exp_vals))
            exploit_indices = np.array([best], dtype=int)
        else:
            exploit_indices = np.array([0], dtype=int)  # default to slot 0

        # 4) app index via argmax (or zero)
        a0 = e1
        if num_app_indices > 0:
            app_vals = action_vector[a0:a0+num_app_indices]
            app_index = int(np.argmax(app_vals)) if app_vals.size>0 else 0
        else:
            app_index = 0

        return (action_type, exploit_indices, device_indices, app_index)

        
    def init_ddpg(self, state_dim, num_action_types):
        num_device_indices = self.env.Max_network_size
        num_exploit_indices = self.env.MaxExploits
        num_app_indices = self.env.get_num_app_indices()

        action_dim = num_action_types + num_device_indices + num_exploit_indices + num_app_indices

        actor = Actor(state_dim, action_dim, self.seed, self.device)
        critic = Critic(state_dim, action_dim, self.seed, self.device)
        target_actor = Actor(state_dim, action_dim, self.seed, self.device)
        target_critic = Critic(state_dim, action_dim, self.seed, self.device)
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

        if self.baseline == "Random":
            l_rate = 0
        else:
            l_rate = 1e-5
           
        actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2) #was 3e-3 
        #actor_optimizer = optim.Adam(actor.parameters(), lr=2e-5) #1008078
        #critic_optimizer = optim.Adam(critic.parameters(), lr=6e-5)
        replay_buffer = ReplayBuffer(100000, self.seed)
        #print(f"[DEBUG] action_dim = {action_dim} = {num_action_types} (types) + {num_device_indices} (devices) + {num_exploit_indices} (exploits) + {num_app_indices} (apps)")



        expected_action_dim = (
        self.env.get_num_action_types(mode="attacker") +
        self.env.Max_network_size +
        self.env.MaxExploits +
        self.env.get_num_app_indices())
        #assert action_dim == expected_action_dim, f"action_dim mismatch: expected {expected_action_dim}, got {action_dim}"
        return {
            'actor': actor,
            'critic': critic,
            'target_actor': target_actor,
            'target_critic': target_critic,
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'replay_buffer': replay_buffer
        }





    def remove_dominated_strategies(self, payoff_matrix):
        """
        Identify non‑dominated strategy indices in a payoff matrix.
        A strategy i is dominated if there exists j ≠ i such that
        payoff_matrix[j, *] ≥ payoff_matrix[i, *] element‑wise
        with at least one strict inequality.

        Returns:
            non_dominated_indices (list of int): the rows of payoff_matrix to KEEP.
        """
        n = payoff_matrix.shape[0]
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # j dominates i?
                if np.all(payoff_matrix[j] >= payoff_matrix[i]) and np.any(payoff_matrix[j] > payoff_matrix[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        return non_dominated


    def solve_nash_equilibrium(self,
                            D_mat: np.ndarray,
                            A_mat: np.ndarray,
                            prune: bool = False):
        """
        Solve for (possibly pure) Nash equilibrium given the exact payoff matrices
        that were passed in.  We do NOT call build_payoff_matrices() here again.
        """
        # original sizes
        n_def, n_att = D_mat.shape
        print("*** ENTERED solve_nash_equilibrium (pure‐strategy check) ***")

        # === 1) Pure‐strategy Nash ===
        pure_nash = []
        for i in range(n_def):
            for j in range(n_att):
                if D_mat[i, j] < D_mat[:, j].max():
                    continue
                if A_mat[j, i] < A_mat[:, i].max():
                    continue
                pure_nash.append((i, j))
        if pure_nash:
            i, j = max(pure_nash, key=lambda ij: D_mat[ij])
            def_eq = np.zeros(n_def, dtype=float)
            att_eq = np.zeros(n_att, dtype=float)
            def_eq[i] = 1.0
            att_eq[j] = 1.0
            print(f"Pure‐strategy Nash found → Defender #{i}, Attacker #{j}")
            self.defender_equilibrium = def_eq
            self.attacker_equilibrium = att_eq
            return def_eq, att_eq

        # === 2) & 3) Pure‐dominance checks ===
        print("No pure‐strategy Nash; falling back to pure‐dominance check")
        # Defender fallback
        for i in range(n_def):
            if any(np.all(D_mat[k] >= D_mat[i]) and np.any(D_mat[k] > D_mat[i]) for k in range(n_def) if k != i):
                continue
            best_a = int(np.argmax(A_mat[:, i]))
            if D_mat[i, best_a] >= D_mat[:, best_a].max() and A_mat[best_a, i] >= A_mat[:, i].max():
                def_eq = np.zeros(n_def, dtype=float); def_eq[i] = 1.0
                att_eq = np.zeros(n_att, dtype=float); att_eq[best_a] = 1.0
                print(f"Pure‐dominance: Defender #{i} → Attacker best‐resp #{best_a}")
                self.defender_equilibrium = def_eq
                self.attacker_equilibrium = att_eq
                return def_eq, att_eq
        # Attacker fallback
        for j in range(n_att):
            if any(np.all(A_mat[k] >= A_mat[j]) and np.any(A_mat[k] > A_mat[j]) for k in range(n_att) if k != j):
                continue
            best_d = int(np.argmax(D_mat[:, j]))
            if D_mat[best_d, j] >= D_mat[:, j].max() and A_mat[j, best_d] >= A_mat[:, best_d].max():
                def_eq = np.zeros(n_def, dtype=float); def_eq[best_d] = 1.0
                att_eq = np.zeros(n_att, dtype=float); att_eq[j]      = 1.0
                print(f"Pure‐dominance: Attacker #{j} → Defender best‐resp #{best_d}")
                self.defender_equilibrium = def_eq
                self.attacker_equilibrium = att_eq
                return def_eq, att_eq


        # === 4) Optional: pruning dominated strategies ===
        if prune:
            # 1) figure out which defender strategies to keep
            #    but always preserve any baseline (baseline_name != None)
            old_defs = self.defender_strategies
            is_baseline_def = {
                idx for idx, strat in enumerate(old_defs)
                if strat.baseline_name is not None
            }
            if D_mat.shape[0] > 1:
                raw_keep = set(self.remove_dominated_strategies(D_mat))
                keep_def = sorted(raw_keep | is_baseline_def)
                if len(keep_def) < len(old_defs):
                    # prune the defender strategy list
                    self.defender_strategies = [old_defs[i] for i in keep_def]
                    # rebuild saved actors/critics from the surviving Strategy objects
                    self.saved_defender_actors  = [
                        strat.actor_state_dict  for strat in self.defender_strategies
                    ]
                    self.saved_defender_critics = [
                        strat.critic_state_dict for strat in self.defender_strategies
                    ]
                    # slice payoff matrices
                    D_mat = D_mat[keep_def, :]
                    A_mat = A_mat[:, keep_def]
                    print(f"Pruned defender strategies → keep {keep_def}")

            # 2) same for attacker
            old_atts = self.attacker_strategies
            is_baseline_att = {
                idx for idx, strat in enumerate(old_atts)
                if strat.baseline_name is not None
            }
            if A_mat.shape[0] > 1 and D_mat.shape[0] == A_mat.shape[1]:
                raw_keep_att = set(self.remove_dominated_strategies(A_mat))
                keep_att = sorted(raw_keep_att | is_baseline_att)
                if len(keep_att) < len(old_atts):
                    self.attacker_strategies = [old_atts[i] for i in keep_att]
                    self.saved_attacker_actors  = [
                        strat.actor_state_dict  for strat in self.attacker_strategies
                    ]
                    self.saved_attacker_critics = [
                        strat.critic_state_dict for strat in self.attacker_strategies
                    ]
                    D_mat = D_mat[:, keep_att]
                    A_mat = A_mat[keep_att, :]
                    print(f"Pruned attacker strategies → keep {keep_att}")

            # 3) write back the pruned payoff‐matrix and sizes
            self.payoff_matrix          = D_mat.copy()
            self.attacker_payoff_matrix = A_mat.copy()   
            n_def, n_att = D_mat.shape



        # === 5) Mixed‐strategy: solve with Nashpy ===
        game = nash.Game(D_mat, A_mat.T)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                all_eqs = list(game.support_enumeration())
            if not all_eqs:
                raise RuntimeError("no support‐enum equilibria")
            payoffs = [p.dot(D_mat).dot(q) for p, q in all_eqs]
            idx     = int(np.argmax(payoffs))
            def_eq, att_eq = all_eqs[idx]
            print(f"Equilibrium via support enumeration (picked {idx+1}/{len(all_eqs)})")
        except:
            try:
                lh = game.lemke_howson(initial_dropped_label=0)
                def_eq, att_eq = lh if isinstance(lh, tuple) else next(lh)
                print("Equilibrium via Lemke–Howson")
            except:
                def_eq = np.ones(n_def)/n_def
                att_eq = np.ones(n_att)/n_att
                print("All solvers failed; using uniform mix")

        # === 6) Sanity‐checks & return ===
        def_eq = np.nan_to_num(def_eq, nan=0.0, posinf=0.0, neginf=0.0)
        att_eq = np.nan_to_num(att_eq, nan=0.0, posinf=0.0, neginf=0.0)
        if def_eq.sum()==0: def_eq = np.ones(n_def)/n_def
        if att_eq.sum()==0: att_eq = np.ones(n_att)/n_att
        # fix length
        def_eq = def_eq[:len(self.defender_strategies)]
        att_eq = att_eq[:len(self.attacker_strategies)]
        def_eq /= def_eq.sum()
        att_eq /= att_eq.sum()
        self.defender_equilibrium = def_eq
        self.attacker_equilibrium = att_eq
        return def_eq, att_eq




    def generate_neighbors(
        self,
        discrete_action,
        n_samples: int = 75,
        sigma: float = 0.1
    ):
        """
        Sample nearby discrete actions by perturbing the continuous
        encoding, then decode back and keep them unique.
        """
        D = self.env.Max_network_size
        E = self.env.MaxExploits
        A = self.env.get_num_app_indices()
        n_types = self.env.get_num_action_types(self.env.mode)

        base_vec = self.encode_action(discrete_action, n_types, D, E, A)
        seen = set()
        for _ in range(n_samples):
            vec_p = base_vec + sigma * np.random.randn(*base_vec.shape)
            vec_p = np.clip(vec_p, -1.0, +1.0)
            # decode into (atype, devs_array, exps_array, appidx)

            
            at, devs, exps, ai = self.decode_action(vec_p, n_types, self.D_init, self.E_init, self.A_init)
            # turn arrays into tuple of ints for hashing
            devs_t = tuple(int(x) for x in devs)
            exps_t = tuple(int(x) for x in exps)
            seen.add((int(at), devs_t, exps_t, int(ai)))

        # convert back to the original discrete‐action format
        neighbors = []
        for at, devs_t, exps_t, ai in seen:
            neighbors.append((
                at,
                np.array(devs_t, dtype=int),
                np.array(exps_t, dtype=int),
                ai
            ))
        return neighbors


    def dynamic_neighborhood_search(
        self,
        state_tensor: torch.Tensor,
        raw_action: np.ndarray,
        actor,                      # optional, not used here
        critic,
        k_init: int = 3,
        beta_init: float = .05,
        c_k: float = 1.0,
        c_beta: float = 0.2,
        max_iters: int = 10
    ):
        """
        Simulated‐annealing over the neighborhood of raw_action.
        Returns the best discrete action found.
        """

        n_types = self.env.get_num_action_types(self.env.mode)

        # 1) initialize
        k = k_init
        beta = beta_init
        a_bar = self.decode_action(raw_action, n_types, self.D_init, self.E_init, self.A_init)



        a_best = a_bar
        Q_bar = self._Q_of(a_bar, state_tensor, critic)
        K_all = set()

        itr= 0
        # 2) SA loop
        while k > 0 or beta > 0:
            itr += 1
            if itr > max_iters:
                break
            # 3) neighbors
            A_prime = self.generate_neighbors(a_bar)
            # 4) score them
            scored = [(self._Q_of(a, state_tensor, critic), a) for a in A_prime]
            # 5) keep top-k
            scored.sort(key=lambda x: x[0], reverse=True)
            K_prime = [a for _, a in scored[:k]]
            for a in K_prime:
                # hash the action the same way
                at, devs, exps, ai = a
                K_all.add((int(at), tuple(int(x) for x in devs), tuple(int(x) for x in exps), int(ai)))

            Q_k1, a_k1 = scored[0]
            if Q_k1 > Q_bar:
                # accept improvement
                a_bar, Q_bar = a_k1, Q_k1
                if Q_k1 > self._Q_of(a_best, state_tensor, critic):
                    a_best = a_k1
            else:
                # maybe accept worse
                prob = math.exp(-(Q_bar - Q_k1) / beta) if beta > 0 else 0.0
                if random.random() < prob:
                    a_bar, Q_bar = a_k1, Q_k1
                    beta = max(0.0, beta - c_beta)
                else:
                    # reject → random from seen
                    choice = random.choice(list(K_all))
                    # turn back into array‐based action
                    a_bar = (
                        choice[0],
                        np.array(choice[1], dtype=int),
                        np.array(choice[2], dtype=int),
                        choice[3]
                    )
                    Q_bar = self._Q_of(a_bar, state_tensor, critic)

            # 15) cool k
            k = max(1, int(math.ceil(k - c_k)))

        return a_best


    def _Q_of(self, a_disc, state_tensor, critic):
        """
        Helper: encode a_disc and query critic(s, a).
        """
        D = self.env.Max_network_size
        E = self.env.MaxExploits
        A = self.env.get_num_app_indices()
        n_types = self.env.get_num_action_types(self.env.mode)

        vec = self.encode_action(
            a_disc,
            self.env.get_num_action_types(self.env.mode),
            self.env.Max_network_size,
            self.env.MaxExploits,
            self.env.get_num_app_indices()
        )

        #print(f"[DEBUG] state_tensor shape: {state_tensor.shape}")
        #print(f"[DEBUG] action_vector shape: {vec.shape}")

        with torch.no_grad():
            q = critic(
                state_tensor.to(next(critic.parameters()).device),
                torch.tensor(vec, dtype=torch.float32, device=next(critic.parameters()).device)
                     .unsqueeze(0)
            )
        return q.item()




    def defense_strategy(self):
        old_mode = self.env.mode
        self.env.mode = "defender"
        acts = [(0, *self.env.sample_action()[1:]) for _ in range(self.steps_per_episode)]
        self.env.mode = old_mode
        return Strategy(actions=acts)

    def init_attack_strategy(self):
        old_mode = self.env.mode
        self.env.mode = "attacker"
        acts = [(0, *self.env.sample_action()[1:]) for _ in range(self.steps_per_episode)]
        self.env.mode = old_mode
        return Strategy(actions=acts)

    def one_hot_encode(self, value, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[value] = 1
        return one_hot

    def encode_action(
        self,
        action: Tuple[int, np.ndarray, np.ndarray, int],
        num_action_types: int,
        num_device_indices: int,
        num_exploit_indices: int,
        num_app_indices: int
    ) -> np.ndarray:
        """
        Turn a discrete action tuple into a concatenated one-hot vector:
        (action_type, exploit_indices, device_indices, app_index).
        Corrects for any inadvertent swapping of exploit/device slots.
        """
        # 1) Unpack in the correct order
        action_type, exploit_indices, device_indices, app_index = action

        # 2) If exploit_indices looks out of range, swap with device_indices
        if exploit_indices.size > 0 and (
            exploit_indices.max() >= num_exploit_indices or
            exploit_indices.min() < 0
        ):
            exploit_indices, device_indices = device_indices, exploit_indices

        # 3) One-hot for action type
        action_type_one_hot = self.one_hot_encode(action_type, num_action_types)

        # 4) One-hot mask for the selected devices
        mask = np.zeros(num_device_indices, dtype=float)
        for d in device_indices:
            mask[d] = 1.0
        device_indices_one_hot = mask

        # 5) One-hot for the chosen exploit index (first one if many)
        exploit_indices_one_hot = self.one_hot_encode(
            int(exploit_indices[0]) if exploit_indices.size > 0 else 0,
            num_exploit_indices
        )

        # 6) One-hot for the chosen app index
        app_index_one_hot = self.one_hot_encode(app_index, num_app_indices)

        # 7) Concatenate: [atype ‖ device_mask ‖ exploit_one_hot ‖ app_one_hot]
        return np.concatenate([
            action_type_one_hot,
            device_indices_one_hot,
            exploit_indices_one_hot,
            app_index_one_hot
        ])




    def run(self, role: str, prune: bool = True):
        # 1) solve equilibrium
        def_eq, att_eq = self.solve_nash_equilibrium(prune=prune)

        # 2) choose which equilibrium to pass to the BR call
        if role == 'defender':
            br_eq    = att_eq
            br_pool  = self.attacker_strategies
        else:
            br_eq    = def_eq
            br_pool  = self.defender_strategies

        # 3) compute best response for `role`
        if self.BR_type == "ddpg" and not self.zero_day:
            new_strat = self.ddpg_best_response(
                opponent_strategies  = br_pool,
                opponent_equilibrium = br_eq,
                role                 = role,
                training_steps       = 15000 #self.steps_per_episode
            )
        if self.zero_day:
            new_strat = self.committee_best_response(br_pool, br_eq, role, training_steps=15000)
        


        # 4) add it in
        if new_strat:
            if role=='defender':
                # append to strategies
                self.defender_strategies.append(new_strat)
                # if it’s parametric, use its dict, otherwise dup the last actor‐dict
                last = self.saved_defender_actors[-1]
                actor_dict = new_strat.actor_state_dict if new_strat.actor_state_dict is not None else copy.deepcopy(last)
                self.saved_defender_actors.append(actor_dict)
                self.payoff_matrix = self.update_payoff_matrix(self.payoff_matrix, new_strat, 'defender')
            else:
                # same trick for the attacker side
                self.attacker_strategies.append(new_strat)
                last = self.saved_attacker_actors[-1]
                actor_dict = new_strat.actor_state_dict if new_strat.actor_state_dict is not None else copy.deepcopy(last)
                self.saved_attacker_actors.append(actor_dict)
                self.payoff_matrix = self.update_payoff_matrix(self.payoff_matrix, new_strat, 'attacker')
        # 5) (re‑)solve and optionally prune
        print("Resolving after adding new best response…")
        return self.solve_nash_equilibrium(prune=prune)

    def train_exploit_committee(self,
                                opponent_strategies,
                                opponent_equilibrium,
                                role,
                                training_steps=5_000,
                                σ=1,
                                σ_min=1e-5):
        """
        For each zero-day z in private_exploit_ids, run a fresh DDPG best -response
        conditioned on that z, and collect the K resulting Strategies.
        """
        experts = []
        zs = list(self.env.private_exploit_ids) or [None]

        for z in zs:
            # set z in the env_copy
            strat_z = self.ddpg_best_response(
                opponent_strategies,
                opponent_equilibrium,
                role,
                training_steps=training_steps,
                σ=σ, σ_min=σ_min,
                exploit_override=z    # 👈 you’ll need to thread this through decode_action
            )
            experts.append((z, strat_z))

        return experts   # list of (z, Strategy)




    def ddpg_best_response(
        self,
        opponent_strategies,
        opponent_equilibrium,
        role,               # 'defender' or 'attacker'
        training_steps=5_000,
        σ=1,
        σ_min=1e-5
    ):
        """
        Run a DDPG best‐response training for `training_steps` steps,
        *once per* zero‐day draw z (if zero_day=True), so that the learned
        policy π(o,z) is conditioned on every possible z.
        Returns a new Strategy (with actor + critic state_dicts).
        """

        # 1) Choose which DDPG agent we’re training:
        if role == 'defender':
            ddpg           = self.defender_ddpg
            my_state_fn    = self.env._get_defender_state
            other_state_fn = self.env._get_attacker_state
            n_types        = self.n_def_types
        else:
            ddpg           = self.attacker_ddpg
            my_state_fn    = self.env._get_attacker_state
            other_state_fn = self.env._get_defender_state
            n_types        = self.n_att_types

        # 2) Fixed dims:
        D = self.D_init
        E = self.E_init
        A = self.A_init

        # 3) Exploration noise schedule:
        decay_rate = (σ_min / σ) ** (1.0 / training_steps)
        noise_std0 = σ

        total_reward = 0.0

        '''       
        # 4) Build the list of zero-days to train on:

        if self.zero_day and getattr(self.env, "private_exploit_ids", None):
            zs = list(self.env.private_exploit_ids)
            random.shuffle(zs) #to mitigate carastrophic forgetting
        else:
            zs = [None]
        
        for z in zs:
        
        '''

        with open(self.env.snapshot_path, 'rb') as f:
            loaded = pickle.load(f)

        if isinstance(loaded, Volt_Typhoon_CyberDefenseEnv):
            # snapshot was the whole env
            env_copy = copy.deepcopy(loaded)
        else:
            # snapshot was the dict {simulator,state}
            env_copy = copy.deepcopy(self.env)
            env_copy.simulator = loaded['simulator']
            env_copy.state     = loaded['state']
        env_copy.randomize_compromise_and_ownership()
        env_copy.step_num = env_copy.defender_step = env_copy.attacker_step = 0
        env_copy.work_done = 0

        # clear any old exploits & vulnerabilities
        #env_copy.simulator.vulneralbilities.clear()
        #env_copy.simulator.exploits.clear()
        '''
        # if z is not None, set the private exploit for this episode
        if z is not None:
            env_copy.private_exploit_id = z
        '''

        # 5) Prepare to run one episode of up to `training_steps`
        state      = my_state_fn()
        dyn_eps    = 0.9
        noise_std  = noise_std0

        for t in range(training_steps):
            turn = 'defender' if (t % 2 == 0) else 'attacker'
            env_copy.mode = turn

            if turn == role:
                # --- agent’s turn: get action + noise ---
                with torch.no_grad():
                    st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    raw       = ddpg['actor'](st_tensor).cpu().numpy()[0]

                noise = np.random.normal(0.0, noise_std, size=raw.shape)
                vec   = np.clip(raw + noise, -1.0, +1.0)
                noise_std = max(σ_min, noise_std * decay_rate)

                # Decode to discrete action
                if not self.dynamic_neighbor_search:
                    action = self.decode_action(
                        vec, n_types, D, E, A,
                        state_tensor=st_tensor,
                        actor=ddpg['actor'],
                        critic=ddpg['critic']
                    )
                else:
                    if np.random.rand() < dyn_eps:
                        action = self.dynamic_neighborhood_search(
                            state_tensor=st_tensor,
                            raw_action=raw,
                            actor=ddpg['actor'],
                            critic=ddpg['critic'],
                            k_init=5, beta_init=1.0,
                            c_k=1.0, c_beta=0.1
                        )
                        dyn_eps = max(dyn_eps/2, 0.001)
                    else:
                        action = self.decode_action(
                            raw, n_types, D, E, A,
                            state_tensor=st_tensor,
                            actor=ddpg['actor'],
                            critic=ddpg['critic']
                        )

                # Step
                _, raw_reward, reward, done, *_ = env_copy.step(action)
                next_state = my_state_fn() if role == turn else other_state_fn()

                # Store and learn
                if self.BR_type != "Cord_asc":
                    ddpg['replay_buffer'].push(state, vec, reward, next_state, done)
                else:
                    disc_vec = self.encode_action(action, n_types, D, E, A)
                    ddpg['replay_buffer'].push(state, disc_vec, reward, next_state, done)

                train_ddpg(
                    actor=ddpg['actor'],
                    critic=ddpg['critic'],
                    target_actor=ddpg['target_actor'],
                    target_critic=ddpg['target_critic'],
                    replay_buffer=ddpg['replay_buffer'],
                    actor_optimizer=ddpg['actor_optimizer'],
                    critic_optimizer=ddpg['critic_optimizer'],
                    batch_size=512,
                    gamma=0.99,
                    device=self.device
                )

                total_reward += raw_reward
                state = next_state

                if done:
                    break

            else:
                # --- opponent’s turn: sample from equilibrium mix ---
                idx   = np.random.choice(len(opponent_strategies), p=opponent_equilibrium)
                strat = opponent_strategies[idx]

                if strat.baseline_name is not None:
                    env_copy.base_line = strat.baseline_name
                    action = None
                elif strat.actions is not None:
                    action = strat.actions[t % len(strat.actions)]
                else:
                    # rebuild opponent net
                    st = (env_copy._get_defender_state() if turn=='defender'
                        else env_copy._get_attacker_state())
                    n_t = (self.n_def_types if turn=='defender'
                        else self.n_att_types)
                    actor_model  = strat.load_actor(Actor,  st.shape[0], n_t+D+E+A, self.seed, self.device)
                    critic_model = strat.load_critic(Critic, st.shape[0], n_t+D+E+A, self.seed, self.device)
                    st_tensor = torch.tensor(st, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_vec = actor_model(st_tensor).cpu().numpy()[0]
                    action = self.decode_action(
                        raw_vec, n_t, D, E, A,
                        state_tensor=st_tensor,
                        actor=actor_model,
                        critic=critic_model
                    )

                _, _, done, *_ = env_copy.step(action)
                state = my_state_fn() if role != turn else other_state_fn()
                if done:
                    break

            # end of one episode for z

        # 6) Package into a new Strategy
        actor_dict  = copy.deepcopy(ddpg['actor'].state_dict())
        critic_dict = copy.deepcopy(ddpg['critic'].state_dict())
        st = my_state_fn()
        my_state_dim  = st.shape[0]
        my_action_dim = n_types + D + E + A

        new_strat = Strategy(
            actor_state_dict  = actor_dict,
            critic_state_dict = critic_dict,
            actions           = None,
            baseline_name     = None,
            actor_dims        = (my_state_dim, my_action_dim),
            critic_dims       = (my_state_dim, my_action_dim)
        )
        new_strat.add_payoff(total_reward)
        return new_strat



    def build_payoff_matrices(self):
        print("Making payoff matrix")
        num_defender_strategies = len(self.defender_strategies)
        num_attacker_strategies = len(self.attacker_strategies)

        defender_matrix = np.zeros((num_defender_strategies, num_attacker_strategies))
        attacker_matrix = np.zeros((num_attacker_strategies, num_defender_strategies))

        for i, defender_strategy in enumerate(self.defender_strategies):
            for j, attacker_strategy in enumerate(self.attacker_strategies):
                avg_defender_payoff, avg_attacker_payoff, _, _, _,_,_,_,_,_= self.simulate_game(defender_strategy, attacker_strategy)
                defender_matrix[i, j] = avg_defender_payoff
                attacker_matrix[j, i] = avg_attacker_payoff

        return defender_matrix, attacker_matrix
    
    def simulate_game(
        self,
        defender_strategy: Strategy,
        attacker_strategy: Strategy,
        num_simulations: int = 1
    ):
        """
        Run “num_simulations” independent episodes, and (if zero_day=True) average over
        each exploit in private_exploit_ids.  Returns a 10-tuple of averaged metrics:
          (avg_def_payoff,
           avg_att_payoff,
           avg_compromised_fraction,
           avg_jobs_completed,
           avg_scan_cnt,
           avg_defensive_cost,
           avg_checkpoint_cnt,
           avg_revert_cnt,
           avg_edges_blocked,
           avg_edges_added)
        """
        # Accumulators
        total_def            = 0.0
        total_att            = 0.0
        total_compromised    = 0.0
        total_jobs_completed = 0.0
        total_scan_cnt       = 0.0
        total_defensive_cost = 0.0
        total_checkpoint_cnt = 0.0
        total_revert_cnt     = 0.0
        total_edges_blocked  = 0.0
        total_edges_added    = 0.0

        # Build zero-day draws & their priors
        if self.env.zero_day:
            pool_ids = list(self.env.private_exploit_ids) or [None]
            probs = [ self.env.prior_pi.get(z, 1.0) if z is not None else 1.0
                      for z in pool_ids ]
            total_p = sum(probs)
            probs = [ p/total_p for p in probs ]
        else:
            pool_ids = [None]
            probs    = [1.0]

        # Total number of episodes we will effectively run

        N =   num_simulations

        if self.zero_day:

            # For each zdraw
            for idx_z, zdraw in enumerate(pool_ids):
                weight_z = probs[idx_z]




                for _ in range(num_simulations):
                    # --- reset a fresh copy ---
                    env_copy = copy.deepcopy(self.env)
                    env_copy.reset(from_init=True)


                    indices = random.sample(range(len(self.env.simulator.exploits)), self.env.preknown)
                    for exp in self.env.simulator.exploits:
                        exp.discovered = False
                    for idx in indices:
                        exp = self.env.simulator.exploits[idx]
                        exp.discovered = True


                    env_copy.randomize_compromise_and_ownership()
                    env_copy.step_num = env_copy.defender_step = env_copy.attacker_step = 0
                    env_copy.work_done = 0
                    env_copy.checkpoint_count = 0
                    env_copy.defensive_cost   = 0
                    env_copy.clearning_cost   = 0
                    env_copy.revert_count     = 0
                    env_copy.scan_cnt         = 0
                    env_copy.compromised_devices_cnt = 0
                    env_copy.edges_blocked           = 0
                    env_copy.edges_added             = 0

                    # fix z if zero-day
                    if zdraw is not None:
                        env_copy.private_exploit_id = zdraw
                    else:
                        env_copy.private_exploit_id = None

                    # per-episode split trackers
                    discovered   = False
                    phase1_def   = phase2_def   = 0.0
                    phase1_att   = phase2_att   = 0.0

                    final_info = {}

                    # --- run exactly self.steps_per_episode steps ---
                    for t in range(self.steps_per_episode):
                        turn = 'defender' if (t % 2 == 0) else 'attacker'
                        env_copy.mode = turn

                        # pick the right strategy
                        strat = defender_strategy if turn=='defender' else attacker_strategy

                        # (A) baseline or fixed sequence?
                        if strat.baseline_name and strat.baseline_name.lower()!='randominit':
                            env_copy.base_line = strat.baseline_name
                            action = None


                        # (B) exploit‐committee
                        elif getattr(strat, 'type_mapping', None) and 'committee' in strat.type_mapping:
                            # call our new CommitteeStrategy
                            action = strat.type_mapping['committee'].select_action()
                        # (C) fixed‐sequence
                        elif strat.actions is not None:
                            action = strat.actions[t % len(strat.actions)]
                        else:
                            # (B) parametric actor→critic decode
                            if turn=='defender':
                                st = env_copy._get_defender_state()
                                n_types = env_copy.get_num_action_types(mode='defender')
                            else:
                                st = env_copy._get_attacker_state()
                                n_types = env_copy.get_num_action_types(mode='attacker')

                            D = env_copy.Max_network_size
                            E = env_copy.MaxExploits
                            A = env_copy.get_num_app_indices()

                            st_tensor = torch.tensor(st, dtype=torch.float32)\
                                            .unsqueeze(0).to(self.device)
                            actor_model  = strat.load_actor(
                                Actor, st.shape[0], n_types+D+E+A, self.seed, self.device
                            )
                            critic_model = strat.load_critic(
                                Critic, st.shape[0], n_types+D+E+A, self.seed, self.device
                            )
                            with torch.no_grad():
                                raw_vec = actor_model(st_tensor).cpu().numpy()[0]
                            action = self.decode_action(
                                raw_vec,
                                num_action_types    = n_types,
                                num_device_indices  = D,
                                num_exploit_indices = E,
                                num_app_indices     = A,
                                state_tensor        = st_tensor,
                                actor               = actor_model,
                                critic              = critic_model
                            )

                        # step
                        _, r, _, done, info, _ = env_copy.step(action)

                        # allocate reward into phase1 or phase2
                        if self.env.zero_day:
                            # check if defender has discovered the true exploit
                            if turn=='defender' and info.get('discovered_private', False):
                                discovered = True

                            if turn=='defender':
                                if not discovered:
                                    phase1_def += r
                                else:
                                    phase2_def += r
                            else:
                                if not discovered:
                                    phase1_att += r
                                else:
                                    phase2_att += r
                        else:
                            # no zero-day → just treat all as phase1
                            if turn=='defender':
                                phase1_def += r
                            else:
                                phase1_att += r

                        if done:
                            final_info = info
                            break
                    else:
                        # if we never broke early, capture last info
                        final_info = info

                    # --- accumulate weighted payoffs ---
                    # defender: prior-weighted phase1 + full-weight phase2
                    total_def += weight_z * phase1_def + phase2_def
                    total_att += weight_z * phase1_att + phase2_att

                    # and all the side‐metrics just get prior-weighted
                    total_compromised    += final_info.get("Compromised_devices", 0.0) * weight_z
                    total_jobs_completed += final_info.get("work_done",           0.0) * weight_z
                    total_scan_cnt       += final_info.get("Scan_count",          0.0) * weight_z
                    total_defensive_cost += final_info.get("defensive_cost",      0.0) * weight_z
                    total_checkpoint_cnt += final_info.get("checkpoint_count",    0.0) * weight_z
                    total_revert_cnt     += final_info.get("revert_count",        0.0) * weight_z
                    total_edges_blocked  += final_info.get("Edges Blocked",       0.0) * weight_z
                    total_edges_added    += final_info.get("Edges Added",         0.0) * weight_z
        else:
           
            for _ in range(num_simulations):
                # --- reset a fresh copy ---
                with open(self.env.snapshot_path, 'rb') as f:
                    loaded = pickle.load(f)

                if isinstance(loaded, Volt_Typhoon_CyberDefenseEnv):
                    # snapshot was the whole env
                    env_copy = copy.deepcopy(loaded)
                else:
                    # snapshot was the dict {simulator,state}
                    env_copy = copy.deepcopy(self.env)
                    env_copy.simulator = loaded['simulator']
                    env_copy.state     = loaded['state']
                env_copy.randomize_compromise_and_ownership()
                env_copy.step_num = env_copy.defender_step = env_copy.attacker_step = 0
                env_copy.work_done = 0
                env_copy.checkpoint_count = 0
                env_copy.defensive_cost   = 0
                env_copy.clearning_cost   = 0
                env_copy.revert_count     = 0
                env_copy.scan_cnt         = 0



                def_r   = att_r   = 0.0


                final_info = {}

                # --- run exactly self.steps_per_episode steps ---
                for t in range(self.steps_per_episode):
                    turn = 'defender' if (t % 2 == 0) else 'attacker'
                    env_copy.mode = turn

                    # pick the right strategy
                    strat = defender_strategy if turn=='defender' else attacker_strategy

                    # (A) baseline or fixed sequence?
                    if strat.baseline_name and strat.baseline_name.lower()!='randominit':
                        env_copy.base_line = strat.baseline_name
                        action = None
                    elif strat.actions is not None:
                        action = strat.actions[t % len(strat.actions)]
                    else:
                        # (B) parametric actor→critic decode
                        if turn=='defender':
                            st = env_copy._get_defender_state()
                            n_types = env_copy.get_num_action_types(mode='defender')
                        else:
                            st = env_copy._get_attacker_state()
                            n_types = env_copy.get_num_action_types(mode='attacker')

                        D = env_copy.Max_network_size
                        E = env_copy.MaxExploits
                        A = env_copy.get_num_app_indices()

                        st_tensor = torch.tensor(st, dtype=torch.float32)\
                                        .unsqueeze(0).to(self.device)
                        actor_model  = strat.load_actor(
                            Actor, st.shape[0], n_types+D+E+A, self.seed, self.device
                        )
                        critic_model = strat.load_critic(
                            Critic, st.shape[0], n_types+D+E+A, self.seed, self.device
                        )
                        with torch.no_grad():
                            raw_vec = actor_model(st_tensor).cpu().numpy()[0]
                        action = self.decode_action(
                            raw_vec,
                            num_action_types    = n_types,
                            num_device_indices  = D,
                            num_exploit_indices = E,
                            num_app_indices     = A,
                            state_tensor        = st_tensor,
                            actor               = actor_model,
                            critic              = critic_model
                        )

                    # step
                    _, r, _, done, info, _ = env_copy.step(action)

                    
                    # no zero-day → just treat all as phase1
                    if turn=='defender':
                        def_r += r
                    else:
                        att_r += r

                    if done:
                        final_info = info
                        break
                else:
                    # if we never broke early, capture last info
                    final_info = info


                total_def += def_r
                total_att += att_r

                # and all the side‐metrics just get prior-weighted
                total_compromised    += final_info.get("Compromised_devices", 0.0) 
                total_jobs_completed += final_info.get("work_done",           0.0) 
                total_scan_cnt       += final_info.get("Scan_count",          0.0) 
                total_defensive_cost += final_info.get("defensive_cost",      0.0) 
                total_checkpoint_cnt += final_info.get("checkpoint_count",    0.0) 
                total_revert_cnt     += final_info.get("revert_count",        0.0) 
                total_edges_blocked  += final_info.get("Edges Blocked",       0.0) 
                total_edges_added    += final_info.get("Edges Added",         0.0) 

        # --- compute averages ---
        avg_compromised_fraction = 0.0
        if N > 0:
            # normalize by number of steps to get a fraction
            last_steps = env_copy.step_num or self.steps_per_episode
            avg_compromised_fraction = (total_compromised / N) / last_steps

        return (
            total_def            / N,
            total_att            / N,
            avg_compromised_fraction,
            total_jobs_completed / N,
            total_scan_cnt       / N,
            total_defensive_cost / N,
            total_checkpoint_cnt / N,
            total_revert_cnt     / N,
            total_edges_blocked  / N,
            total_edges_added    / N
        )



    def update_payoff_matrix(self, payoff_matrix, new_strategy, role):
        if role == 'defender':
            new_row = np.zeros((1, payoff_matrix.shape[1]))
            for j, attacker_strategy in enumerate(self.attacker_strategies):
                new_row[0, j] = self.get_payoff(new_strategy, attacker_strategy)  # Ensure it returns only defender's payoff
            payoff_matrix = np.vstack([payoff_matrix, new_row])
        elif role == 'attacker':
            new_column = np.zeros((payoff_matrix.shape[0], 1))
            for i, defender_strategy in enumerate(self.defender_strategies):
                new_column[i, 0] = self.get_payoff(defender_strategy, new_strategy)  # Ensure it returns only attacker's payoff
            payoff_matrix = np.hstack([payoff_matrix, new_column])
        return payoff_matrix


    def get_payoff(self, defender_strategy, attacker_strategy):
        total_defender_payoff, total_attacker_payoff, _, _, _,_,_,_,_,_ = self.simulate_game(defender_strategy, attacker_strategy)
        return total_defender_payoff

    def greedy_device_coord_ascent(
        self,
        n_types: int,
        D:      int,
        E:      int,
        A:      int,
        state_tensor: torch.Tensor,
        raw_action:   np.ndarray,
        actor:        nn.Module,
        critic:       nn.Module,
        exploit_override: Optional[int] = None
    ):
        """
        Top‐K sampling coordinate ascent with optional exploration noise 
        when Critic.train()==True. Uses exactly the passed‐in dims.
        Ensures exactly one exploit slot in the final action.

        If exploit_override is not None, we immediately return:
        (atype, [exploit_override], [], 0)
        """
        # ── 0) override mode ────────────────────────────────────────
        if exploit_override is not None:
            # pick greedy action‐type
            action_type = int(np.argmax(raw_action[:n_types])) if n_types>0 else 0
            return (
                action_type,
                np.array([exploit_override], dtype=int),  # exploit_indices
                np.array([], dtype=int),                  # device_indices
                0                                          # app_index
            )

        # ── 1) define “no‐op” using exploit slot 0 by default ──────
        no_op_type = n_types - 1
        no_op = (
            no_op_type,
            np.array([], dtype=int),
            np.array([0], dtype=int),  # exploit slot 0
            0
        )

        # ── 2) Q‐value helper ───────────────────────────────────────
        def Q_of(a_list):
            if isinstance(a_list, tuple):
                a_list = [a_list]
            vecs = [ self.encode_action(a, n_types, D, E, A) for a in a_list ]
            at   = torch.tensor(np.stack(vecs), dtype=torch.float32,
                                device=next(critic.parameters()).device)
            st_rep = state_tensor.repeat(len(vecs), 1)
            with torch.no_grad():
                qv = critic(st_rep, at).squeeze(1)
            return np.nan_to_num(qv.cpu().numpy(), nan=-1e9, posinf=1e9, neginf=-1e9)

        # ── 3) get base Q for no-op ─────────────────────────────────
        Q_base   = Q_of(no_op)[0]
        is_train = critic.training

        # ── 4) per‐device Top‐K sampling ────────────────────────────
        best_map = {d: no_op for d in range(D)}
        for d in range(D):
            # all (atype, e_idx) for device=d
            raw_list = [
                (atype,
                np.array([d], dtype=int),
                np.array([e_idx], dtype=int),
                0)
                for atype in range(n_types)
                for e_idx in range(E)
            ]
            cand = [(no_op, Q_base)]
            if raw_list:
                qv = Q_of(raw_list)
                if is_train:
                    qv = qv + self.coord_noise_std * np.random.randn(*qv.shape)
                cand += list(zip(raw_list, qv.tolist()))

            # keep top‐K, softmax‐sample one
            cand.sort(key=lambda x: x[1], reverse=True)
            topk = cand[: self.coord_K]
            qs   = np.array([q for _, q in topk], dtype=np.float64)
            exp_q = np.nan_to_num(np.exp(qs / self.coord_tau))
            if exp_q.sum() > 0:
                probs = exp_q / exp_q.sum()
            else:
                probs = np.ones_like(exp_q) / len(exp_q)
            choice = np.random.choice(len(topk), p=probs)
            best_map[d] = topk[choice][0]

        # ── 5) merge per‐device picks, ensure one exploit slot ───────
        final_atype = no_op_type
        devs, exps = [], []
        for (at, ds, es, _) in best_map.values():
            if at != no_op_type:
                final_atype = at
                devs.extend(ds.tolist())
                exps.extend(es.tolist())

        final_devs = np.array(devs, dtype=int) if devs else np.array([], dtype=int)

        # force at least exploit 0 if none picked
        if not exps:
            final_exps = np.array([0], dtype=int)
        else:
            final_exps = np.array([exps[0]], dtype=int)

        # ← correct order: (atype, exploit_indices, device_indices, app_index)
        return (final_atype, final_exps, final_devs, 0)




    def test_fixed_player(self,
                        fixed_role: str,
                        steps_per_episode: int,
                        test_runs: int):
        """
        Evaluate one side (the “variable” player) against a frozen opponent strategy,
        sampling the fixed side from its equilibrium mix each rollout.
        
        fixed_role: either "attacker" or "defender"
        steps_per_episode: number of steps per rollout.
        test_runs: how many independent rollouts to average over.
        
        Returns:
            List of tuples [(def_rewards, att_rewards), ...] per run.
        """
        # 1) restore the one‐true network snapshot
        with open("initial_net_DO.pkl", "rb") as f:
            snap = pickle.load(f)
        saved_sim   = snap["simulator"]
        saved_state = snap["state"]

        # 2) precompute dims for actor instantiation & decoding
        D = self.env.Max_network_size
        E = self.env.MaxExploits
        A = self.env.get_num_app_indices()
        def_types = self.env.get_num_action_types(mode="defender")
        att_types = self.env.get_num_action_types(mode="attacker")
        def_action_dim = def_types + D + E + A
        att_action_dim = att_types + D + E + A
        def_state_dim = self.env._get_defender_state().shape[0]
        att_state_dim = self.env._get_attacker_state().shape[0]

        results = []
        for _ in range(test_runs):
            # 3) sample **one** fixed strategy index from its equilibrium mix
            if fixed_role == "defender":
                probs       = self.defender_equilibrium
                strat_pool  = self.defender_strategies
                state_dim   = def_state_dim
                action_dim  = def_action_dim
            else:
                probs       = self.attacker_equilibrium
                strat_pool  = self.attacker_strategies
                state_dim   = att_state_dim
                action_dim  = att_action_dim

            idx_fixed = np.random.choice(len(strat_pool), p=probs)
            fixed_strat = strat_pool[idx_fixed]
            # instantiate its Actor net
            fixed_actor = fixed_strat.load_actor(
                Actor, state_dim, action_dim, self.seed, self.device
            )

            # 4) restore the env to the same snapshot
            self.env.simulator = copy.deepcopy(saved_sim)
            self.env.state     = copy.deepcopy(saved_state)
            self.env.step_num  = 0
            self.env.tech      = "DO"
            self.env.mode      = "defender"

            def_rews = []
            att_rews = []
            done     = False

            # 5) play out one episode
            for step in range(steps_per_episode):
                turn = "defender" if (step % 2 == 0) else "attacker"

                if turn == fixed_role:
                    # use the sampled fixed_actor
                    if turn == "defender":
                        st = self.env._get_defender_state()
                        n_types, s_dim = def_types, def_state_dim
                    else:
                        st = self.env._get_attacker_state()
                        n_types, s_dim = att_types, att_state_dim

                    with torch.no_grad():
                        vec = fixed_actor(
                            torch.tensor(st, dtype=torch.float32)
                                .unsqueeze(0)
                                .to(self.device)
                        )
                    av = vec.cpu().numpy()[0]
                    action = self.decode_action(av, n_types, D, E, A)

                else:
                    # variable side still uses its *last* strategy
                    if turn == "defender":
                        var_strat = self.defender_strategies[-1]
                    else:
                        var_strat = self.attacker_strategies[-1]
                    action = var_strat.actions[step % len(var_strat.actions)]

                _, r,_, done, _, _ = self.env.step(action)
                if turn == "defender":
                    def_rews.append(r)
                else:
                    att_rews.append(r)

                if done:
                    break
            print("def_rews:"+str(def_rews))
            print("Att rews:"+str(att_rews))
            results.append((def_rews, att_rews))

        return results


              
