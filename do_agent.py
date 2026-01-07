# do_agent.py
from __future__ import annotations

import os
import math
import copy
import pickle
import random
import logging
import warnings
from collections import deque
from typing import Optional, Dict, List, Tuple, Type
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from scipy.optimize import linprog
import nashpy as nash

import multiprocessing as mp

from timing_utils import timing
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv

# Best-response variants
from hierarchical_br import HierarchicalBestResponse
from meta_hierarchical_br import MetaHierarchicalBestResponse

# Use the shared Strategy implementation
from strategy import Strategy


# -------------------- Lightweight helpers --------------------

def clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Clone a state_dict without Python-level deepcopy.
    Tensors are detached, cloned, and moved to CPU to avoid aliasing and GPU memory build-up.
    """
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone().cpu()
        else:
            out[k] = v
    return out


class _EnvCheckpoint:
    """
    Compact snapshot of env.state and simulator state (if the simulator
    exposes get_state()/set_state()). Avoids deepcopy of large graphs.
    """
    __slots__ = ("state", "mode", "sim_state", "has_sim_state_api", "logs")
    def __init__(self, state, mode, sim_state, has_api: bool, logs):
        self.state = state
        self.mode  = mode
        self.sim_state = sim_state
        self.has_sim_state_api = has_api
        self.logs = logs  # tuple ("state", payload) or ("logs", list) or None


def _export_sim_state(sim):
    """Prefer sim.get_state() if available. Otherwise return None (no heavy copy)."""
    get_state = getattr(sim, "get_state", None)
    if callable(get_state):
        return get_state()
    return None


def _import_sim_state(sim, sim_state):
    """Prefer sim.set_state(state) if available. Returns True on success."""
    if sim_state is None:
        return False
    set_state = getattr(sim, "set_state", None)
    if callable(set_state):
        set_state(sim_state)
        return True
    return False


def _mp_worker_init():
    """Called once per worker process."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
def _get_mp_ctx():
    """
    Prefer 'spawn' when CUDA is available (safe). Use 'fork' otherwise on POSIX
    for faster worker startup. Always falls back to 'spawn' if anything goes wrong.
    """
    try:
        # Import torch lazily (module-level import exists, but keep safe)
        import torch as _torch
        # If CUDA is available we must use 'spawn' to avoid CUDA re-init in forked children
        if getattr(_torch, "cuda", None) and _torch.cuda.is_available():
            return mp.get_context("spawn")
    except Exception:
        # If torch import fails for some reason, continue to choose fork where safe
        pass

    # If not CUDA, prefer fork on POSIX (cheaper). On Windows must use spawn.
    try:
        if os.name != "nt":
            return mp.get_context("fork")
    except Exception:
        pass

    return mp.get_context("spawn")


# ------------------------------------------------------------------
# Worker-side cached DO / fast-knobs (module globals used by workers)
# ------------------------------------------------------------------
# Each worker will unpickle the sent "do_bytes" on first job and store
# it into _WORKER_DO so subsequent jobs reuse the same in-process DO.
#
# The fast-knobs below are applied to the env inside the worker to make
# rollouts cheap (turbo, caps, fast_scan, etc.). IMPORTANT: we do NOT
# modify env.work_scale here — that value is controlled by the runner.
_WORKER_DO = None
_WORKER_FAST_KNOBS = None


# --- parallel rollout worker (no .execute() dependency) ---
def _sim_rollout_worker(args):
    """
    Run a single rollout and return the per-run aggregates.
    ...
    """
    import pickle, torch, numpy as np
    from strategy import Strategy  # safe to import in worker
    global _WORKER_DO, _WORKER_FAST_KNOBS

    do_bytes, d_pl, a_pl, T, zdraw, weight_z = args

    # --- worker-local lazy unpickle of the DO (one-time) ---
    if _WORKER_DO is None:
        _WORKER_DO = pickle.loads(do_bytes)
        # ensure worker keeps DO on CPU
        _WORKER_DO.device = torch.device("cpu")
        # optional fast knobs applied once
        if _WORKER_FAST_KNOBS:
            try:
                env = _WORKER_DO.env
                for k, v in _WORKER_FAST_KNOBS.items():
                    if hasattr(env, k):
                        try:
                            setattr(env, k, v)
                        except Exception:
                            pass
            except Exception:
                pass

    do: DoubleOracle = _WORKER_DO

    # Rebuild strategies from payloads
    d_strat = Strategy.from_payload(d_pl)
    a_strat = Strategy.from_payload(a_pl)

    # Per-call model cache so we don’t rebuild every step
    _cache = {}  # id -> (actor, critic)
    def _get_models(strat):
        # Skip model loading for non-parametric strategies
        if strat.baseline_name is not None or strat.actions is not None or strat.type_mapping:
            return (None, None)

        key = id(strat)
        if key not in _cache:
            actor  = strat.load_actor(Actor,  seed=do.seed, device=do.device)
            critic = strat.load_critic(Critic, seed=do.seed, device=do.device)
            # Either/both may be None; only eval() non-None models
            if actor is not None:
                actor.eval()
                for p in actor.parameters():
                    p.requires_grad = False
            if critic is not None:
                critic.eval()
                for p in critic.parameters():
                    p.requires_grad = False
            _cache[key] = (actor, critic)
        return _cache[key]


    env = do.fresh_env()               # restore in-memory snapshot into this process’s env
    if hasattr(env, "randomize_compromise_and_ownership"):
        env.randomize_compromise_and_ownership()
    # reset counters
    for attr in ["step_num","defender_step","attacker_step","work_done",
                 "checkpoint_count","defensive_cost","clearing_cost" if False else "clearning_cost",
                 "revert_count","scan_cnt","compromised_devices_cnt",
                 "edges_blocked","edges_added"]:
        if hasattr(env, attr): setattr(env, attr, 0)

    # zero-day fixing for this rollout (if any)
    if hasattr(env, "private_exploit_id"):
        env.private_exploit_id = zdraw

    def_total = att_total = 0.0
    final_info = {}

    with torch.inference_mode():
        for t in range(T):
            turn  = "defender" if (t % 2 == 0) else "attacker"
            env.mode = turn
            strat = d_strat if turn == "defender" else a_strat

            # state for this turn
            st_vec = (env._get_defender_state() if turn == "defender"
                      else env._get_attacker_state())

            # (A) baselines (only real baselines) → None-action + env.base_line
            base = (strat.baseline_name or "").lower()
            is_true_baseline = (base in ("no attack", "no_attack", "no defense", "no_defense", "preset")) and (strat.actor_state_dict is None)
            if is_true_baseline:
                env.base_line = strat.baseline_name
                action = None

            # (B) typed strategies (hierarchical/meta/committee/MARL)
            elif strat.type_mapping:
                if "meta" in strat.type_mapping:
                    from meta_hierarchical_br import MetaHierarchicalBestResponse
                    action = MetaHierarchicalBestResponse(do, turn).execute(strat, st_vec)
                elif "hierarchical" in strat.type_mapping:
                    from hierarchical_br import HierarchicalBestResponse
                    action = HierarchicalBestResponse(do, turn).execute(strat, st_vec)
                elif "committee" in strat.type_mapping:
                    action = strat.type_mapping["committee"].select_action(st_vec)
                elif any(k in strat.type_mapping for k in ("ippo","mappo","marl")):
                    agent = strat.type_mapping.get("ippo") or strat.type_mapping.get("mappo") or strat.type_mapping.get("marl")
                    action = agent.select_action(st_vec, env=env)
                else:
                    action = None  # unknown typed mapping → no-op baseline step

            # (C) fixed sequence
            elif strat.actions is not None:
                action = strat.actions[t % len(strat.actions)]

            # (D) parametric actor-critic (includes RandomInit with weights)
            else:
                actor, critic = _get_models(strat)
                if actor is None:
                    # Broken/partial parametric; treat as a no-op baseline step
                    env.base_line = "Nash"
                    action = None
                else:
                    s_t = torch.tensor(st_vec, dtype=torch.float32, device=do.device).unsqueeze(0)
                    with torch.no_grad():
                        raw = actor(s_t).cpu().numpy()[0]
                    n_types = (do.n_def_types if turn == "defender" else do.n_att_types)
                    action = do.decode_action(
                        raw,
                        num_action_types    = n_types,
                        num_device_indices  = do.D_init,
                        num_exploit_indices = do.E_init,
                        num_app_indices     = do.A_init,
                        state_tensor        = s_t,
                        actor               = actor,
                        critic              = critic
                    )

            _, r, _, done, info, _ = env.step(action)
            if turn == "defender":
                def_total += r
            else:
                att_total += r
            if done:
                final_info = info
                break
        else:
            final_info = info

    side = {
        "Compromised_devices": final_info.get("Compromised_devices", 0.0),
        "work_done":           final_info.get("work_done",           0.0),
        "Scan_count":          final_info.get("Scan_count",          0.0),
        "defensive_cost":      final_info.get("defensive_cost",      0.0),
        "checkpoint_count":    final_info.get("checkpoint_count",    0.0),
        "revert_count":        final_info.get("revert_count",        0.0),
        "Edges Blocked":       final_info.get("Edges Blocked",       0.0),
        "Edges Added":         final_info.get("Edges Added",         0.0),
        "steps":               getattr(env, "step_num", 0),
    }
    return def_total, att_total, side, float(weight_z)



def _eval_row_worker(args):
    """
    args: (row_i, att_indices, do_serialized)
    returns: (row_i, att_indices, row_def, row_att)
    """

    global _WORKER_DO, _WORKER_FAST_KNOBS

    row_i, att_indices, do_bytes = args

    # worker-local lazy unpickle of the DO (one-time)
    if _WORKER_DO is None:
        _WORKER_DO = pickle.loads(do_bytes)
        _WORKER_DO.device = torch.device("cpu")
        if _WORKER_FAST_KNOBS:
            try:
                env = _WORKER_DO.env
                for k, v in _WORKER_FAST_KNOBS.items():
                    if hasattr(env, k):
                        try:
                            setattr(env, k, v)
                        except Exception:
                            pass
            except Exception:
                pass

    do: DoubleOracle = _WORKER_DO

    env = do.fresh_env()
    d_strat = do.defender_strategies[row_i]

    row_def, row_att = [], []
    with torch.inference_mode():
        for j in att_indices:
            do.restore(env, reset_counters=True)
            # IMPORTANT: avoid nested multiprocessing from inside a pool worker.
            dR, aR, *_ = do.simulate_game(
                d_strat,
                do.attacker_strategies[j],
                do.N_MC,
                allow_parallel=False  # ⬅ disable per-episode pool here
            )
            row_def.append(float(dR))
            row_att.append(float(aR))
    return row_i, att_indices, row_def, row_att



# -------------------- Core code --------------------

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
        # Respect the device passed in (important for worker CPU-only exec)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_dim, 256).to(self.device)
        self.fc2 = nn.Linear(256, 256).to(self.device)
        self.fc3 = nn.Linear(256, action_dim).to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed, device):
        super(Critic, self).__init__()
        # Respect the device passed in (important for worker CPU-only exec)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states      = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    actions     = torch.stack([torch.tensor(a, dtype=torch.float32) for a in actions]).to(device)
    next_states = torch.stack([torch.tensor(ns, dtype=torch.float32) for ns in next_states]).to(device)
    dones       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    rewards = rewards.clamp(-10.0, +10.0)

    if actions.dim() == 3 and actions.size(1) == 1:
        actions = actions.squeeze(1)
    if states.dim() == 3 and states.size(1) == 1:
        states = states.squeeze(1)
    if next_states.dim() == 3 and next_states.size(1) == 1:
        next_states = next_states.squeeze(1)

    with torch.no_grad():
        next_actions    = target_actor(next_states)
        if next_actions.dim() == 3 and next_actions.size(1) == 1:
            next_actions = next_actions.squeeze(1)
        target_q_values = target_critic(next_states, next_actions)
        td_target       = rewards + gamma * (1 - dones) * target_q_values

    current_q   = critic(states, actions)
    loss_critic = nn.SmoothL1Loss()(current_q, td_target)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()

    actor_optimizer.zero_grad()
    pred_actions = actor(states)
    if pred_actions.dim() == 3 and pred_actions.size(1) == 1:
        pred_actions = pred_actions.squeeze(1)
    loss_actor = -critic(states, pred_actions).mean()
    loss_actor.backward()
    utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    actor_optimizer.step()

    tau = 1e-2
    for tgt, src in zip(target_actor.parameters(), actor.parameters()):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)
    for tgt, src in zip(target_critic.parameters(), critic.parameters()):
        tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)


class CommitteeStrategy:
    """
    A “committee” that holds one Strategy per z (exploit id) and at action-time
    asks each expert for its Q(s,a) and picks the best.
    """
    def __init__(self,
                 oracle: 'DoubleOracle',
                 experts: List[Tuple[Optional[int], 'Strategy']],
                 role: str):
        self.oracle  = oracle
        self.experts = experts
        self.role    = role
        self.device  = oracle.device
        self.seed    = oracle.seed
        self.n_types = (oracle.n_att_types if role=='attacker' else oracle.n_def_types)
        self.D       = oracle.D_init
        self.E       = oracle.E_init
        self.A       = oracle.A_init

    def select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, int]:
        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        best_Q = -float('inf'); best_a = None
        for z, strat in self.experts:
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
            with torch.no_grad():
                vec = actor(s_tensor).cpu().numpy()[0]
            a_z = self.oracle.decode_action(vec, self.n_types, self.D, self.E, self.A, exploit_override=z)
            onehot = self.oracle.encode_action(a_z, self.n_types, self.D, self.E, self.A)
            onehot_t = torch.tensor(onehot, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_val = critic(s_tensor, onehot_t).item()
            if q_val > best_Q:
                best_Q = q_val; best_a = a_z
        return best_a


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
        #self.N_MC = max(2, int(round(6 * (100.0 / max(100, 6))**0.5)))
        self.N_MC = 1

        self.defender_strategies = [ self.defense_strategy() ]
        self.attacker_strategies = [ self.init_attack_strategy() ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.payoff_matrix = np.zeros((1, 1))
        self.attacker_payoff_matrix = np.zeros((1,1))

        # Coord-ascent hyperparams
        self.coord_K = 5
        self.coord_tau = 0.5
        self.coord_noise_std = 0.1
        self.epsilon = 0.05
        self.K = 1
        self.noise_std = .1
        self.tau       = self.coord_tau

        self.defender_equilibrium = None
        self.attacker_equilibrium = None
        self.merge_rule_atype     = "best_q"

        # DDPG init
        defender_state_dim = self.env._get_defender_state().shape[0]
        attacker_state_dim = self.env._get_attacker_state().shape[0]
        n_def_actions = self.env.get_num_action_types(mode="defender")
        n_att_actions = self.env.get_num_action_types(mode="attacker")
        self.defender_ddpg = self.init_ddpg(defender_state_dim, n_def_actions)
        self.attacker_ddpg = self.init_ddpg(attacker_state_dim, n_att_actions)
        self.n_def_types = n_def_actions
        self.n_att_types = n_att_actions
        self.D_init      = self.env.Max_network_size
        self.E_init      = self.env.MaxExploits
        self.A_init      = self.env.get_num_app_indices()

        self.saved_defender_actors  = [clone_state_dict(self.defender_ddpg['actor'].state_dict())]
        self.saved_defender_critics = [clone_state_dict(self.defender_ddpg['critic'].state_dict())]
        self.saved_attacker_actors  = [clone_state_dict(self.attacker_ddpg['actor'].state_dict())]
        self.saved_attacker_critics = [clone_state_dict(self.attacker_ddpg['critic'].state_dict())]

        self.checkpoint: Optional[_EnvCheckpoint] = None  # in-memory only

        # payoff caches
        self._payoff_cache: Dict[Tuple[int, int], Tuple[float, float, float, float, float, float, float, float, float, float]] = {}
        self._computed_pairs: set[Tuple[int, int]] = set()

        # persistent matrices
        self.D_mat = np.zeros((len(self.defender_strategies), len(self.attacker_strategies)))
        self.A_mat = np.zeros((len(self.attacker_strategies), len(self.defender_strategies)))

        # -------- parallel rollout config (for simulate_game) ----------
        self.parallel_rollouts: bool = True
        self.rollout_workers: int = max(1, min(mp.cpu_count() or 1, 8))
        # Attach snapshot path if env has one (set in your main driver)
        self._snapshot_path: Optional[str] = getattr(self.env, "snapshot_path", None)

    # ---------- internal helpers to produce actions from Strategy (no .execute) ----------

    @staticmethod
    def _is_true_baseline(strat: Strategy) -> bool:
        base = (getattr(strat, "baseline_name", None) or "").lower()
        # Only treat these as baselines if there are NO actor weights
        return (base in ("no attack", "no_attack", "no defense", "no_defense", "preset")) and (strat.actor_state_dict is None)

    def _load_models_cached(self, strat: Strategy, cache: dict[int, tuple[nn.Module, nn.Module]]) -> tuple[nn.Module|None, nn.Module|None]:
        """Load (actor, critic) for a parametric strategy, using a small cache keyed by id(strat)."""
        if self._is_true_baseline(strat) or strat.actions is not None or strat.type_mapping:
            return (None, None)
        key = id(strat)
        if key in cache:
            return cache[key]
        actor  = strat.load_actor(Actor,  seed=self.seed, device=self.device)
        critic = strat.load_critic(Critic, seed=self.seed, device=self.device)
        if actor is not None:
            actor.eval()
            for p in actor.parameters():
                p.requires_grad = False
        if critic is not None:
            critic.eval()
            for p in critic.parameters():
                p.requires_grad = False
        cache[key] = (actor, critic)
        return actor, critic

    # ---------- worker serialization helpers ----------
    def _rebuild_module_cpu(self, mod):
        """
        Return a CPU copy of an nn.Module instance `mod` without modifying the original.
        Attempts to reconstruct using the known Actor/Critic constructor pattern;
        falls back to deepcopy-&-cpu if introspection fails.
        """
        if mod is None:
            return None
        try:
            # best-effort introspection for our Actor/Critic classes:
            # handle cases where fc1/fc3 exist (Actor/Critic shapes)
            state_dim = getattr(mod, "fc1").in_features if hasattr(mod, "fc1") else None
            action_dim = getattr(mod, "fc3").out_features if hasattr(mod, "fc3") else None
            cls = type(mod)
            if state_dim is not None and action_dim is not None:
                new_mod = cls(state_dim, action_dim, getattr(mod, 'seed', self.seed), torch.device("cpu"))
                new_mod.load_state_dict(clone_state_dict(mod.state_dict()))
            else:
                # fallback deepcopy then move to CPU
                new_mod = copy.deepcopy(mod)
                new_mod.load_state_dict(clone_state_dict(mod.state_dict()))
            new_mod.to(torch.device("cpu"))
            new_mod.eval()
            for p in new_mod.parameters():
                p.requires_grad = False
            return new_mod
        except Exception:
            # final fallback: try deepcopy & .cpu()
            try:
                new_mod = copy.deepcopy(mod)
                try:
                    new_mod.to(torch.device("cpu"))
                except Exception:
                    pass
                new_mod.eval()
                for p in new_mod.parameters():
                    p.requires_grad = False
                return new_mod
            except Exception:
                return None

    def _worker_pickle(self) -> bytes:
        """
        Create a worker-safe pickled copy of this DoubleOracle for multiprocessing:
          - shallow-copy top-level object (so we don't mutate `self`)
          - create CPU-only copies of actor/critic modules inside defender_ddpg/attacker_ddpg
          - set device to CPU on the copy
        Returns bytes suitable for sending to worker processes.
        """
        # shallow copy of top-level container to avoid mutating self
        try:
            worker_copy = copy.copy(self)
        except Exception:
            # fallback to deepcopy if shallow copy fails (rare)
            worker_copy = copy.deepcopy(self)

        # force worker to be CPU-only
        worker_copy.device = torch.device("cpu")

        # Defensive: rebuild defender_ddpg/attacker_ddpg with CPU modules and safe lightweight fields
        def _build_safe_ddpg(ddpg_src):
            if ddpg_src is None:
                return None
            dd = {}
            # Rebuild modules to CPU-safe versions
            for k in ('actor', 'critic', 'target_actor', 'target_critic'):
                mod = ddpg_src.get(k, None)
                try:
                    dd[k] = self._rebuild_module_cpu(mod) if mod is not None else None
                except Exception:
                    dd[k] = None
            # Do not send optimizers (they reference GPU tensors). Keep replay_buffer (python-only).
            dd['actor_optimizer'] = None
            dd['critic_optimizer'] = None
            dd['replay_buffer'] = ddpg_src.get('replay_buffer', None)
            return dd

        try:
            worker_copy.defender_ddpg = _build_safe_ddpg(self.defender_ddpg)
        except Exception:
            worker_copy.defender_ddpg = None

        try:
            worker_copy.attacker_ddpg = _build_safe_ddpg(self.attacker_ddpg)
        except Exception:
            worker_copy.attacker_ddpg = None

        # Ensure saved_* lists are CPU-cloned state_dicts (they may be large)
        try:
            worker_copy.saved_defender_actors = [clone_state_dict(x) for x in getattr(self, 'saved_defender_actors', [])]
            worker_copy.saved_defender_critics = [clone_state_dict(x) for x in getattr(self, 'saved_defender_critics', [])]
            worker_copy.saved_attacker_actors = [clone_state_dict(x) for x in getattr(self, 'saved_attacker_actors', [])]
            worker_copy.saved_attacker_critics = [clone_state_dict(x) for x in getattr(self, 'saved_attacker_critics', [])]
        except Exception:
            pass

        # Avoid sending large caches if present
        try:
            worker_copy._payoff_cache = {}
            worker_copy._computed_pairs = set()
        except Exception:
            pass

        # Finally pickle the worker-safe copy
        return pickle.dumps(worker_copy)

    def _strategy_decide_action(self,
                                env,
                                strat: Strategy,
                                turn: str,
                                t_step: Optional[int],
                                model_cache: dict[int, tuple[nn.Module, nn.Module]]):

        # (A) real baselines

        if self._is_true_baseline(strat):
            #print("baseline calculation start")
            env.base_line = strat.baseline_name
            return None
        

        # (B) typed strategies
        if strat.type_mapping:
            #print("Get state start")
            st_vec = env._get_defender_state() if turn == "defender" else env._get_attacker_state()
            #print("Get state done")
            if "meta" in strat.type_mapping:
                return MetaHierarchicalBestResponse(self, turn).execute(strat, st_vec)
            if "hierarchical" in strat.type_mapping:
                return HierarchicalBestResponse(self, turn).execute(strat, st_vec)
            if "committee" in strat.type_mapping:
                return strat.type_mapping["committee"].select_action(st_vec)
            if any(k in strat.type_mapping for k in ("ippo","mappo","marl")):
                agent = strat.type_mapping.get("ippo") or strat.type_mapping.get("mappo") or strat.type_mapping.get("marl")
                return agent.select_action(st_vec, env=env)
            # unknown typed → fall through to no-op
            env.base_line = "Nash"
            return None

        # (C) fixed sequence
        if strat.actions is not None:
            idx = (t_step or 0) % len(strat.actions)
            return strat.actions[idx]

        # (D) parametric (includes "RandomInit" with weights)
        actor, critic = self._load_models_cached(strat, model_cache)
        if actor is None:
            env.base_line = "Nash"
            return None
        st = env._get_defender_state() if turn == "defender" else env._get_attacker_state()
        s_t = torch.tensor(st, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw = actor(s_t).cpu().numpy()[0]
        n_types = (self.n_def_types if turn == "defender" else self.n_att_types)
        return self.decode_action(
            raw,
            num_action_types    = n_types,
            num_device_indices  = self.D_init,
            num_exploit_indices = self.E_init,
            num_app_indices     = self.A_init,
            state_tensor        = s_t,
            actor               = actor,
            critic              = critic
        )

    # ---------- matrix sizing ----------
    def _ensure_matrix_sizes(self):
        n_def = len(self.defender_strategies)
        n_att = len(self.attacker_strategies)
        if self.D_mat.shape != (n_def, n_att):
            D_new = np.zeros((n_def, n_att))
            d_old, a_old = self.D_mat.shape
            D_new[:d_old, :a_old] = self.D_mat
            self.D_mat = D_new
        if self.A_mat.shape != (n_att, n_def):
            A_new = np.zeros((n_att, n_def))
            a_old, d_old = self.A_mat.shape
            A_new[:a_old, :d_old] = self.A_mat
            self.A_mat = A_new

    # -------- in-memory checkpoint --------
    @staticmethod
    def _snapshot_logger(sim) -> tuple | None:
        lg = getattr(sim, "logger", None)
        if lg is None:
            return None
        lg_get_state = getattr(lg, "get_state", None)
        if callable(lg_get_state):
            try:
                return ("state", lg_get_state())
            except Exception:
                pass
        lg_get_logs = getattr(lg, "get_logs", None)
        if callable(lg_get_logs):
            try:
                data = lg_get_logs()
                if isinstance(data, list):
                    return ("logs", data[-2000:].copy())
                return ("logs", list(data)[-2000:])
            except Exception:
                pass
        return None

    def checkpoint_now(self):
        if getattr(self.env, "simulator", None) is None or getattr(self.env, "state", None) is None:
            raise RuntimeError("DoubleOracle.checkpoint_now(): env not initialized (simulator/state missing).")
        sim_state = _export_sim_state(self.env.simulator)
        has_api   = (sim_state is not None)
        if not has_api:
            logging.warning(
                "[checkpoint_now] simulator has no get_state()/set_state(); "
                "we will not copy simulator internals. Ensure your per-episode reset fully restores dynamics."
            )
        logger_snap = self._snapshot_logger(self.env.simulator)
        self.checkpoint = _EnvCheckpoint(
            state   = (self.env.state.copy() if hasattr(self.env.state, "copy") else copy.deepcopy(self.env.state)),
            mode    = self.env.mode,
            sim_state = sim_state,
            has_api   = has_api,
            logs      = logger_snap
        )

    @staticmethod
    def _restore_logger(sim, snap):
        if not snap:
            return
        lg = getattr(sim, "logger", None)
        if lg is None: return
        kind, payload = snap
        if kind == "state":
            lg_set_state = getattr(lg, "set_state", None)
            if callable(lg_set_state):
                try:
                    lg_set_state(payload); return
                except Exception:
                    pass
        elif kind == "logs":
            lg_set_logs = getattr(lg, "set_logs", None)
            if callable(lg_set_logs):
                try:
                    lg_set_logs(list(payload)); return
                except Exception:
                    pass
            for attr in ("buffer", "logs", "_logs", "_buffer"):
                if hasattr(lg, attr):
                    try:
                        setattr(lg, attr, list(payload)); return
                    except Exception:
                        pass

    def restore(self, env_obj=None, *, reset_counters: bool = True) -> bool:
        if not getattr(self, "checkpoint", None):
            return False
        tgt = env_obj or self.env

        sim_restored = False
        if self.checkpoint.has_sim_state_api:
            ok = _import_sim_state(tgt.simulator, self.checkpoint.sim_state)
            sim_restored = bool(ok)
            if not ok:
                logging.warning("[restore] set_state() failed or unavailable despite checkpoint; simulator left unchanged.")

        tgt.state = (self.checkpoint.state.copy() if hasattr(self.checkpoint.state, "copy")
                    else copy.deepcopy(self.checkpoint.state))
        tgt.mode  = self.checkpoint.mode

        need_logger_restore = True
        try:
            lg = getattr(tgt.simulator, "logger", None)
            if sim_restored and lg is not None:
                gl = getattr(lg, "get_logs", None)
                if callable(gl):
                    cur = gl()
                    if cur and len(cur) > 0:
                        need_logger_restore = False
        except Exception:
            pass
        if need_logger_restore:
            self._restore_logger(tgt.simulator, self.checkpoint.logs)

        if reset_counters:
            for attr in [
                "step_num","defender_step","attacker_step","work_done",
                "checkpoint_count","defensive_cost","clearing_cost",
                "revert_count","scan_cnt","compromised_devices_cnt",
                "edges_blocked","edges_added"
            ]:
                if hasattr(tgt, attr):
                    setattr(tgt, attr, 0)

        if hasattr(tgt, "_rebuild_graph_cache"):
            try:
                tgt._rebuild_graph_cache()
            except Exception:
                pass

        return True

    def fresh_env(self):
        """Restore the in-memory snapshot into self.env and return it."""
        self.restore(self.env, reset_counters=True)
        return self.env

    # ---------- action encode/decode ----------
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
        action_type, exploit_indices, device_indices, app_index = action
        if exploit_indices.size > 0 and (exploit_indices.max() >= num_exploit_indices or exploit_indices.min() < 0):
            exploit_indices, device_indices = device_indices, exploit_indices
        action_type_one_hot = self.one_hot_encode(action_type, num_action_types)
        mask = np.zeros(num_device_indices, dtype=float)
        for d in device_indices:
           

            mask[d] = 1.0
        device_indices_one_hot = mask
        exploit_indices_one_hot = self.one_hot_encode(
            int(exploit_indices[0]) if exploit_indices.size > 0 else 0,
            num_exploit_indices
        )
        app_index_one_hot = self.one_hot_encode(app_index, num_app_indices)
        return np.concatenate([action_type_one_hot, device_indices_one_hot, exploit_indices_one_hot, app_index_one_hot])

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
        orig_D, orig_E, orig_A = self.D_init, self.E_init, self.A_init
        if (num_device_indices, num_exploit_indices, num_app_indices) != (orig_D, orig_E, orig_A):
            num_device_indices, num_exploit_indices, num_app_indices = orig_D, orig_E, orig_A

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

        at_slice = action_vector[:num_action_types]
        if at_slice.size > 0:
            if np.random.rand() < self.epsilon:
                action_type = random.randint(0, num_action_types - 1)
            else:
                action_type = int(np.argmax(at_slice))
        else:
            action_type = 0

        d0, d1 = num_action_types, num_action_types + num_device_indices
        dev_vals = action_vector[d0:d1]
        device_indices = np.where(dev_vals > 0)[0] if dev_vals.size > 0 else np.array([], dtype=int)

        e0, e1 = d1, d1 + num_exploit_indices
        exp_vals = action_vector[e0:e1]
        if exp_vals.size > 0:
            best = int(np.argmax(exp_vals))
            exploit_indices = np.array([best], dtype=int)
        else:
            exploit_indices = np.array([0], dtype=int)

        a0 = e1
        if num_app_indices > 0:
            app_vals = action_vector[a0:a0+num_app_indices]
            app_index = int(np.argmax(app_vals)) if app_vals.size>0 else 0
        else:
            app_index = 0

        return (action_type, exploit_indices, device_indices, app_index)

    # ---------- initial strategies ----------
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

    # ---------- DDPG setup ----------
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

        actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = optim.Adam(critic.parameters(), lr=1e-2)
        replay_buffer = ReplayBuffer(100000, self.seed)
        return {
            'actor': actor,
            'critic': critic,
            'target_actor': target_actor,
            'target_critic': target_critic,
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer,
            'replay_buffer': replay_buffer
        }

    # ---------- Dominance / Nash ----------
    def remove_dominated_strategies(self, payoff_matrix):
        n = payoff_matrix.shape[0]
        non_dominated = []
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j: continue
                if np.all(payoff_matrix[j] >= payoff_matrix[i]) and np.any(payoff_matrix[j] > payoff_matrix[i]):
                    dominated = True; break
            if not dominated:
                non_dominated.append(i)
        return non_dominated

    def solve_nash_equilibrium(self,
                               D_mat: np.ndarray,
                               A_mat: np.ndarray,
                               prune: bool = False):
        n_def, n_att = D_mat.shape
        print("*** ENTERED solve_nash_equilibrium (pure‐strategy check) ***")

        with timing("Nash: pure_strategy_check"):
            pure_nash = []
            col_max = D_mat.max(axis=0) if n_def > 0 and n_att > 0 else np.array([])
            row_max = A_mat.max(axis=0) if n_att > 0 and n_def > 0 else np.array([])
            for i in range(n_def):
                for j in range(n_att):
                    if D_mat[i, j] < col_max[j]: continue
                    if A_mat[j, i] < row_max[i]: continue
                    pure_nash.append((i, j))

        if pure_nash:
            i, j = max(pure_nash, key=lambda ij: D_mat[ij])
            def_eq = np.zeros(n_def, dtype=float); def_eq[i] = 1.0
            att_eq = np.zeros(n_att, dtype=float); att_eq[j] = 1.0
            print(f"Pure‐strategy Nash found → Defender #{i}, Attacker #{j}")
            self.defender_equilibrium = def_eq
            self.attacker_equilibrium = att_eq
            return def_eq, att_eq

        print("No pure‐strategy Nash; falling back to dominance/probabilistic")

        if prune:
            with timing("Nash: prune_dominated"):
                # defender
                old_defs = self.defender_strategies
                is_baseline_def = {idx for idx, strat in enumerate(old_defs) if strat.baseline_name is not None}
                if D_mat.shape[0] > 1:
                    raw_keep = set(self.remove_dominated_strategies(D_mat))
                    keep_def = sorted(raw_keep | is_baseline_def)
                    if len(keep_def) < len(old_defs):
                        self.defender_strategies = [old_defs[i] for i in keep_def]
                        self.saved_defender_actors  = [strat.actor_state_dict  for strat in self.defender_strategies]
                        self.saved_defender_critics = [strat.critic_state_dict for strat in self.defender_strategies]
                        D_mat = D_mat[keep_def, :]
                        A_mat = A_mat[:, keep_def]
                        print(f"Pruned defender strategies → keep {keep_def}")

                # attacker
                old_atts = self.attacker_strategies
                is_baseline_att = {idx for idx, strat in enumerate(old_atts) if strat.baseline_name is not None}
                if A_mat.shape[0] > 1 and D_mat.shape[0] == A_mat.shape[1]:
                    raw_keep_att = set(self.remove_dominated_strategies(A_mat))
                    keep_att = sorted(raw_keep_att | is_baseline_att)
                    if len(keep_att) < len(old_atts):
                        self.attacker_strategies = [old_atts[i] for i in keep_att]
                        self.saved_attacker_actors  = [strat.actor_state_dict  for strat in self.attacker_strategies]
                        self.saved_attacker_critics = [strat.critic_state_dict for strat in self.attacker_strategies]
                        D_mat = D_mat[:, keep_att]
                        A_mat = A_mat[keep_att, :]
                        print(f"Pruned attacker strategies → keep {keep_att}")

                self.payoff_matrix          = D_mat.copy()
                self.attacker_payoff_matrix = A_mat.copy()
                n_def, n_att = D_mat.shape

        with timing("Nash: build_game"):
            game = nash.Game(D_mat, A_mat.T)

        solved = False
        with timing("Nash: support_enumeration"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    all_eqs = list(game.support_enumeration())
                if all_eqs:
                    with timing("Nash: pick_best_support"):
                        payoffs = [p.dot(D_mat).dot(q) for p, q in all_eqs]
                        idx     = int(np.argmax(payoffs))
                        def_eq, att_eq = all_eqs[idx]
                        print(f"Equilibrium via support enumeration (picked {idx+1}/{len(all_eqs)})")
                        solved = True
            except Exception:
                pass

        if not solved:
            with timing("Nash: lemke_howson"):
                try:
                    lh = game.lemke_howson(initial_dropped_label=0)
                    def_eq, att_eq = lh if isinstance(lh, tuple) else next(lh)
                    print("Equilibrium via Lemke–Howson")
                    solved = True
                except Exception:
                    pass

        if not solved:
            with timing("Nash: uniform_fallback"):
                def_eq = np.ones(n_def)/n_def
                att_eq = np.ones(n_att)/n_att
                print("All solvers failed; using uniform mix")

        with timing("Nash: sanitize_and_normalize"):
            def_eq = np.nan_to_num(def_eq, nan=0.0, posinf=0.0, neginf=0.0)
            att_eq = np.nan_to_num(att_eq, nan=0.0, posinf=0.0, neginf=0.0)
            if def_eq.sum() == 0: def_eq = np.ones(n_def)/n_def
            if att_eq.sum() == 0: att_eq = np.ones(n_att)/n_att
            def_eq = def_eq[:len(self.defender_strategies)]
            att_eq = att_eq[:len(self.attacker_strategies)]
            def_eq /= def_eq.sum()
            att_eq /= att_eq.sum()

        self.defender_equilibrium = def_eq
        self.attacker_equilibrium = att_eq
        return def_eq, att_eq

    # ---------- neighborhood search helpers ----------
    def generate_neighbors(self, discrete_action, n_samples: int = 75, sigma: float = 0.1):
        D = self.env.Max_network_size
        E = self.env.MaxExploits
        A = self.env.get_num_app_indices()
        n_types = self.env.get_num_action_types(self.env.mode)

        base_vec = self.encode_action(discrete_action, n_types, D, E, A)
        seen = set()
        for _ in range(n_samples):
            vec_p = base_vec + sigma * np.random.randn(*base_vec.shape)
            vec_p = np.clip(vec_p, -1.0, +1.0)
            at, devs, exps, ai = self.decode_action(vec_p, n_types, self.D_init, self.E_init, self.A_init)
            devs_t = tuple(int(x) for x in devs)
            exps_t = tuple(int(x) for x in exps)
            seen.add((int(at), devs_t, exps_t, int(ai)))

        neighbors = []
        for at, devs_t, exps_t, ai in seen:
            neighbors.append((at, np.array(devs_t, dtype=int), np.array(exps_t, dtype=int), ai))
        return neighbors

    def _Q_of(self, a_disc, state_tensor, critic):
        vec = self.encode_action(
            a_disc,
            self.env.get_num_action_types(self.env.mode),
            self.env.Max_network_size,
            self.env.MaxExploits,
            self.env.get_num_app_indices()
        )
        with torch.no_grad():
            q = critic(
                state_tensor.to(next(critic.parameters()).device),
                torch.tensor(vec, dtype=torch.float32, device=next(critic.parameters()).device).unsqueeze(0)
            )
        return q.item()

    def dynamic_neighborhood_search(self,
                                    state_tensor: torch.Tensor,
                                    raw_action: np.ndarray,
                                    actor, critic,
                                    k_init: int = 3,
                                    beta_init: float = .05,
                                    c_k: float = 1.0,
                                    c_beta: float = 0.2,
                                    max_iters: int = 10):
        n_types = self.env.get_num_action_types(self.env.mode)
        k = k_init; beta = beta_init
        a_bar = self.decode_action(raw_action, n_types, self.D_init, self.E_init, self.A_init)
        a_best = a_bar
        Q_bar = self._Q_of(a_bar, state_tensor, critic)
        K_all = set()
        itr= 0
        while k > 0 or beta > 0:
            itr += 1
            if itr > max_iters: break
            A_prime = self.generate_neighbors(a_bar)
            scored = [(self._Q_of(a, state_tensor, critic), a) for a in A_prime]
            scored.sort(key=lambda x: x[0], reverse=True)
            topk = cand = scored[:k]
            for _, a in topk:
                at, devs, exps, ai = a
                K_all.add((int(at), tuple(int(x) for x in devs), tuple(int(x) for x in exps), int(ai)))
            if not scored:
                break
            Q_k1, a_k1 = scored[0]
            if Q_k1 > Q_bar:
                a_bar, Q_bar = a_k1, Q_k1
                if Q_k1 > self._Q_of(a_best, state_tensor, critic):
                    a_best = a_k1
            else:
                prob = math.exp(-(Q_bar - Q_k1) / beta) if beta > 0 else 0.0
                if random.random() < prob:
                    a_bar, Q_bar = a_k1, Q_k1
                    beta = max(0.0, beta - c_beta)
                else:
                    choice = random.choice(list(K_all))
                    a_bar = (choice[0],
                             np.array(choice[1], dtype=int),
                             np.array(choice[2], dtype=int),
                             choice[3])
                    Q_bar = self._Q_of(a_bar, state_tensor, critic)
            k = max(1, int(math.ceil(k - c_k)))
        return a_best

    # ---------- committee BR ----------
    def committee_best_response(self, opponent_strategies, opponent_equilibrium, role,
                                training_steps=5000, σ=1, σ_min=1e-5):
        experts = self.train_exploit_committee(opponent_strategies, opponent_equilibrium, role,
                                               training_steps=training_steps, σ=σ, σ_min=σ_min)
        committee = CommitteeStrategy(self, experts, role)
        return Strategy(type_mapping={'committee': committee})

    def train_exploit_committee(self,
                                opponent_strategies,
                                opponent_equilibrium,
                                role,
                                training_steps=5_000,
                                σ=1,
                                σ_min=1e-5):
        experts = []
        zs = list(getattr(self.env, "private_exploit_ids", [])) or [None]
        for z in zs:
            strat_z = self.ddpg_best_response(opponent_strategies,
                                              opponent_equilibrium,
                                              role,
                                              training_steps=training_steps,
                                              σ=σ, σ_min=σ_min,
                                              exploit_override=z)
            experts.append((z, strat_z))
        return experts

    # ---------- DDPG BR ----------
    def ddpg_best_response(self,
                        opponent_strategies,
                        opponent_equilibrium,
                        role,
                        training_steps=5_000,
                        σ=1,
                        σ_min=1e-5,
                        exploit_override: int | None = None,
                        meta_controller=None):
        """
        DDPG best-response training with optional MetaDOAR observer training.

        If meta_controller is not None, we:
        - Let the meta-controller select a subset of nodes for the *current state*
            (in observer mode; we don't change DOAR's action).
        - Log (state, selected_nodes, reward, next_state, done) into its replay.
        - Periodically call meta_controller.train_controller().
        """
        # Fresh environment for BR training
        env_copy = self.fresh_env()
        if hasattr(env_copy, "randomize_compromise_and_ownership"):
            env_copy.randomize_compromise_and_ownership()
        env_copy.step_num = env_copy.defender_step = env_copy.attacker_step = 0
        env_copy.work_done = 0

        # Bind state functions to *env_copy* (not self.env)
        if role == 'defender':
            ddpg           = self.defender_ddpg
            my_state_fn    = env_copy._get_defender_state
            other_state_fn = env_copy._get_attacker_state
            n_types        = self.n_def_types
        else:
            ddpg           = self.attacker_ddpg
            my_state_fn    = env_copy._get_attacker_state
            other_state_fn = env_copy._get_defender_state
            n_types        = self.n_att_types

        # If we have a meta-controller, point it at this env_copy for training
        if meta_controller is not None:
            meta_controller.env = env_copy  # override its internal copy for this BR run

        D, E, A = self.D_init, self.E_init, self.A_init
        decay_rate = (σ_min / σ) ** (1.0 / training_steps)
        noise_std0 = σ

        total_reward = 0.0

        state      = my_state_fn()
        dyn_eps    = 0.9
        noise_std  = noise_std0

        # small cache for opponent models in this BR
        _opp_model_cache: dict[int, tuple[nn.Module, nn.Module]] = {}

        for t in range(training_steps):
            turn = 'defender' if (t % 2 == 0) else 'attacker'
            env_copy.mode = turn

            if turn == role:
                # ------------------------------
                # Meta-controller: observer mode
                # ------------------------------
                if meta_controller is not None:
                    try:
                        v_meta = build_visibility_mask(env_copy, role=role).cpu().numpy()
                        visible_mask = (v_meta > 0.5).astype(bool)
                    except Exception:
                        visible_mask = None

                    # Select subset of devices under current meta-parameters
                    selected_nodes = meta_controller.select_devices(
                        env_copy,
                        state_vec=state,
                        visible_mask=visible_mask,
                        k=meta_controller.select_k
                    )

                    if selected_nodes:
                        meta_controller._last_selection = {
                            "state": np.asarray(state, dtype=np.float32),
                            "selected": np.array(sorted(selected_nodes), dtype=np.int32)
                        }

                # ------------------------------
                # Standard DOAR / DDPG action
                # ------------------------------
                with torch.no_grad():
                    st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    raw       = ddpg['actor'](st_tensor).cpu().numpy()[0]

                noise = np.random.normal(0.0, noise_std, size=raw.shape)
                vec   = np.clip(raw + noise, -1.0, +1.0)
                noise_std = max(σ_min, noise_std * decay_rate)

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
                            critic=ddpg['critic'],
                            exploit_override=exploit_override,
                        )

                # ------------------------------
                # Env step
                # ------------------------------
                _, raw_reward, reward, done, *_ = env_copy.step(action)
                next_state = my_state_fn() if role == turn else other_state_fn()

                # ------------------------------
                # Meta-controller logging/training
                # ------------------------------
                if meta_controller is not None:
                    # Use raw_reward so meta learns w.r.t. game payoff
                    meta_controller.store_transition(raw_reward, next_state, done)

                    # Train meta-controller every N steps (tune N as needed)
                    if (t + 1) % 10 == 0:
                        meta_controller.train_controller(iterations=1)

                # ------------------------------
                # DDPG replay + update
                # ------------------------------
                if self.BR_type != "Cord_asc":
                    ddpg['replay_buffer'].push(state, vec, reward, next_state, done)
                else:
                    disc_vec = self.encode_action(action, n_types, D, E, A)
                    ddpg['replay_buffer'].push(state, disc_vec, reward, next_state, done)

                train_ddpg(ddpg['actor'], ddpg['critic'],
                        ddpg['target_actor'], ddpg['target_critic'],
                        ddpg['replay_buffer'],
                        ddpg['actor_optimizer'], ddpg['critic_optimizer'],
                        batch_size=512, gamma=0.99, device=self.device)

                total_reward += raw_reward
                state = next_state

                if getattr(env_copy, "time_budget_exceeded", False):
                    print("Time budget exceeded — ending training early")
                    break

                if done:
                    break

            else:
                # ------------------------------
                # Opponent turn
                # ------------------------------
                idx   = np.random.choice(len(opponent_strategies), p=opponent_equilibrium)
                strat = opponent_strategies[idx]
                # Use unified action selection (works for baseline/typed/parametric)
                action = self._strategy_decide_action(env_copy, strat, turn, t, _opp_model_cache)

                _, _, done, *_ = env_copy.step(action)
                state = my_state_fn() if role != turn else other_state_fn()

                if getattr(env_copy, "time_budget_exceeded", False):
                    print("Time budget exceeded — ending training early")
                    break

                if done:
                    break

        actor_dict  = clone_state_dict(ddpg['actor'].state_dict())
        critic_dict = clone_state_dict(ddpg['critic'].state_dict())
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


    # ---------- payoff sparsification helpers ----------
    def estimate_payoff_thresholds(self,
                                   D_mat: np.ndarray,
                                   A_mat: np.ndarray,
                                   sample_pairs: int = 100,
                                   low_pct: float = 33.0,
                                   high_pct: float = 66.0):
        """
        Sample up to `sample_pairs` (i,j) pairs (favoring not-yet-computed pairs)
        and return percentile-based thresholds for defender and attacker payoffs:
            (def_low, def_high, att_low, att_high)

        NOTE: sampling runs serial simulate_game(...) with allow_parallel=False to avoid
        multiprocessing/CUDA re-init problems.
        """
        n_def = len(self.defender_strategies)
        n_att = len(self.attacker_strategies)
        max_pairs = n_def * n_att
        if max_pairs == 0:
            return (0.0, 0.0, 0.0, 0.0)

        sample_pairs = min(sample_pairs, max_pairs)
        rng = np.random.default_rng(self.seed + 12345)

        # Favor uncomputed pairs
        candidates = [(i, j) for i in range(n_def) for j in range(n_att) if (i, j) not in self._computed_pairs]
        if len(candidates) < sample_pairs:
            # add already-computed pairs if needed
            more = [(i, j) for i in range(n_def) for j in range(n_att) if (i, j) in self._computed_pairs]
            candidates.extend(more)
        # trim to desired size deterministically/randomly
        if len(candidates) > sample_pairs:
            idxs = rng.choice(len(candidates), size=sample_pairs, replace=False)
            candidates = [candidates[k] for k in idxs]

        def_vals = []
        att_vals = []
        # Run sampled games serially (safe)
        #with torch.inference_mode():
        for (i, j) in candidates:
            key = (i, j)
            if key in self._payoff_cache:
                d_pay, a_pay = float(self._payoff_cache[key][0]), float(self._payoff_cache[key][1])
            else:
                try:
                    d_pay, a_pay, *_ = self.simulate_game(self.defender_strategies[i],
                                                            self.attacker_strategies[j],
                                                            num_simulations=max(1, min(self.N_MC, 4)),
                                                            allow_parallel=True)
                except Exception:
                    # on any failure, fallback to 0 scores
                    d_pay, a_pay = 0.0, 0.0
            def_vals.append(float(d_pay))
            att_vals.append(float(a_pay))

        if not def_vals: def_vals = [0.0]
        if not att_vals: att_vals = [0.0]

        def_low, def_high = np.percentile(def_vals, [low_pct, high_pct])
        att_low, att_high = np.percentile(att_vals, [low_pct, high_pct])

        # small safety
        if def_low >= def_high:
            def_low = def_high - 1e-6
        if att_low >= att_high:
            att_low = att_high - 1e-6

        return float(def_low), float(def_high), float(att_low), float(att_high)


    def sparsify_payoff_matrices(self,
                                 D_mat: np.ndarray,
                                 A_mat: np.ndarray,
                                 sample_pairs: int = 200,
                                 low_pct: float = 33.0,
                                 high_pct: float = 66.0,
                                 per_role: bool = True):
        """
        Convert continuous payoff matrices into ternary {-1,0,1} using sampled thresholds.

        Returns (D_sparse:int8, A_sparse:int8, thresholds_dict)
        """
        D = D_mat.copy()
        A = A_mat.copy()

        try:
            if per_role:
                def_low, def_high, att_low, att_high = self.estimate_payoff_thresholds(
                    D, A, sample_pairs=sample_pairs, low_pct=low_pct, high_pct=high_pct)
            else:
                combined = np.concatenate([D.flatten(), A.flatten()]) if (D.size + A.size) > 0 else np.array([0.0])
                l, h = np.percentile(combined, [low_pct, high_pct])
                def_low = att_low = float(l)
                def_high = att_high = float(h)
        except Exception:
            # If anything fails, fallback to simple min/median/max heuristic
            flat_def = D.flatten() if D.size else np.array([0.0])
            flat_att = A.flatten() if A.size else np.array([0.0])
            def_low, def_high = np.percentile(flat_def, [low_pct, high_pct])
            att_low, att_high = np.percentile(flat_att, [low_pct, high_pct])

        # mapping functions
        def map_def(v):
            if np.isnan(v):
                return 0
            if v < def_low:
                return -1
            if v > def_high:
                return 1
            return 0

        def map_att(v):
            if np.isnan(v):
                return 0
            if v < att_low:
                return -1
            if v > att_high:
                return 1
            return 0

        vmap_def = np.vectorize(map_def)
        vmap_att = np.vectorize(map_att)

        D_sparse = vmap_def(D).astype(np.int8)
        A_sparse = vmap_att(A).astype(np.int8)

        thresholds = {
            'def_low': def_low, 'def_high': def_high,
            'att_low': att_low, 'att_high': att_high
        }
        return D_sparse, A_sparse, thresholds

    def _cheap_single_rollout(self,def_strat: Strategy, att_strat: Strategy, steps: int) -> Tuple[float, float]:
                #print("checkpoint 8.1")
                env = self.fresh_env()
                #print("checkpoint 8.2")
                # If available, randomize per-rollout small things to match other rollouts
                if hasattr(env, "randomize_compromise_and_ownership"):
                    env.randomize_compromise_and_ownership()
                #print("checkpoint 8.3")

                # reset only key counters (keep heavy simulator internals intact)
                for attr in ["step_num", "defender_step", "attacker_step", "work_done",
                            "checkpoint_count", "defensive_cost", "clearing_cost",
                            "revert_count", "scan_cnt", "compromised_devices_cnt",
                            "edges_blocked", "edges_added"]:
                    if hasattr(env, attr):
                        setattr(env, attr, 0)

                def_total = att_total = 0.0
                info = {}
                # single episode, no zero-day importance-weighting
                #print("Starting rollout")
                
                for t in range(steps):
                    #print("ROllout step"+str(t))
                    
                    turn = "defender" if (t % 2 == 0) else "attacker"
                    #print("ROllout step"+str(t)+".1")
                    env.mode = turn
                    #print("ROllout step"+str(t)+".2")
                    strat = def_strat if turn == "defender" else att_strat
                    #print("ROllout step"+str(t)+".3")
                    action = self._strategy_decide_action(env, strat, turn, t, {})  # empty model cache
                    #print("ROllout step"+str(t)+".4")
                    #print(action)
                    _, r, _, done, info, _ = env.step(action)
                    #print("ROllout step"+str(t)+".5")
                    if turn == "defender":
                        def_total += r
                        #print("ROllout step"+str(t)+".6")
                    else:
                        att_total += r
                        #print("ROllout step"+str(t)+".6")
                    if done:
                        #print("ROllout step"+str(t)+".7")
                        break
                #print("ROllout step"+str(t)+".8")
                return float(def_total), float(att_total)
    
    def quantize(self, v, low, high):
        if v <= low: return -1.0
        if v >= high: return 1.0
        return 0.0

    # ---------- build_payoff_matrices (sparsify optional argument) ----------
    def build_payoff_matrices(self,
                              n_workers: int | None = None,
                              sparsify: bool | None = None,
                              sparsify_min_cached_ratio: float = 0.02,
                              sparsify_sample_pairs: int = 200,
                              sparsify_low_pct: float = 33.0,
                              sparsify_high_pct: float = 66.0,
                              short_rollout_steps: int = 8):
        """
        Build payoff matrices with optional sparsification.

        - sparsify: True/False/None. If None, autodetect based on env size (>5000).
        - sparsify_min_cached_ratio: fraction of exact pairs required to use pure-proxy path.
        - sparsify_sample_pairs: cap for sample-pairs used to estimate quantile thresholds.
        - short_rollout_steps: when falling back to quick exact sims, use this many steps per bootstrap sim.
        """
        # --- autodetect if not specified ---
        if sparsify is None:
            sparsify = bool(getattr(self.env, "numOfDevice", 0) > 5000)
        #print("Check point 1")
        # Helper: a very cheap single in-process rollout for bootstrapping thresholds.
        # This avoids simulate_game()'s zero-day loops, parallelization, and other overhead.
        

        with timing("DO.build_payoff_matrices(cached)"):
            self._ensure_matrix_sizes()

            n_def = len(self.defender_strategies)
            n_att = len(self.attacker_strategies)
            total_pairs = max(1, n_def * n_att)

            # Fill from exact cached results
            for i in range(n_def):
                for j in range(n_att):
                    key = (i, j)
                    if key in self._computed_pairs:
                        d_pay, a_pay, *_ = self._payoff_cache[key]
                        self.D_mat[i, j] = d_pay
                        self.A_mat[j, i] = a_pay
            #print("Check point 2")
            # Identify missing pairs grouped by row
            rows_to_eval = []
            for i in range(n_def):
                missing_js = [j for j in range(n_att) if (i, j) not in self._computed_pairs]
                if missing_js:
                    rows_to_eval.append((i, missing_js))

            if not rows_to_eval:
                return self.D_mat.copy(), self.A_mat.copy()
            #print("Check point 3")
            # If sparsify is False -> use original exact behavior (serial or parallel)
            if not sparsify:
                if n_workers is None or n_workers <= 1:
                    with torch.inference_mode():
                        for i, js in rows_to_eval:
                            env = self.fresh_env()
                            d_strat = self.defender_strategies[i]
                            for j in js:
                                key = (i, j)
                                if key in self._computed_pairs:
                                    continue
                                self.restore(env, reset_counters=True)
                                with timing(f"simulate_game[{i},{j}]"):
                                    d_pay, a_pay, *rest = self.simulate_game(d_strat, self.attacker_strategies[j], self.N_MC, allow_parallel=False)
                                self._payoff_cache[key] = (d_pay, a_pay, *rest)
                                self._computed_pairs.add(key)
                                self.D_mat[i, j] = d_pay
                                self.A_mat[j, i] = a_pay
                    return self.D_mat.copy(), self.A_mat.copy()
                #print("Check point 4")
                # parallel exact path: unchanged semantics (uses worker pickle etc.)
                ctx = _get_mp_ctx() if "_get_mp_ctx" in globals() else mp.get_context("spawn")
                max_procs = mp.cpu_count()
                n_workers_eff = max(1, min(n_workers, max_procs, len(rows_to_eval)))
                do_bytes = self._worker_pickle()
                job_args = [(i, js, do_bytes) for (i, js) in rows_to_eval]
                with ctx.Pool(processes=n_workers_eff, initializer=_mp_worker_init) as pool:
                    chunksize = max(1, len(job_args) // (n_workers_eff * 4))
                    for row_i, js, row_def, row_att in pool.imap_unordered(_eval_row_worker, job_args, chunksize=chunksize):
                        for val_d, val_a, j in zip(row_def, row_att, js):
                            key = (row_i, j)
                            if key in self._computed_pairs:
                                continue
                            self._payoff_cache[key] = (val_d, val_a)
                            self._computed_pairs.add(key)
                            self.D_mat[row_i, j] = val_d
                            self.A_mat[j, row_i] = val_a
                return self.D_mat.copy(), self.A_mat.copy()
            #print("Check point 5")
            # -------------------- SPARSIFY PATH --------------------
            n_computed = len(self._computed_pairs)
            computed_ratio = float(n_computed) / float(total_pairs)

            # If we already have a reasonable fraction of exact pairs -> pure proxy averaging
            if computed_ratio >= sparsify_min_cached_ratio:
                # build per-def / per-att means from cached exact entries
                def_sums = np.zeros(n_def, dtype=float); def_counts = np.zeros(n_def, dtype=int)
                att_sums = np.zeros(n_att, dtype=float); att_counts = np.zeros(n_att, dtype=int)
                existing_vals_def = []

                for (i, j) in self._computed_pairs:
                    d_pay = float(self._payoff_cache[(i, j)][0])
                    a_pay = float(self._payoff_cache[(i, j)][1])
                    def_sums[i] += d_pay; def_counts[i] += 1
                    att_sums[j] += a_pay; att_counts[j] += 1
                    existing_vals_def.append(d_pay)

                overall_mean = float(np.mean(existing_vals_def)) if existing_vals_def else 0.0
                def_mean = np.array([ (def_sums[i]/def_counts[i]) if def_counts[i] > 0 else overall_mean for i in range(n_def) ])
                att_mean = np.array([ (att_sums[j]/att_counts[j]) if att_counts[j] > 0 else overall_mean for j in range(n_att) ])

                # estimate for missing pairs using additive proxy
                missing_pairs = []
                est_def_vals = []; est_att_vals = []
                for i, js in rows_to_eval:
                    for j in js:
                        missing_pairs.append((i, j))
                        est_d = def_mean[i] + att_mean[j] - overall_mean
                        est_a = att_mean[j] + def_mean[i] - overall_mean
                        est_def_vals.append(est_d); est_att_vals.append(est_a)

                # choose thresholds using existing exact distribution if possible, otherwise estimated
                def_vals_for_thresholds = np.array(existing_vals_def) if len(existing_vals_def) >= 5 else np.array(est_def_vals)
                att_vals_for_thresholds = np.array([float(self._payoff_cache[k][1]) for k in self._computed_pairs]) if len(self._computed_pairs) >= 5 else np.array(est_att_vals)

                if def_vals_for_thresholds.size < 3 or att_vals_for_thresholds.size < 3:
                    # not enough data — do a tiny number of cheap short rollouts to bootstrap
                    rng = random.Random(self.seed if hasattr(self, "seed") else None)
                    sample_needed = min(6, len(missing_pairs))
                    if sample_needed == 0:
                        # nothing to sample -> just return current (partial) matrix
                        return self.D_mat.copy(), self.A_mat.copy()
                    sampled_pairs = rng.sample(missing_pairs, sample_needed)

                    # very small bootstrap using the cheap single-rollout helper
                    sample_def_vals = []; sample_att_vals = []
                    for (i, j) in sampled_pairs:
                        d_pay, a_pay = self._cheap_single_rollout(self.defender_strategies[i], self.attacker_strategies[j], short_rollout_steps)
                        sample_def_vals.append(d_pay); sample_att_vals.append(a_pay)

                    # augment estimated lists for threshold computation
                    est_def_vals.extend(sample_def_vals); est_att_vals.extend(sample_att_vals)
                    def_vals_for_thresholds = np.array(est_def_vals)
                    att_vals_for_thresholds = np.array(est_att_vals)
                #print("Check point 6")
                # percentiles for quantization into {-1,0,1}
                low_def, high_def = np.percentile(def_vals_for_thresholds, [sparsify_low_pct, sparsify_high_pct])
                low_att, high_att = np.percentile(att_vals_for_thresholds, [sparsify_low_pct, sparsify_high_pct])



                # fill in missing pairs with quantized proxy (do NOT write into exact caches)
                idx = 0
                for (i, j) in missing_pairs:
                    qd = self.quantize(est_def_vals[idx], low_def, high_def)
                    qa = self.quantize(est_att_vals[idx], low_att, high_att)
                    self.D_mat[i, j] = float(qd)
                    self.A_mat[j, i] = float(qa)
                    idx += 1

                return self.D_mat.copy(), self.A_mat.copy()
            #print("Check point 7")
            # ELSE: computed_ratio < min_cached -> do tiny bootstrap sample and proxy off that sample
            rng = random.Random(self.seed if hasattr(self, "seed") else None)
            missing_pairs_all = [(i, j) for i, js in rows_to_eval for j in js]
            # severely cap the bootstrap sample for speed
            sample_count = min( max(4, int(total_pairs // 200) ), min(int(sparsify_sample_pairs), len(missing_pairs_all)) )
            if sample_count <= 0:
                return self.D_mat.copy(), self.A_mat.copy()
            sample_pairs = rng.sample(missing_pairs_all, sample_count)

            # run short cheap rollouts for the bootstrap sample
            sample_def_vals = []; sample_att_vals = []
            #print("Check point 8")
            for (i, j) in sample_pairs:
                d_pay, a_pay = self._cheap_single_rollout(self.defender_strategies[i], self.attacker_strategies[j], short_rollout_steps)
                sample_def_vals.append(d_pay); sample_att_vals.append(a_pay)

            if len(sample_def_vals) < 2:
                return self.D_mat.copy(), self.A_mat.copy()
            #print("Check point 9")
            low_def, high_def = np.percentile(np.array(sample_def_vals), [sparsify_low_pct, sparsify_high_pct])
            low_att, high_att = np.percentile(np.array(sample_att_vals), [sparsify_low_pct, sparsify_high_pct])

            # build per-index means from the bootstrap sample
            def_sums = np.zeros(n_def, dtype=float); def_counts = np.zeros(n_def, dtype=int)
            att_sums = np.zeros(n_att, dtype=float); att_counts = np.zeros(n_att, dtype=int)
            for k, (i, j) in enumerate(sample_pairs):
                def_sums[i] += sample_def_vals[k]; def_counts[i] += 1
                att_sums[j] += sample_att_vals[k]; att_counts[j] += 1
            overall_mean = float(np.mean(sample_def_vals)) if sample_def_vals else 0.0
            def_mean = np.array([ (def_sums[i]/def_counts[i]) if def_counts[i] > 0 else overall_mean for i in range(n_def) ])
            att_mean = np.array([ (att_sums[j]/att_counts[j]) if att_counts[j] > 0 else overall_mean for j in range(n_att) ])

            #print("Check point 10")

            # fill missing using additive proxy + quantize
            for i, js in rows_to_eval:
                for j in js:
                    est_d = def_mean[i] + att_mean[j] - overall_mean
                    est_a = att_mean[j] + def_mean[i] - overall_mean
                    self.D_mat[i, j] = float(self.quantize(est_d, low_def, high_def))
                    self.A_mat[j, i] = float(self.quantize(est_a, low_att, high_att))
            #print("Check point 11")
            return self.D_mat.copy(), self.A_mat.copy()



    # ---------- PARALLEL simulate_game ----------
    def simulate_game(self,
                      defender_strategy: Strategy,
                      attacker_strategy: Strategy,
                      num_simulations: int = 1,
                      allow_parallel: bool = True):
        """
        Parallelized over independent rollouts (process-per-episode) when
        num_simulations > 1, self.parallel_rollouts is True, and allow_parallel is True.

        Returns averaged 10-tuple:
          (avg_def, avg_att, avg_comp_frac, avg_jobs, avg_scan,
           avg_def_cost, avg_ckpt, avg_revert, avg_edges_blocked, avg_edges_added)
        """
        # Precompute zero-day sampling distribution
        if getattr(self.env, "zero_day", False):
            pool_ids = list(getattr(self.env, "private_exploit_ids", [])) or [None]
            probs = [self.env.prior_pi.get(z, 1.0) if z is not None else 1.0 for z in pool_ids]
            s = float(sum(probs))
            probs = [p/s for p in probs] if s > 0 else [1.0/len(pool_ids)]*len(pool_ids)
        else:
            pool_ids = [None]; probs = [1.0]

        can_parallel = (allow_parallel and self.parallel_rollouts and num_simulations > 1)

        # If not parallel: run in-process (fast path)
        if not can_parallel:
            return self._simulate_game_serial(defender_strategy, attacker_strategy, num_simulations, pool_ids, probs)

        # Parallel path using a lean, pickled DO for each worker
        ctx = _get_mp_ctx()
        n_workers = max(1, min(self.rollout_workers, mp.cpu_count()))
        d_payload = defender_strategy.to_payload()
        a_payload = attacker_strategy.to_payload()

        # Monte Carlo over z with importance weights carried back
        rng = np.random.default_rng(self.seed + 777)
        zdraws = rng.choice(pool_ids, size=num_simulations, p=probs)

        # Build worker tasks: (do_bytes, d_pl, a_pl, T, zdraw, weight_z)
        do_bytes = self._worker_pickle()
        tasks = []
        for k in range(num_simulations):
            z = zdraws[k].item() if hasattr(zdraws[k], "item") else zdraws[k]
            w = probs[pool_ids.index(z)] if z in pool_ids else 1.0
            tasks.append((do_bytes, d_payload, a_payload, int(self.steps_per_episode), z, float(w)))

        # Aggregate weighted sums
        tot_def = tot_att = 0.0
        tot_comp = tot_jobs = tot_scan = tot_defcost = 0.0
        tot_ckpt = tot_rev = tot_block = tot_add = 0.0
        tot_steps = 0.0
        tot_w = 0.0

        with ctx.Pool(processes=n_workers, initializer=_mp_worker_init) as pool:
            chunksize = max(1, num_simulations // (n_workers * 4))
            for dsum, asum, side, w in pool.imap_unordered(_sim_rollout_worker, tasks, chunksize=chunksize):
                tot_def   += dsum * w
                tot_att   += asum * w
                tot_comp  += side.get("Compromised_devices", 0.0) * w
                tot_jobs  += side.get("work_done",           0.0) * w
                tot_scan  += side.get("Scan_count",          0.0) * w
                tot_defcost += side.get("defensive_cost",    0.0) * w
                tot_ckpt  += side.get("checkpoint_count",    0.0) * w
                tot_rev   += side.get("revert_count",        0.0) * w
                tot_block += side.get("Edges Blocked",       0.0) * w
                tot_add   += side.get("Edges Added",         0.0) * w
                tot_steps += side.get("steps", self.steps_per_episode) * w
                tot_w     += w

        if tot_w <= 0:
            tot_w = 1.0

        avg_def  = tot_def / tot_w
        avg_att  = tot_att / tot_w
        avg_comp_frac = (tot_comp / tot_w) / max(1.0, (tot_steps / tot_w))

        return (
            float(avg_def), float(avg_att), float(avg_comp_frac),
            float(tot_jobs   / tot_w), float(tot_scan / tot_w), float(tot_defcost / tot_w),
            float(tot_ckpt   / tot_w), float(tot_rev  / tot_w), float(tot_block  / tot_w), float(tot_add / tot_w)
        )

    def _simulate_game_serial(self,
                              defender_strategy: Strategy,
                              attacker_strategy: Strategy,
                              num_simulations: int,
                              pool_ids: List[Optional[int]],
                              probs: List[float]):
        """Original in-process implementation with .execute() removed (uses unified action selector)."""
        # Per-call cache for parametric models
        _model_cache: dict[int, tuple[torch.nn.Module, torch.nn.Module]] = {}

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
        last_steps           = 0.0

        if getattr(self.env, "zero_day", False):
            # importance-weighted loop over draws
            for idx_z, zdraw in enumerate(pool_ids):
                weight_z = probs[idx_z]
                for _ in range(num_simulations):
                    env = self.fresh_env()
                    if hasattr(env, "randomize_compromise_and_ownership"):
                        env.randomize_compromise_and_ownership()
                    for attr in [
                        "step_num", "defender_step", "attacker_step",
                        "work_done", "checkpoint_count", "defensive_cost",
                        "clearing_cost" if False else "clearning_cost", "revert_count", "scan_cnt",
                        "compromised_devices_cnt", "edges_blocked", "edges_added"
                    ]:
                        if hasattr(env, attr):
                            setattr(env, attr, 0)
                    env.private_exploit_id = zdraw

                    phase1_def = phase2_def = 0.0
                    phase1_att = phase2_att = 0.0
                    discovered = False
                    info = {}

                    with torch.inference_mode():
                        for t in range(self.steps_per_episode):
                            turn = 'defender' if (t % 2 == 0) else 'attacker'
                            env.mode = turn
                            strat = defender_strategy if turn == 'defender' else attacker_strategy

                            action = self._strategy_decide_action(env, strat, turn, t, _model_cache)

                            _, r, _, done, info, _ = env.step(action)

                            if info.get('discovered_private', False) and turn == 'defender':
                                discovered = True
                            if turn == 'defender':
                                (phase2_def if discovered else phase1_def).__iadd__(r)
                            else:
                                (phase2_att if discovered else phase1_att).__iadd__(r)

                            if done:
                                break

                    total_def += weight_z * phase1_def + phase2_def
                    total_att += weight_z * phase1_att + phase2_att

                    total_compromised    += info.get("Compromised_devices", 0.0) * weight_z
                    total_jobs_completed += info.get("work_done",           0.0) * weight_z
                    total_scan_cnt       += info.get("Scan_count",          0.0) * weight_z
                    total_defensive_cost += info.get("defensive_cost",      0.0) * weight_z
                    total_checkpoint_cnt += info.get("checkpoint_count",    0.0) * weight_z
                    total_revert_cnt     += info.get("revert_count",        0.0) * weight_z
                    total_edges_blocked  += info.get("Edges Blocked",       0.0) * weight_z
                    total_edges_added    += info.get("Edges Added",         0.0) * weight_z
                    last_steps = getattr(env, "step_num", self.steps_per_episode)
        else:
            for _ in range(num_simulations):
                env = self.fresh_env()
                if hasattr(env, "randomize_compromise_and_ownership"):
                    env.randomize_compromise_and_ownership()
                for attr in [
                    "step_num", "defender_step", "attacker_step",
                    "work_done", "checkpoint_count",
                    "defensive_cost", "clearing_cost" if False else "clearning_cost",
                    "revert_count", "scan_cnt"
                ]:
                    if hasattr(env, attr):
                        setattr(env, attr, 0)

                def_r = att_r = 0.0
                info = {}

                with torch.inference_mode():
                    for t in range(self.steps_per_episode):
                        turn = 'defender' if (t % 2 == 0) else 'attacker'
                        env.mode = turn
                        strat = defender_strategy if turn == 'defender' else attacker_strategy

                        action = self._strategy_decide_action(env, strat, turn, t, _model_cache)

                        _, r, _, done, info, _ = env.step(action)
                        if turn == 'defender': def_r += r
                        else:                  att_r += r
                        if done: break

                total_def += def_r; total_att += att_r
                total_compromised    += info.get("Compromised_devices", 0.0)
                total_jobs_completed += info.get("work_done",           0.0)
                total_scan_cnt       += info.get("Scan_count",          0.0)
                total_defensive_cost += info.get("defensive_cost",      0.0)
                total_checkpoint_cnt += info.get("checkpoint_count",    0.0)
                total_revert_cnt     += info.get("revert_count",        0.0)
                total_edges_blocked  += info.get("Edges Blocked",       0.0)
                total_edges_added    += info.get("Edges Added",         0.0)
                last_steps = getattr(env, "step_num", self.steps_per_episode)

        N = float(num_simulations)
        avg_compromised_fraction = (total_compromised / N) / max(1.0, last_steps)

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

    # ---------- update matrix with new strategy ----------
    def update_payoff_matrix(self, payoff_matrix, new_strategy, role):
        self._ensure_matrix_sizes()
        if role == 'defender':
            i = len(self.defender_strategies) - 1
            for j in range(len(self.attacker_strategies)):
                key = (i, j)
                if key in self._computed_pairs: continue
                with timing(f"simulate_game[new D row {i},{j}]"):
                    res = self.simulate_game(self.defender_strategies[i], self.attacker_strategies[j], num_simulations=self.N_MC, allow_parallel=False)
                self._payoff_cache[key] = res
                self._computed_pairs.add(key)
                d_pay, a_pay, *_ = res
                self.D_mat[i, j] = d_pay
                self.A_mat[j, i] = a_pay

        elif role == 'attacker':
            j = len(self.attacker_strategies) - 1
            for i in range(len(self.defender_strategies)):
                key = (i, j)
                if key in self._computed_pairs: continue
                with timing(f"simulate_game[new A col {i},{j}]"):
                    res = self.simulate_game(self.defender_strategies[i], self.attacker_strategies[j], num_simulations=self.N_MC, allow_parallel=False)
                self._payoff_cache[key] = res
                self._computed_pairs.add(key)
                d_pay, a_pay, *_ = res
                self.D_mat[i, j] = d_pay
                self.A_mat[j, i] = a_pay

        return self.D_mat.copy()

    def get_payoff(self, defender_strategy, attacker_strategy):
        i = self.defender_strategies.index(defender_strategy)
        j = self.attacker_strategies.index(attacker_strategy)
        key = (i, j)
        if key not in self._computed_pairs:
            with timing(f"simulate_game[get_payoff {i},{j}]"):
                res = self.simulate_game(defender_strategy, attacker_strategy, num_simulations=self.N_MC, allow_parallel=False)
            self._payoff_cache[key] = res
            self._computed_pairs.add(key)
            d_pay, a_pay, *_ = res
            self.D_mat[i, j] = d_pay
            self.A_mat[j, i] = a_pay
        return self._payoff_cache[key][0]  # defender payoff

    # ---------- greedy device coord-ascent ----------
    def greedy_device_coord_ascent(self,
                                   n_types: int,
                                   D: int,
                                   E: int,
                                   A: int,
                                   state_tensor: torch.Tensor,
                                   raw_action:   np.ndarray,
                                   actor:        nn.Module,
                                   critic:       nn.Module,
                                   exploit_override: Optional[int] = None):
        if exploit_override is not None:
            action_type = int(np.argmax(raw_action[:n_types])) if n_types>0 else 0
            return (action_type, np.array([exploit_override], dtype=int), np.array([], dtype=int), 0)

        no_op_type = n_types - 1
        no_op = (no_op_type, np.array([], dtype=int), np.array([0], dtype=int), 0)

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

        Q_base   = Q_of(no_op)[0]
        is_train = critic.training

        best_map = {d: no_op for d in range(D)}
        for d in range(D):
            raw_list = [
                (atype, np.array([d], dtype=int), np.array([e_idx], dtype=int), 0)
                for atype in range(n_types) for e_idx in range(E)
            ]
            cand = [(no_op, Q_base)]
            if raw_list:
                qv = Q_of(raw_list)
                if is_train:
                    qv = qv + self.coord_noise_std * np.random.randn(*qv.shape)
                cand += list(zip(raw_list, qv.tolist()))
            cand.sort(key=lambda x: x[1], reverse=True)
            topk = cand[: self.coord_K]
            qs   = np.array([q for _, q in topk], dtype=np.float64)
            exp_q = np.nan_to_num(np.exp(qs / self.coord_tau))
            probs = (exp_q / exp_q.sum()) if exp_q.sum() > 0 else np.ones_like(exp_q) / len(exp_q)
            choice = np.random.choice(len(topk), p=probs)
            best_map[d] = topk[choice][0]

        final_atype = no_op_type
        devs, exps = [], []
        if self.merge_rule_atype == "best_q":
            # Paper-correct merge (Algorithm 1 erratum):
            # Pick action type from the device whose sampled a_d has the highest Qϕ(s, a_d).
            # This change is order-independent and matches the text:
            #     t* = argmax_d Qϕ(s, a_d)
            scored = []
            for d, (at, ds, es, p) in best_map.items():
                qd = Q_of((at, ds, es, p))[0]
                scored.append((qd, at, ds, es, p))
                if at != no_op_type:
                    devs.extend(ds.tolist()); exps.extend(es.tolist())

            non_noop = [row for row in scored if row[1] != no_op_type]
            final_atype = (max(non_noop, key=lambda r: r[0])[1] if non_noop else no_op_type)

        else:
            # Historical merge (used in all published results; kept for reproducibility):
            # "last non-noop wins" — order-dependent but empirically adequate vs fixed opponents.
            # See paper erratum: Algorithm 1 now states t* = argmax_d Qϕ(s, a_d).
            for (at, ds, es, _) in best_map.values():
                if at != no_op_type:
                    final_atype = at
                    devs.extend(ds.tolist())
                    exps.extend(es.tolist())
        non_noop = [row for row in scored if row[1] != no_op_type]
        final_atype = (max(non_noop, key=lambda r: r[0])[1] if non_noop else no_op_type)

        final_devs = np.array(devs, dtype=int) if devs else np.array([], dtype=int)
        final_exps = np.array([0], dtype=int) if not exps else np.array([exps[0]], dtype=int)
        return (final_atype, final_exps, final_devs, 0)

    # ---------- testing helper ----------
    def test_fixed_player(self, fixed_role: str, steps_per_episode: int, test_runs: int):
        if not self.restore(self.env, reset_counters=True):
            raise RuntimeError("No in-memory checkpoint set. Call do.checkpoint_now() first.")

        D = self.D_init; E = self.E_init; A = self.A_init
        def_types = self.env.get_num_action_types(mode="defender")
        att_types = self.env.get_num_action_types(mode="attacker")

        results = []
        for _ in range(test_runs):
            self.restore(self.env, reset_counters=True)
            self.env.tech = "DO"; self.env.mode = "defender"

            if fixed_role == "defender":
                probs, strat_pool = self.defender_equilibrium, self.defender_strategies
            else:
                probs, strat_pool = self.attacker_equilibrium, self.attacker_strategies

            idx_fixed   = np.random.choice(len(strat_pool), p=probs)
            fixed_strat = strat_pool[idx_fixed]

            _cache: dict[int, tuple[nn.Module, nn.Module]] = {}

            def_rews, att_rews = [], []
            for step in range(steps_per_episode):
                turn = "defender" if (step % 2 == 0) else "attacker"
                if turn == fixed_role:
                    action = self._strategy_decide_action(self.env, fixed_strat, turn, step, _cache)
                else:
                    var_strat = self.defender_strategies[-1] if turn == "defender" else self.attacker_strategies[-1]
                    action = self._strategy_decide_action(self.env, var_strat, turn, step, _cache)

                _, r, _, done, _, _ = self.env.step(action)
                if turn == "defender": def_rews.append(r)
                else:                  att_rews.append(r)
                if done: break

            self.env.base_line = "Nash"
            results.append((def_rews, att_rews))
        return results
