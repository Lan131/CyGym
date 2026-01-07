# HMARL.py
# Baseline hierarchical BRs mirroring H-MARL Expert and H-MARL Meta (paper baselines)
# for comparison against MetaDoar / MetaHierarchicalBestResponse.
#
# This version adds cost- and locality-structured experts and REAL TRAINING:
#   • Optional PPO fine-tuning for each sub-policy (skill head)
#   • PPO for the master policy (which selects among sub-policies)
#
# Defender skills (example mapping used by runners):
#   Skill 0 = CheapLocal      (atype {1,5,6,7,9,11})
#   Skill 1 = CostlyLocal     (atype {4,12,13})
#   Skill 2 = Global          (atype {2,3,8,10})
#
# Returned action is always a grouped list compatible with env.step_grouped():
#   [(atype, exploit_indices_np, device_indices_np, app_idx_int), ...]
#
# Notes:
# - policy_net output is assumed to be logits over that expert's allowed atypes.
# - If policy_net is None or not usable, we fall back to heuristic/random choice.
# - We budget per-device actions to avoid absurdly large batches.

from __future__ import annotations
import copy
import os
import random
import tempfile
from typing import List, Tuple, Optional, Dict, Any, Deque
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from strategy import Strategy  # assumed same Strategy class used by DO/PSRO


# ===============================
# On-policy PPO buffer (GAE-λ)
# ===============================
class PPOBuffer:
    """On-policy buffer with GAE-λ for discrete actions (skill indices)."""
    def __init__(self, obs_dim, steps, gamma=0.99, lam=0.95, device=None):
        self.device = device or torch.device("cpu")
        self.gamma, self.lam = gamma, lam
        self.obs = torch.zeros((steps, obs_dim), dtype=torch.float32, device=self.device)
        self.act = torch.zeros((steps,), dtype=torch.int64, device=self.device)
        self.logp = torch.zeros((steps,), dtype=torch.float32, device=self.device)
        self.rew = torch.zeros((steps,), dtype=torch.float32, device=self.device)
        self.val = torch.zeros((steps,), dtype=torch.float32, device=self.device)
        self.adv = torch.zeros((steps,), dtype=torch.float32, device=self.device)
        self.ret = torch.zeros((steps,), dtype=torch.float32, device=self.device)
        self.ptr = 0

    def store(self, s, a, r, logp, v):
        i = self.ptr
        if i >= self.obs.size(0):
            return
        self.obs[i]  = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.act[i]  = int(a)
        self.rew[i]  = float(r)
        self.logp[i] = float(logp)
        self.val[i]  = float(v)
        self.ptr += 1

    def finish(self, last_value=0.0):
        # Compute GAE advantages and returns.
        T = self.ptr
        vals = torch.cat([self.val[:T], torch.tensor([last_value], device=self.device)])
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            delta = self.rew[t] + self.gamma * vals[t+1] - vals[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        self.adv[:T] = adv
        self.ret[:T] = adv + self.val[:T]
        # Normalize advantages
        if T > 1:
            m = self.adv[:T].mean()
            s = self.adv[:T].std(unbiased=False) + 1e-8
            self.adv[:T] = (self.adv[:T] - m) / s

    def dataset(self, batch_size):
        T = self.ptr
        if T == 0:
            return
        idx = torch.randperm(T, device=self.device)
        for i in range(0, T, batch_size):
            j = min(i + batch_size, T)
            sel = idx[i:j]
            yield (self.obs[sel], self.act[sel], self.logp[sel], self.adv[sel], self.ret[sel])


# ===========================================
# Utility helpers for costs/targets/batching
# ===========================================
DEFENDER_PER_DEVICE_COST_EST = {
    1: lambda dev: 0.3 if getattr(dev, "isCompromised", False) else 0.01,  # clean
    4: lambda dev: 1.0,   # patch bump
    5: lambda dev: 0.5,   # scan
    6: lambda dev: 0.5,   # block edges
    7: lambda dev: 0.5,   # remove device from network
    9: lambda dev: 0.5,   # reopen edge
    11: lambda dev: 0.1,  # checkpoint single device mem
    12: lambda dev: 1.0,  # restore single device mem
    13: lambda dev: 3.0,  # hard isolate / aggressive scrub
}
DEFENDER_GLOBAL_COST_EST = {  # not used directly—global ops are batched as single tuples
    2: 0.5,  # checkpoint whole net
    3: 1.0,  # revert whole net
    8: 0.0,  # noop
    10: 1.0, # train detector
}
DEFENDER_PER_DEVICE_ATYPES = {1,4,5,6,7,9,11,12,13}
DEFENDER_GLOBAL_ATYPES     = {2,3,8,10}

# Example mapping (used by your runner when constructing subpolicies)
DEFENDER_SKILL_MAP = {
    0: [1,5,6,7,9,11],  # CheapLocal
    1: [4,12,13],       # CostlyLocal
    2: [2,3,8,10],      # Global
}

def _current_valid_device_ids(env) -> List[int]:
    try:
        return list(env.simulator.subnet.net.keys())
    except Exception:
        return []

def _visible_devices(env):
    try:
        return [d for d in env.simulator.subnet.net.values()
                if not getattr(d, "Not_yet_added", False)]
    except Exception:
        return []

def _high_value_targets(env):
    devs = _visible_devices(env)
    def score(dev):
        is_dc = (getattr(dev, "device_type", None) == "DomainController")
        compromised = bool(getattr(dev, "isCompromised", False))
        attacker_owned = bool(getattr(dev, "attacker_owned", False))
        if compromised and is_dc and not attacker_owned:
            return 100
        if compromised and not attacker_owned:
            return 50
        if compromised and attacker_owned:
            return 40
        if getattr(dev, "reachable_by_attacker", False):
            return 20
        return 0
    return sorted(devs, key=score, reverse=True)

def _random_targets(env):
    devs = _visible_devices(env)
    random.shuffle(devs)
    return devs

def _estimate_cost_for_device(atype: int, dev) -> float:
    fn = DEFENDER_PER_DEVICE_COST_EST.get(atype, None)
    if fn is None:
        return 0.0
    try:
        return float(fn(dev))
    except Exception:
        return 0.0

def _batch_devices_by_cost(atype: int, dev_list: List, budget: float) -> List[List[int]]:
    if not dev_list:
        return [[]] if atype in DEFENDER_GLOBAL_ATYPES else []
    if atype not in DEFENDER_PER_DEVICE_ATYPES:
        return [[]]
    batches: List[List[int]] = []
    cur_batch: List[int] = []
    cur_cost = 0.0
    for dev in dev_list:
        dcost = _estimate_cost_for_device(atype, dev)
        if cur_batch and (cur_cost + dcost) > budget:
            batches.append(cur_batch)
            cur_batch, cur_cost = [], 0.0
        cur_batch.append(dev.id)
        cur_cost += dcost
    if cur_batch:
        batches.append(cur_batch)
    return batches


# ==============================
# Frozen subpolicies (skills)
# ==============================
class FrozenSubPolicy:
    """
    Adapter for a (pre-)trained sub-policy ψ_c ("expert skill").
    Restricted to a fixed set of allowed action_types.
    Returns grouped tuples compatible with env.step_grouped():
        [(atype, exploit_indices_np, device_indices_np, app_idx_int), ...]
    """

    def __init__(self,
                 policy_net: Optional[nn.Module],
                 device: torch.device,
                 name: str,
                 role: str,
                 allowed_action_types: List[int],
                 per_group_cost_budget: float = 3.0):
        self.policy_net = policy_net
        if self.policy_net is not None:
            self.policy_net = self.policy_net.to(device)
            self.policy_net.eval()
        self.device = device
        self.name = name
        self.role = role  # "defender" or "attacker"
        self.allowed_action_types = list(allowed_action_types)
        self.cost_budget = float(per_group_cost_budget)

    def state_dict(self):
        if self.policy_net is None:
            return {}
        return self.policy_net.state_dict()

    def load_state_dict(self, sd):
        if self.policy_net is None:
            return
        self.policy_net.load_state_dict(sd)
        self.policy_net.eval()

    @torch.no_grad()
    def _pick_action_type(self, state_vec: np.ndarray) -> int:
        # Greedy argmax over policy_net logits if available, else random.
        if self.policy_net is None:
            return random.choice(self.allowed_action_types)
        st = torch.tensor(np.asarray(state_vec, np.float32), device=self.device).unsqueeze(0)
        try:
            logits = self.policy_net(st).squeeze(0)
            if logits.dim() == 0:
                return random.choice(self.allowed_action_types)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            idx = int(np.argmax(probs))
            idx = max(0, min(idx, len(self.allowed_action_types) - 1))
            return self.allowed_action_types[idx]
        except Exception:
            return random.choice(self.allowed_action_types)

    def _choose_devices_for_action(self, atype: int, env) -> List:
        # Defender heuristics
        if self.role == "defender":
            if atype in (4,12,13):  # costly local / escalation
                return _high_value_targets(env)
            if atype in (1,5,6,7,9,11):  # cheap local
                prioritized = _high_value_targets(env)
                if not prioritized:
                    prioritized = _random_targets(env)
                return prioritized
            if atype in (2,3):  # checkpoint / revert -> can conceptually touch all
                return _visible_devices(env)
            if atype in (8,10):  # noop / train detector
                return []
        # Attacker placeholder heuristics (simple):
        if self.role == "attacker":
            devs = _visible_devices(env)
            if atype == 1:  # lateral spread
                seeds = [d for d in devs if getattr(d, "attacker_owned", False) or getattr(d, "isCompromised", False)]
                if not seeds: seeds = devs
                random.shuffle(seeds)
                return seeds
            if atype == 2:  # probe
                seeds = [d for d in devs if getattr(d, "isCompromised", False)]
                if not seeds: seeds = devs
                random.shuffle(seeds)
                return seeds
            return []
        return []

    def _batchify(self, atype: int, dev_objs: List, env) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
        # Exploit indices: only meaningful for attacker; defender sends [0]
        exploit_idx_arr = np.array([0], dtype=int)

        grouped: List[Tuple[int, np.ndarray, np.ndarray, int]] = []

        # Global-like defender actions or attacker-wide
        if atype in DEFENDER_GLOBAL_ATYPES or (self.role == "attacker" and atype in (2, 3)):
            batches = _batch_devices_by_cost(atype=atype, dev_list=dev_objs, budget=self.cost_budget)
            valid_ids = set(_current_valid_device_ids(env))
            for batch_ids in batches:
                legal_batch_ids = [d for d in batch_ids if d in valid_ids]
                MAX_FANOUT = 5
                if len(legal_batch_ids) > MAX_FANOUT:
                    legal_batch_ids = legal_batch_ids[:MAX_FANOUT]
                didxs = np.array(legal_batch_ids, dtype=int)
                grouped.append((atype, exploit_idx_arr, didxs, 0))
            if not grouped:
                grouped = [(atype, exploit_idx_arr, np.array([], dtype=int), 0)]
            return grouped

        # Per-device style actions (defender locals, attacker local)
        batches = _batch_devices_by_cost(atype=atype, dev_list=dev_objs, budget=self.cost_budget)
        valid_ids = set(_current_valid_device_ids(env))
        for batch_ids in batches:
            legal_batch_ids = [d for d in batch_ids if d in valid_ids]
            if not legal_batch_ids:
                continue
            MAX_FANOUT = 5
            if len(legal_batch_ids) > MAX_FANOUT:
                legal_batch_ids = legal_batch_ids[:MAX_FANOUT]
            didxs = np.array(legal_batch_ids, dtype=int)
            grouped.append((atype, exploit_idx_arr, didxs, 0))
        if not grouped:
            # if nothing legal, send a safe no-op style
            fallback_atype = 8 if self.role == "defender" else 3
            grouped = [(fallback_atype, exploit_idx_arr, np.array([], dtype=int), 0)]
        return grouped

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, env=None):
        # 1) choose an allowed atype
        atype = self._pick_action_type(state_vec)
        # 2) choose target devices in priority order
        dev_objs = self._choose_devices_for_action(atype, env)
        # 3) convert to grouped tuples with cost-based batching
        return self._batchify(atype, dev_objs, env)


# ==========================
# Master policies
# ==========================
class ExpertRuleMaster:
    """Deterministic rule-based master."""
    def __init__(self, cheaplocal_idx: int, costlylocal_idx: int, global_idx: int, global_prob: float = 0.1):
        self.cheaplocal_idx = int(cheaplocal_idx)
        self.costlylocal_idx = int(costlylocal_idx)
        self.global_idx = int(global_idx)
        self.global_prob = float(global_prob)

    def _count_compromised(self, env):
        total = 0
        dc_flag = False
        for d in env.simulator.subnet.net.values():
            if getattr(d, "isCompromised", False) and not getattr(d, "attacker_owned", False):
                total += 1
                if getattr(d, "device_type", None) == "DomainController":
                    dc_flag = True
        return total, dc_flag

    def select_skill_index(self, state_vec: np.ndarray, env) -> int:
        compromised_cnt, dc_flag = self._count_compromised(env)
        if dc_flag:
            return self.costlylocal_idx
        if compromised_cnt >= 3:
            return self.cheaplocal_idx
        if random.random() < self.global_prob:
            return self.global_idx
        return self.cheaplocal_idx

    def get_skill_config(self) -> Dict[str, Any]:
        return {
            "cheaplocal_idx": self.cheaplocal_idx,
            "costlylocal_idx": self.costlylocal_idx,
            "global_idx": self.global_idx,
            "global_prob": self.global_prob,
        }

class LearnedMasterPolicy(nn.Module):
    """PPO actor-critic for selecting which subpolicy/skill to invoke."""
    def __init__(self, state_dim: int, num_skills: int, hidden: int = 128):
        super().__init__()
        self.pi_fc1 = nn.Linear(state_dim, hidden)
        self.pi_fc2 = nn.Linear(hidden, num_skills)
        self.v_fc1  = nn.Linear(state_dim, hidden)
        self.v_fc2  = nn.Linear(hidden, 1)

    def pi(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pi_fc1(s))
        return self.pi_fc2(x)  # logits

    def v(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.v_fc1(s))
        return self.v_fc2(x).squeeze(-1)  # (B,)

    @torch.no_grad()
    def select_skill_index(self, state_vec: np.ndarray, device: torch.device) -> Tuple[int, float, float]:
        s = torch.tensor(np.asarray(state_vec, np.float32), device=device).unsqueeze(0)
        logits = self.pi(s)
        dist   = torch.distributions.Categorical(logits=logits)
        a      = int(dist.sample().item())
        logp   = float(dist.log_prob(torch.tensor([a], device=device)).item())
        v      = float(self.v(s).item())
        return a, logp, v


# ===================================
# Optional PPO for sub-policies
# ===================================
class SubPolicyPPO:
    """Lightweight PPO fine-tuning for a subpolicy's policy_net (if it exists)."""
    def __init__(self, subpolicy: FrozenSubPolicy, obs_dim: int, hidden=64, lr=3e-4,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                 epochs=3, mini_batch=64, device=None):
        self.sp = subpolicy
        self.device = device or torch.device("cpu")
        self.gamma, self.lam = gamma, lam
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.epochs, self.mini_batch = epochs, mini_batch

        # If the subpolicy already has a small policy_net, reuse it; else make one.
        if self.sp.policy_net is None:
            self.sp.policy_net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, len(self.sp.allowed_action_types))
            ).to(self.device)
        else:
            self.sp.policy_net = self.sp.policy_net.to(self.device)

        # Value head for this subpolicy PPO
        self.v_head = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        ).to(self.device)

        self.opt = optim.Adam(list(self.sp.policy_net.parameters()) + list(self.v_head.parameters()), lr=lr)

    def _pi(self, s):
        logits = self.sp.policy_net(s)
        return torch.distributions.Categorical(logits=logits)

    def update(self, buf: PPOBuffer):
        if buf.ptr == 0:
            return
        buf.finish(last_value=0.0)
        for _ in range(self.epochs):
            for s, a, old_logp, adv, ret in buf.dataset(self.mini_batch):
                pi = self._pi(s)
                logp = pi.log_prob(a)
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                loss_pi = -(torch.min(ratio * adv, clipped)).mean()
                ent = pi.entropy().mean()

                v = self.v_head(s).squeeze(-1)
                loss_v = F.mse_loss(v, ret)

                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.sp.policy_net.parameters()) + list(self.v_head.parameters()), 1.0)
                self.opt.step()


# ===================================
# Base hierarchical BR wrapper
# ===================================
class BaseHMARLBR:
    """
    Shared scaffolding for HMARLExpertBestResponse and HMARLMetaBestResponse.
    Presents a MetaDoar-like interface to Double Oracle.
    """
    def __init__(self,
                 oracle,
                 role: str,
                 subpolicies: List[FrozenSubPolicy],
                 master,
                 device: Optional[torch.device] = None):
        self.oracle = oracle
        self.role = role  # "defender" or "attacker"
        self.env = copy.deepcopy(oracle.env)  # private env

        # Clamp env.numOfDevice to live nodes after deepcopy (prevents sampler issues)
        try:
            live_ids = [d_id for d_id, dev in self.env.simulator.subnet.net.items()
                        if not getattr(dev, "Not_yet_added", False)]
            live_count = len(live_ids)
            if hasattr(self.env, "numOfDevice") and live_count > 0:
                if self.env.numOfDevice > live_count:
                    self.env.numOfDevice = live_count
            elif hasattr(self.env, "numOfDevice") and live_count == 0:
                self.env.numOfDevice = 1
        except Exception:
            pass

        self.device = device or getattr(oracle, "device", torch.device("cpu"))
        self.subpolicies = subpolicies
        self.master = master

        self._last_decision = None  # track last (state,skill) for replay

        if role == "defender":
            self.state_dim = self.env._get_defender_state().shape[0]
        else:
            self.state_dim = self.env._get_attacker_state().shape[0]

    def fresh_env(self):
        try:
            return self.oracle.env_factory()  # if provided by caller
        except Exception:
            return copy.deepcopy(self.oracle.env)

    def _env_state_for_role(self, env):
        return env._get_defender_state() if self.role == "defender" else env._get_attacker_state()

    # ---- NEW: normalize any step return to (obs, rew, done, info) ----
    def _coerce_step_ret(self, ret):
        try:
            if isinstance(ret, (tuple, list)):
                if len(ret) == 4:
                    obs, rew, done, info = ret
                    return obs, float(rew), bool(done), info
                if len(ret) == 5:
                    obs, rew, terminated, truncated, info = ret
                    done = bool(terminated) or bool(truncated)
                    return obs, float(rew), done, info
        except Exception:
            pass
        # fallback safe shape
        try:
            obs = self._env_state_for_role(self.env)
        except Exception:
            obs = None
        return obs, 0.0, False, {}

    def _step_env_grouped(self, env, grouped_actions):
        """
        Execute grouped actions robustly and coerce return:
          • Ensure each tuple is (atype:int, exploit_idx:np.int[], device_idx:np.int[], app_idx:int)
          • Drop invalid device ids (not in env)
          • Prefer env.step_grouped; fall back to env.step on first tuple
          • Always return (obs, rew, done, info); never raise
        """
        # -------- normalize & sanitize --------
        if not grouped_actions:
            grouped_actions = [
                (8 if self.role == "defender" else 3,
                 np.array([0], dtype=int),
                 np.array([], dtype=int),
                 0)
            ]

        try:
            valid_ids = set(env.simulator.subnet.net.keys())
        except Exception:
            valid_ids = set()

        normalized = []
        for tup in grouped_actions:
            try:
                atype, exp_idx, dev_idx, app_idx = tup
            except Exception:
                atype, exp_idx, dev_idx, app_idx = (
                    8 if self.role == "defender" else 3,
                    np.array([0], dtype=int),
                    np.array([], dtype=int),
                    0,
                )
            exp_idx = np.asarray(exp_idx, dtype=int)
            try:
                dev_list = np.asarray(dev_idx).tolist()
            except Exception:
                dev_list = []
            dev_list = [int(d) for d in dev_list if d in valid_ids]
            # modest fanout safety
            if len(dev_list) > 64:
                dev_list = dev_list[:64]
            dev_idx = np.asarray(dev_list, dtype=int)
            normalized.append((int(atype), exp_idx, dev_idx, int(app_idx)))

        # -------- try grouped API first --------
        if hasattr(env, "step_grouped"):
            try:
                ret = env.step_grouped(normalized)
                return self._coerce_step_ret(ret)
            except Exception:
                pass  # fall through to single-step fallback

        # -------- fallback: single step on first tuple --------
        atype, exp_idx, dev_idx, app_idx = normalized[0]
        try:
            ret = env.step((atype, exp_idx, dev_idx, app_idx))
            return self._coerce_step_ret(ret)
        except Exception:
            # final safety: strict no-op single step
            fallback = (
                8 if self.role == "defender" else 3,
                np.array([0], dtype=int),
                np.array([], dtype=int),
                0,
            )
            try:
                ret = env.step(fallback)
                return self._coerce_step_ret(ret)
            except Exception:
                # if even that fails, synthesize a safe tuple
                obs = self._env_state_for_role(env)
                return obs, 0.0, False, {}

    def execute(self, strat: Strategy, state: np.ndarray, env=None, **kwargs):
        """
        Produce grouped actions:
          1) Load weights from strat if present (eval).
          2) Master picks skill index.
          3) Run subpolicy[skill] to get grouped low-level actions.
        """
        env = env or self.env
        self._load_from_strategy_mapping(strat)
        skill_idx = self._select_skill(state, env)
        grouped_actions = self.subpolicies[skill_idx].select_action(state, env=env)
        self._last_decision = {"state": np.asarray(state, np.float32), "skill": int(skill_idx)}
        return grouped_actions

    def store_transition(self, reward: float, next_state: Optional[np.ndarray] = None, done: bool = False):
        self._last_decision = None  # overridden in meta version if using replay

    def _select_skill(self, state_vec: np.ndarray, env) -> int:
        raise NotImplementedError

    def _load_from_strategy_mapping(self, strat: Strategy):
        pass

    def save(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _strategy_key(self) -> str:
        raise NotImplementedError

    # Fallback (kept for interface compatibility if a caller ever invokes Base.train)
    def train(self,
              opponent_strategies: List[Strategy],
              opponent_equilibrium: List[float],
              T: int = 15000,
              σ: float = 1.0,
              σ_min: float = 1e-5,
              return_meta: bool = False,
              **kwargs) -> Strategy:
        if return_meta:
            fd, path = tempfile.mkstemp(prefix=f"hmarl_{self.role}_", suffix=".pth")
            os.close(fd)
            payload = self.save(path)
            payload["path"] = path
            return Strategy(type_mapping={self._strategy_key(): payload})
        try:
            new_strat = self.oracle.ddpg_best_response(
                opponent_strategies, opponent_equilibrium, self.role,
                training_steps=int(T), σ=σ, σ_min=σ_min
            )
            return new_strat
        except Exception:
            D = getattr(self.oracle, "D_init", getattr(self.oracle, "Max_network_size", 1))
            acts = [(0, np.array([], dtype=int), np.array([np.random.randint(0, D)], dtype=int), 0)
                    for _ in range(getattr(self.oracle, "steps_per_episode", 50))]
            return Strategy(actions=acts)


# =========================================
# HMARL Expert Best Response (rule-based)
# =========================================
class HMARLExpertBestResponse(BaseHMARLBR):
    def __init__(self, oracle, role: str, subpolicies: List[FrozenSubPolicy],
                 expert_master: ExpertRuleMaster, device: Optional[torch.device] = None):
        super().__init__(oracle, role, subpolicies, expert_master, device=device)

    def _select_skill(self, state_vec: np.ndarray, env) -> int:
        return self.master.select_skill_index(state_vec, env)

    def _load_from_strategy_mapping(self, strat: Strategy):
        payload = getattr(strat, "type_mapping", {}).get("hmarl_expert")
        if not payload:
            return
        # restore frozen subpolicies
        sub_sd_list = payload.get("subpolicies", [])
        for sd, subp in zip(sub_sd_list, self.subpolicies):
            try:
                subp.load_state_dict(sd)
            except Exception:
                pass
        # master config
        try:
            cfg = payload.get("master_cfg", {})
            self.master.cheaplocal_idx = int(cfg.get("cheaplocal_idx", self.master.cheaplocal_idx))
            self.master.costlylocal_idx = int(cfg.get("costlylocal_idx", self.master.costlylocal_idx))
            self.master.global_idx = int(cfg.get("global_idx", self.master.global_idx))
            self.master.global_prob = float(cfg.get("global_prob", self.master.global_prob))
        except Exception:
            pass

    def save(self, path: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "master_type": "expert_rule",
            "master_cfg": self.master.get_skill_config(),
            "subpolicies": [sp.state_dict() for sp in self.subpolicies],
        }
        try:
            torch.save(payload, path)
        except Exception:
            pass
        return payload

    def _strategy_key(self) -> str:
        return "hmarl_expert"


# =========================================
# HMARL Meta Best Response (PPO training)
# =========================================
class HMARLMetaBestResponse(BaseHMARLBR):
    """
    Paper-faithful H-MARL Meta:
      - (optional) PPO for subpolicies (skills) on their sub-tasks
      - PPO for master over skills with environment reward
    """
    def __init__(self,
                 oracle,
                 role: str,
                 subpolicies: List[FrozenSubPolicy],
                 state_dim: int,
                 device: Optional[torch.device] = None,
                 hidden: int = 128,
                 master_lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 epochs: int = 4,
                 mini_batch: int = 128,
                 rollout_steps: int = 1024,
                 subpolicy_ft_iters: int = 0,         # >0 enables subpolicy PPO pretraining
                 subpolicy_rollout_steps: int = 512,
                 subpolicy_epochs: int = 3,
                 subpolicy_mini_batch: int = 64):
        master_net = LearnedMasterPolicy(state_dim, num_skills=len(subpolicies), hidden=hidden)
        super().__init__(oracle, role, subpolicies, master_net, device=device)

        self.master = master_net.to(self.device)
        self.master.eval()

        # PPO hyperparams
        self.gamma, self.lam = gamma, lam
        self.clip, self.ent_coef, self.vf_coef = clip, ent_coef, vf_coef
        self.epochs, self.mini_batch = epochs, mini_batch
        self.rollout_steps = rollout_steps
        self.master_opt = optim.Adam(self.master.parameters(), lr=master_lr)

        # Optional subpolicy fine-tuning
        self.subpolicy_trainers = [
            SubPolicyPPO(sp, obs_dim=self.state_dim, hidden=64, lr=3e-4,
                         gamma=gamma, lam=lam, clip=0.2, ent_coef=0.01, vf_coef=0.5,
                         epochs=subpolicy_epochs, mini_batch=subpolicy_mini_batch,
                         device=self.device)
            for sp in self.subpolicies
        ]
        self.subpolicy_ft_iters = max(0, int(subpolicy_ft_iters))
        self.subpolicy_rollout_steps = subpolicy_rollout_steps

    # ---- selection used during evaluation path ----
    def _select_skill(self, state_vec: np.ndarray, env) -> int:
        a, _, _ = self.master.select_skill_index(state_vec, self.device)
        return a

    # ---- master PPO components ----
    def _master_act(self, s_t: torch.Tensor):
        logits = self.master.pi(s_t)
        dist   = torch.distributions.Categorical(logits=logits)
        a      = dist.sample()
        logp   = dist.log_prob(a)
        v      = self.master.v(s_t)
        return a, logp, v

    def _master_update(self, buf: PPOBuffer):
        if buf.ptr == 0:
            return
        buf.finish(last_value=0.0)
        for _ in range(self.epochs):
            for s, a, old_logp, adv, ret in buf.dataset(self.mini_batch):
                logits = self.master.pi(s)
                dist   = torch.distributions.Categorical(logits=logits)
                logp   = dist.log_prob(a)
                ratio  = torch.exp(logp - old_logp)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
                loss_pi = -(torch.min(surr1, surr2)).mean()
                ent     = dist.entropy().mean()

                v_pred = self.master.v(s)
                loss_v = F.mse_loss(v_pred, ret)

                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * ent
                self.master_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.master.parameters(), 1.0)
                self.master_opt.step()

    # ---- rollout helpers ----
    def _phase1_train_subpolicies(self, opponent_strategies, opponent_equilibrium, T):
        if self.subpolicy_ft_iters <= 0:
            return
        env = copy.deepcopy(self.env)
        try:
            env.initialize_environment()
        except Exception:
            pass
        for _ in range(self.subpolicy_ft_iters):
            for sp_tr in self.subpolicy_trainers:
                # skip if heuristic-only
                if sp_tr.sp.policy_net is None:
                    continue
                buf = PPOBuffer(self.state_dim, self.subpolicy_rollout_steps, gamma=sp_tr.gamma, lam=sp_tr.lam, device=self.device)
                s = self._env_state_for_role(env)
                for _t in range(self.subpolicy_rollout_steps):
                    # sample atype from the subpolicy policy_net directly for PPO
                    s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                    dist = torch.distributions.Categorical(logits=sp_tr.sp.policy_net(s_t))
                    a_idx = int(dist.sample().item())
                    a_idx = max(0, min(a_idx, len(sp_tr.sp.allowed_action_types) - 1))
                    atype = sp_tr.sp.allowed_action_types[a_idx]
                    logp = float(dist.log_prob(torch.tensor([a_idx], device=self.device)).item())
                    v = float(sp_tr.v_head(s_t).squeeze(0).item())

                    grouped = sp_tr.sp._batchify(atype, sp_tr.sp._choose_devices_for_action(atype, env), env)
                    if not grouped:
                        grouped = [(8 if self.role == "defender" else 3, np.array([0], dtype=int), np.array([], dtype=int), 0)]
                    obs, rew, done, info = self._step_env_grouped(env, grouped)
                    s2 = self._env_state_for_role(env)

                    buf.store(s, a_idx, rew, torch.tensor(logp, device=self.device), v)
                    s = s2
                    if done:
                        try:
                            env.reset()
                        except Exception:
                            pass
                        s = self._env_state_for_role(env)
                sp_tr.update(buf)

    def _phase2_train_master(self, opponent_strategies, opponent_equilibrium, T):
        env = copy.deepcopy(self.env)
        try:
            env.initialize_environment()
        except Exception:
            pass
        steps_remaining = int(T)
        while steps_remaining > 0:
            steps = min(self.rollout_steps, steps_remaining)
            buf = PPOBuffer(self.state_dim, steps, gamma=self.gamma, lam=self.lam, device=self.device)
            s = self._env_state_for_role(env)
            for _ in range(steps):
                if getattr(env, "time_budget_exceeded", False):
                    print("HMARL time budget exceeded — ending training early")
                    break
                s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                a_t, logp_t, v_t = self._master_act(s_t)
                a = int(a_t.item()); logp = float(logp_t.item()); v = float(v_t.item())

                grouped = self.subpolicies[a].select_action(s, env=env)
                if not grouped:
                    grouped = [(8 if self.role == "defender" else 3, np.array([0], dtype=int), np.array([], dtype=int), 0)]

                obs, rew, done, info = self._step_env_grouped(env, grouped)
                s2 = self._env_state_for_role(env)

                buf.store(s, a, rew, torch.tensor(logp, device=self.device), v)
                s = s2
                if done:
                    try:
                        env.reset()
                    except Exception:
                        pass
                    s = self._env_state_for_role(env)

            with torch.no_grad():
                last_v = self.master.v(torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
            buf.finish(last_value=last_v)
            self._master_update(buf)
            steps_remaining -= steps

    # ---- public training API ----
    def train(self,
              opponent_strategies: List[Strategy],
              opponent_equilibrium: List[float],
              T: int = 15000,
              return_meta: bool = False,
              **kwargs) -> Strategy:
        # Phase 1: (optional) PPO subpolicy fine-tuning
        self._phase1_train_subpolicies(opponent_strategies, opponent_equilibrium, T)
        # Phase 2: PPO master over skills
        self._phase2_train_master(opponent_strategies, opponent_equilibrium, T)

        if return_meta:
            fd, path = tempfile.mkstemp(prefix=f"hmarl_meta_{self.role}_", suffix=".pth")
            os.close(fd)
            payload = self.save(path)
            payload["path"] = path
            return Strategy(type_mapping={self._strategy_key(): payload})

        # If not asked for a snapshot, still return a Strategy handle with payload
        return Strategy(type_mapping={self._strategy_key(): self.save("<inmem>")})

    # ---- (de)serialization ----
    def _load_from_strategy_mapping(self, strat: Strategy):
        payload = getattr(strat, "type_mapping", {}).get("hmarl_meta")
        if not payload:
            return
        path = payload.get("path", None)
        ckpt = None
        if path and os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
            except Exception:
                ckpt = None
        if ckpt is None:
            ckpt = payload
        try:
            self.master.load_state_dict(ckpt["master_state_dict"])
            self.master.eval()
        except Exception:
            pass
        sub_sd_list = ckpt.get("subpolicies", [])
        for sd, subp in zip(sub_sd_list, self.subpolicies):
            try:
                subp.load_state_dict(sd)
            except Exception:
                pass

    def save(self, path: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "master_type": "learned_meta_ppo",
            "master_state_dict": self.master.state_dict(),
            "subpolicies": [sp.state_dict() for sp in self.subpolicies],
            "state_dim": self.state_dim,
            "num_skills": len(self.subpolicies),
        }
        try:
            torch.save(payload, path)
        except Exception:
            pass
        return payload

    def _strategy_key(self) -> str:
        return "hmarl_meta"
