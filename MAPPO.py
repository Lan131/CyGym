# MAPPO.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Reuse Strategy to plug back into your Double-Oracle pools
from do_agent import Strategy, Actor, Critic

# =============================
# Config (tune as needed)
# =============================
USE_GAT: bool = False
USE_AMP: bool = False  # keep off until stable

# Env action ids (must match your env)
DEFENDER_NOOP = 8
ATTACKER_NOOP = 3
SINGLE_DEVICE_TYPES = {11, 12}

# Stability knobs
POLICY_LR        = 1e-4
REWARD_SCALE     = 1e-1
ADV_NORM_MIN_N   = 8          # only normalize advantages if batch >= 8
ADV_CLIP         = 1e4
RET_CLIP         = 1e4
CLIP_LOGP_DIFF   = 20.0
VALUE_CLIP_EPS   = 0.2
VALUE_TARGET_CLIP= 1e4
ENT_COEF         = 1e-3
VF_COEF          = 0.5
MAX_GRAD_NORM    = 0.5

PROGRESS_EVERY_STEPS = 5000

# =============================
# Utilities
# =============================
def _to_device(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    return x.to(device)

def build_adjacency(env, D: int) -> torch.Tensor:
    """Best-effort adjacency from env; fallback: fully-connected with self-loops. Returns [1, D, D]."""
    try:
        G = getattr(env.simulator, "subnet", None)
        if G is None:
            raise AttributeError
        if hasattr(G, "edges") and callable(getattr(G, "edges")):
            edges = G.edges()
            adj = np.zeros((D, D), dtype=np.float32)
            for u, v in edges:
                if 0 <= u < D and 0 <= v < D:
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
            np.fill_diagonal(adj, 1.0)
        elif hasattr(G, "net"):
            adj = np.ones((D, D), dtype=np.float32)
        else:
            adj = np.ones((D, D), dtype=np.float32)
    except Exception:
        adj = np.ones((D, D), dtype=np.float32)
    return torch.tensor(adj, dtype=torch.float32).unsqueeze(0)

def build_visibility_mask(env, role: str) -> torch.Tensor:
    """
    Role-specific visibility mask v (length D, values in {0,1}).

    Attacker visible iff:
      Known_to_attacker AND attacker_owned AND NOT Not_yet_added

    Defender visible iff:
      (NOT Not_yet_added) AND attacker_owned
    """
    D = int(env.Max_network_size)
    v = torch.zeros(D, dtype=torch.float32)
    devices = env._get_ordered_devices()
    for i, d in enumerate(devices):
        if role == "attacker":
            visible = (getattr(d, "Known_to_attacker", False)
                       and getattr(d, "attacker_owned", False)
                       and not getattr(d, "Not_yet_added", False))
        else:  # defender
            visible = (not getattr(d, "Not_yet_added", False)
                       and getattr(d, "attacker_owned", False))
        v[i] = 1.0 if visible else 0.0
    return v  # (D,)

def masked_adjacency(adj: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    adj: [1, D, D], v: [D] in {0,1}
    Returns adj ⊙ (v vᵀ) with diagonal set to v (i.e., isolates invisible nodes).
    """
    vv = torch.ger(v, v).unsqueeze(0)  # [1,D,D]
    out = adj * vv
    # put back self-loops only for visible nodes
    B, D, _ = out.shape
    eye = torch.eye(D, device=out.device).unsqueeze(0) * v.view(1, -1, 1)
    out = out * (1 - eye) + eye
    return out

# =============================
# Graph Attention (optional)
# =============================
class GATLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, D, H = x.shape
        q = self.q(x); k = self.k(x); v = self.v(x)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(H)  # [B,D,D]
        scores = scores.masked_fill(adj <= 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = self.proj(torch.matmul(attn, v))
        return self.ln(x + out)

# =============================
# Actor-Critic (MAPPO, per-device actors)
# =============================
class CommActorCritic(nn.Module):
    """
    - Per-device *independent* actor: for each device d, a Categorical over action types (size n_types).
    - Global heads for exploit/app (pooled token).
    - Centralized critic on the pooled token.
    """
    def __init__(self, state_dim: int, n_types: int, D: int, E: int, A: int,
                 hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.state_dim, self.n_types, self.D, self.E, self.A = state_dim, n_types, D, E, A

        # Per-device token: concat(global proj, device embedding)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.id_emb = nn.Embedding(D, hidden)
        self.merge = nn.Linear(2 * hidden, hidden)

        self.gats = nn.ModuleList([GATLayer(hidden) for _ in range(n_layers)])

        # Heads
        self.dev_type_head = nn.Linear(hidden, n_types)      # [B, D, n_types]
        self.exp_head = nn.Linear(hidden, E)                 # [B, E]
        self.app_head = nn.Linear(hidden, A) if A > 0 else None

        # centralized critic
        self.v_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        # cache device ids
        self.register_buffer("_ids", torch.arange(D, dtype=torch.long), persistent=False)

    def _tokens(self, state: torch.Tensor) -> torch.Tensor:
        B = state.size(0)
        base = torch.relu(self.state_proj(state)).unsqueeze(1).repeat(1, self.D, 1)  # [B,D,H]
        id_tok = self.id_emb(self._ids.to(state.device)).unsqueeze(0).repeat(B, 1, 1)
        tok = torch.relu(self.merge(torch.cat([base, id_tok], dim=-1)))             # [B,D,H]
        return tok

    def forward(self, state: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        tok = self._tokens(state)  # [B,D,H]
        if USE_GAT:
            for gat in self.gats:
                tok = gat(tok, adj)

        ctx = tok.mean(dim=1)  # [B,H]

        per_dev_type_logits = self.dev_type_head(tok)            # [B, D, n_types]
        exp_logits          = self.exp_head(ctx)                 # [B, E]
        app_logits          = self.app_head(ctx) if self.A > 0 else None
        value               = self.v_head(ctx).squeeze(-1)       # [B]

        # sanitize numerics
        per_dev_type_logits = torch.nan_to_num(per_dev_type_logits, nan=0.0, posinf=0.0, neginf=0.0)
        exp_logits = torch.nan_to_num(exp_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if app_logits is not None:
            app_logits = torch.nan_to_num(app_logits, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "per_dev_type_logits": per_dev_type_logits,
            "exp_logits": exp_logits,
            "app_logits": app_logits,
            "value": value,
        }

# =============================
# Policy wrapper (execution)
# =============================
class MAPPOCommPolicy:
    """
    Greedy executor for trained MAPPO with per-device independent actors.
    Builds grouped actions and relies on env.step(list_of_groups) -> step_grouped.
    Visibility-aware: only acts on visible devices.
    """
    def __init__(self, oracle, role: str, state_dict: Dict[str, torch.Tensor]):
        assert role in ("defender", "attacker")
        self.oracle = oracle
        self.role = role
        self.device = oracle.device

        self.D = oracle.D_init
        self.E = oracle.E_init
        self.A = oracle.A_init
        self.n_types = oracle.n_def_types if role == "defender" else oracle.n_att_types

        trained_state_dim = (oracle.env._get_defender_state().shape[0]
                             if role == "defender" else oracle.env._get_attacker_state().shape[0])

        self.net = CommActorCritic(trained_state_dim, self.n_types, self.D, self.E, self.A)
        self.net.load_state_dict(state_dict)
        self.net.eval().to(self.device)

        self._expected_in = self.net.state_proj.in_features
        self._expected_D  = self.D

    def _ensure_sizes(self, state_vec: np.ndarray, env=None):
        s = np.asarray(state_vec, dtype=np.float32)
        if s.shape[0] != self._expected_in:
            raise RuntimeError(f"[MAPPO] State len {s.shape[0]} != expected {self._expected_in}.")
        if env is not None:
            D_env = int(getattr(env, "Max_network_size", self._expected_D))
            if D_env != self._expected_D:
                raise RuntimeError(f"[MAPPO] Device count {D_env} != policy {self._expected_D}.")

    @torch.inference_mode()
    def select_action(self, state_vec: np.ndarray, env=None):
        self._ensure_sizes(state_vec, env)
        D = self._expected_D
        s = _to_device(state_vec, self.device).unsqueeze(0)

        # visibility + masked adjacency (if used)
        if env is not None:
            v = build_visibility_mask(env, self.role).to(self.device)           # [D]
            adj = build_adjacency(env, D).to(self.device)
            adj = masked_adjacency(adj, v) if USE_GAT else adj
        else:
            v = torch.ones(D, device=self.device)
            adj = torch.ones((1, D, D), dtype=torch.float32, device=self.device)

        out = self.net(s, adj)
        pdt = out["per_dev_type_logits"].squeeze(0)   # [D, K]
        K = pdt.size(-1)

        # Greedy per device, but only for visible devices
        types = []
        for d in range(D):
            if v[d] < 0.5:
                types.append(0)  # in-range dummy; invisible devices won’t be grouped
            else:
                types.append(int(torch.argmax(pdt[d]).item()))

        # Global exploit/app choices (greedy)
        exp_idx = int(torch.argmax(out["exp_logits"], dim=-1).item()) if self.E > 0 else 0
        app_idx = int(torch.argmax(out["app_logits"], dim=-1).item()) if self.A > 0 and out["app_logits"] is not None else 0

        # Group devices by chosen type; enforce env constraints; only visible devices are considered
        noop = DEFENDER_NOOP if self.role == "defender" else ATTACKER_NOOP
        groups = []
        for t in range(K):
            if t == noop:
                continue
            devs = [d for d in range(D) if v[d] > 0.5 and types[d] == t]
            if not devs:
                continue
            if t in SINGLE_DEVICE_TYPES:
                devs = [random.choice(devs)]  # EXACTLY one visible device
            groups.append((t, np.array([exp_idx], dtype=int), np.array(devs, dtype=int), app_idx))

        if not groups:
            groups = [(noop, np.array([0], dtype=int), np.array([], dtype=int), 0)]

        return groups

# =============================
# PPO utilities
# =============================
@dataclass
class Step:
    state: np.ndarray
    logp: float
    value: float
    reward: float
    done: bool
    per_dev_types: np.ndarray  # [D] int
    exp: int
    app: int
    vis_mask: np.ndarray       # [D] in {0,1}

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns

# =============================
# MAPPO Best-Response Trainer
# =============================
class MAPPOCommBestResponse:
    """
    Best-response trainer with *independent per-device actors* + centralized critic.
    Visibility-aware: never samples/updates/acts on invisible devices.
    Produces grouped actions and uses env.step_grouped(groups).
    """
    def __init__(self, oracle, role: str):
        assert role in ("defender", "attacker")
        self.oracle = oracle
        self.role = role
        self.device = oracle.device
        self.seed = oracle.seed

        self.D = oracle.D_init
        self.E = oracle.E_init
        self.A = oracle.A_init
        self.n_types = oracle.n_def_types if role == "defender" else oracle.n_att_types
        self.state_dim = (oracle.env._get_defender_state().shape[0]
                          if role == "defender" else oracle.env._get_attacker_state().shape[0])

        torch.manual_seed(self.seed); random.seed(self.seed); np.random.seed(self.seed)

        self.net = CommActorCritic(self.state_dim, self.n_types, self.D, self.E, self.A).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=POLICY_LR)

        # PPO hyperparams
        self.clip_eps = 0.2
        self.ent_coef = ENT_COEF
        self.vf_coef  = VF_COEF
        self.max_grad_norm = MAX_GRAD_NORM

        self.amp_enabled = USE_AMP and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self._opp_cache: Dict[Tuple[int, str], Tuple[Optional[nn.Module], Optional[nn.Module]]] = {}

    # ---------- opponent helpers ----------
    def _try_parametric_action(self, env, turn: str, strat: Strategy):
        key = (id(strat), turn)
        if key in self._opp_cache:
            actor, critic = self._opp_cache[key]
        else:
            actor  = strat.load_actor(Actor,  seed=self.seed, device=self.device)
            critic = strat.load_critic(Critic, seed=self.seed, device=self.device)
            self._opp_cache[key] = (actor, critic)
        if actor is None:
            return False, None

        st = env._get_defender_state() if turn == "defender" else env._get_attacker_state()
        if len(st) != actor.fc1.in_features:
            return False, None

        s_tensor = torch.tensor(st, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw = actor(s_tensor).cpu().numpy()[0]

        n_types = self.oracle.n_def_types if turn == "defender" else self.oracle.n_att_types
        action = self.oracle.decode_action(
            raw,
            num_action_types=n_types,
            num_device_indices=self.D,
            num_exploit_indices=self.E,
            num_app_indices=self.A,
            state_tensor=s_tensor,
            actor=actor,
            critic=critic,
        )
        # Wrap into grouped format expected by our trainer (single tuple → one-group list)
        return True, [action]

    def _opponent_action(self, env, turn: str, opponent_strategies: List[Strategy], opponent_equilibrium: np.ndarray):
        N = len(opponent_strategies)
        indices = list(range(N))
        fallback_baseline_name = None
        fallback_fixed_action: Optional[List[Tuple]] = None

        for _ in range(max(N, 1)):
            idx = np.random.choice(indices, p=opponent_equilibrium)
            strat = opponent_strategies[idx]

            if strat.baseline_name is not None:
                env.base_line = strat.baseline_name
                return None  # env handles None with baseline

            if strat.actions is not None:
                t = env.step_num
                fixed = strat.actions[t % len(strat.actions)]
                return [fixed]  # single-group list

            if strat.type_mapping and ('mappo' in strat.type_mapping or 'marl' in strat.type_mapping):
                state_vec = env._get_defender_state() if turn == 'defender' else env._get_attacker_state()
                marl_agent = strat.type_mapping.get('mappo', strat.type_mapping.get('marl'))
                try:
                    groups = marl_agent.select_action(state_vec, env=env)
                    return groups if isinstance(groups, list) else [groups]
                except Exception:
                    continue

            ok, action = self._try_parametric_action(env, turn, strat)
            if ok:
                return action

            if strat.baseline_name is not None:
                fallback_baseline_name = strat.baseline_name

        if fallback_baseline_name is not None:
            env.base_line = fallback_baseline_name
            return None
        if fallback_fixed_action is not None:
            return fallback_fixed_action

        try:
            samp = env.sample_action()
            return [ (0, samp[1], samp[2], samp[3]) ]
        except Exception:
            return None

    # ---------- training ----------
    def train(self,
            opponent_strategies: List[Strategy],
            opponent_equilibrium: np.ndarray,
            T: int = 15_000,
            rollout_len: int = 1,
            ppo_epochs: int = 1,
            minibatch_size: int = 256,
            *,
            budget_type: str = "steps",     # "steps" or "updates"
            budget: Optional[int] = None,   # if None, defaults to T
            single_update_per_rollout: bool = False) -> Strategy:
        """
        Enforce a strict budget:
        - budget_type="steps": never exceed `budget` env steps collected (both players combined).
        - budget_type="updates": perform exactly `budget` optimizer updates
            (counting each mini-batch backward/step as one update).
        If `single_update_per_rollout=True`, we do 1 update per rollout, regardless of batch size.
        """
        assert budget_type in ("steps", "updates")
        if budget is None:
            budget = T
        if single_update_per_rollout:
            ppo_epochs = 1  # enforced

        env = self.oracle.fresh_env()
        if hasattr(env, "randomize_compromise_and_ownership"):
            env.randomize_compromise_and_ownership()
        for attr in ["step_num","defender_step","attacker_step","work_done",
                    "checkpoint_count","defensive_cost","clearning_cost",
                    "revert_count","scan_cnt","compromised_devices_cnt",
                    "edges_blocked","edges_added"]:
            if hasattr(env, attr): setattr(env, attr, 0)

        my_state_fn = env._get_defender_state if self.role == "defender" else env._get_attacker_state
        state = my_state_fn()
        total_reward = 0.0

        steps_done = 0           # how many env steps we actually called
        updates_done = 0         # how many optimizer updates we performed
        rollout_num = 0

        def steps_budget_remaining() -> int:
            return max(0, budget - steps_done)

        def updates_budget_remaining() -> int:
            return max(0, budget - updates_done)

        def budget_exhausted() -> bool:
            if budget_type == "steps":
                return steps_done >= budget
            else:
                return updates_done >= budget

        last_progress_print = -PROGRESS_EVERY_STEPS

        while not budget_exhausted():
            rollout_num += 1

            # We will rebuild adjacency with visibility each self-turn
            # (visibility can change as the episode evolves)
            if budget_type == "steps":
                max_collect = min(rollout_len, steps_budget_remaining())
                if max_collect <= 0:
                    break
            else:
                max_collect = rollout_len

            local_batch: List[Step] = []
            rollout_reward = 0.0

            while len(local_batch) == 0 and not budget_exhausted():
                if getattr(env, "time_budget_exceeded", False):
                    print("Time budget exceeded — ending training early")
                    break
                turn = "defender" if (env.step_num % 2 == 0) else "attacker"
                env.mode = turn

                if turn == self.role:
                    # visibility at this decision
                    v = build_visibility_mask(env, self.role).to(self.device)  # [D]
                    adj = build_adjacency(env, self.D).to(self.device)
                    adj = masked_adjacency(adj, v) if USE_GAT else adj

                    s_tensor = _to_device(state, self.device).unsqueeze(0)
                    with torch.no_grad():
                        out = self.net(s_tensor, adj)
                        pdt = out["per_dev_type_logits"]   # [1, D, K]
                        v_est= out["value"].squeeze(0)

                    D, K = pdt.size(1), pdt.size(2)
                    per_dev_dists = [Categorical(logits=pdt[:, d, :]) for d in range(D)]

                    per_dev_types: List[int] = []
                    logp_parts: List[torch.Tensor] = []
                    # sample ONLY for visible devices; invisible → forced NOOP and no logp contribution
                    noop = DEFENDER_NOOP if self.role == "defender" else ATTACKER_NOOP
                    for d, dist_d in enumerate(per_dev_dists):
                        if v[d] < 0.5:
                            # invisible → in-range dummy label (will be masked out later)
                            per_dev_types.append(0)
                            continue
                        t_samp = dist_d.sample().clamp(0, self.n_types - 1)
                        per_dev_types.append(int(t_samp.item()))
                        logp_parts.append(dist_d.log_prob(t_samp).squeeze(0))

                    # global heads unchanged
                    if self.E > 0:
                        exp_dist = Categorical(logits=out["exp_logits"])
                        exp_t = exp_dist.sample().clamp(0, exp_dist.logits.size(-1) - 1)
                        logp_parts.append(exp_dist.log_prob(exp_t).squeeze(0))
                        exp_idx = int(exp_t.item())
                    else:
                        exp_idx = 0

                    if self.A > 0 and out["app_logits"] is not None:
                        app_dist = Categorical(logits=out["app_logits"])
                        app_t = app_dist.sample().clamp(0, app_dist.logits.size(-1) - 1)
                        logp_parts.append(app_dist.log_prob(app_t).squeeze(0))
                        app_idx = int(app_t.item())
                    else:
                        app_idx = 0

                    logp = float(torch.stack(logp_parts).sum().item()) if logp_parts else 0.0
                    value = float(v_est.item())

                    # group per-device actions by type; enforce single-device constraints; only visible devices count
                    groups = []
                    for t in range(self.n_types):
                        if t == noop:
                            continue
                        devs = [i for i in range(D) if (v[i] > 0.5) and (per_dev_types[i] == t)]
                        if not devs:
                            continue
                        if t in SINGLE_DEVICE_TYPES:
                            devs = [random.choice(devs)]
                        groups.append((t, np.array([exp_idx], dtype=int), np.array(devs, dtype=int), app_idx))
                    if not groups:
                        groups = [(noop, np.array([0], dtype=int), np.array([], dtype=int), 0)]

                    # ---- env.step() (sanitize reward) ----
                    _, raw_reward, reward, done, *_ = env.step(groups)
                    if reward is None or not (isinstance(reward, (int, float)) and math.isfinite(reward)):
                        try:
                            reward = float(np.nan_to_num(raw_reward, nan=0.0, posinf=0.0, neginf=0.0))
                        except Exception:
                            reward = 0.0
                    reward = float(np.clip(reward, -1e6, 1e6))

                    total_reward += raw_reward if isinstance(raw_reward, (int, float)) else 0.0
                    rollout_reward += raw_reward if isinstance(raw_reward, (int, float)) else 0.0

                    local_batch.append(Step(
                        state=state.copy(),
                        logp=logp,
                        value=value,
                        reward=reward,
                        done=bool(done),
                        per_dev_types=np.array(per_dev_types, dtype=np.int64),
                        exp=exp_idx,
                        app=app_idx,
                        vis_mask=v.detach().cpu().numpy().astype(np.float32),
                    ))
                    state = my_state_fn() if not done else state

                else:
                    # Opponent turn: just step the env so we can reach our next turn
                    groups = self._opponent_action(env, turn, opponent_strategies, opponent_equilibrium)
                    _, _, _, done, *_ = env.step(groups)
                    state = my_state_fn() if not done else state

                steps_done += 1

                if budget_type == "steps" and steps_done >= budget:
                    break

                if done:
                    env = self.oracle.fresh_env()
                    if hasattr(env, "randomize_compromise_and_ownership"):
                        env.randomize_compromise_and_ownership()
                    for attr in ["step_num","defender_step","attacker_step","work_done",
                                "checkpoint_count","defensive_cost","clearning_cost",
                                "revert_count","scan_cnt","compromised_devices_cnt",
                                "edges_blocked","edges_added"]:
                        if hasattr(env, attr): setattr(env, attr, 0)
                    state = my_state_fn()

                    if budget_exhausted():
                        break

            if not local_batch:
                break

            # bootstrap value (recompute visibility for adjacency if GAT)
            with torch.no_grad():
                v_boot = build_visibility_mask(env, self.role).to(self.device)
                adj_boot = build_adjacency(env, self.D).to(self.device)
                adj_boot = masked_adjacency(adj_boot, v_boot) if USE_GAT else adj_boot
                s_tensor = _to_device(state, self.device).unsqueeze(0)
                nxt = self.net(s_tensor, adj_boot)
                next_v = float(nxt["value"].item())

            rewards = np.array([b.reward for b in local_batch], dtype=np.float32)
            dones   = np.array([b.done   for b in local_batch], dtype=np.float32)
            values  = np.array([b.value  for b in local_batch] + [next_v], dtype=np.float32)

            rewards = np.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)
            values  = np.nan_to_num(values,  nan=0.0, posinf=0.0,  neginf=0.0)

            adv, rets = compute_gae(rewards * REWARD_SCALE, values, dones)
            adv  = np.clip(np.nan_to_num(adv,  nan=0.0, posinf=ADV_CLIP,  neginf=-ADV_CLIP),  -ADV_CLIP, ADV_CLIP)
            rets = np.clip(np.nan_to_num(rets, nan=0.0, posinf=RET_CLIP,  neginf=-RET_CLIP),  -RET_CLIP, RET_CLIP)

            adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(rets, dtype=torch.float32, device=self.device)

            N_batch = adv_t.numel()
            if N_batch >= ADV_NORM_MIN_N:
                adv_mean = torch.nan_to_num(adv_t.mean(), nan=0.0)
                adv_std  = torch.nan_to_num(adv_t.std(),  nan=1.0).clamp_min(1e-3)
                adv_t    = ((adv_t - adv_mean) / adv_std).clamp(-3.0, 3.0)

            # tensors for update
            state_t = torch.tensor(np.stack([b.state for b in local_batch]), dtype=torch.float32, device=self.device)
            per_dev_types_t = torch.tensor(np.stack([b.per_dev_types for b in local_batch]),
                                           dtype=torch.long, device=self.device)  # [N, D]
            vis_t   = torch.tensor(np.stack([b.vis_mask for b in local_batch]), dtype=torch.float32, device=self.device)  # [N, D]
            exp_t   = torch.tensor([b.exp for b in local_batch], dtype=torch.long, device=self.device) if self.E > 0 else None
            app_t   = torch.tensor([b.app for b in local_batch], dtype=torch.long, device=self.device) if self.A > 0 else None
            old_logp= torch.tensor([b.logp for b in local_batch], dtype=torch.float32, device=self.device)
            old_val = torch.tensor([b.value for b in local_batch], dtype=torch.float32, device=self.device)

            # ---- PPO update with update-budget accounting ----
            N = state_t.size(0)
            epochs_this_rollout = 1 if single_update_per_rollout else ppo_epochs
            mb_size = N if single_update_per_rollout else minibatch_size

            def updates_budget_remaining() -> int:
                return max(0, budget - updates_done)

            if budget_type == "updates" and not single_update_per_rollout:
                nominal_mbs_per_epoch = (N + mb_size - 1) // mb_size
                nominal_total_updates = ppo_epochs * nominal_mbs_per_epoch
                rem = updates_budget_remaining()
                if rem < nominal_total_updates:
                    epochs_this_rollout = 1
                    mb_size = max(1, (N + rem - 1) // rem)

            loss = torch.tensor(0.0, device=self.device)
            policy_loss = torch.tensor(0.0, device=self.device)
            v_loss = torch.tensor(0.0, device=self.device)
            ent = torch.tensor(0.0, device=self.device)

            for ep in range(epochs_this_rollout):
                idxs = np.arange(N)
                np.random.shuffle(idxs)

                start = 0
                while start < N:
                    if budget_type == "updates" and updates_budget_remaining() <= 0:
                        break

                    end = min(start + mb_size, N)
                    mb = idxs[start:end]

                    s_mb   = state_t[mb]
                    pd_mb  = per_dev_types_t[mb]      # [B, D]
                    vis_mb = vis_t[mb]                # [B, D]
                    exp_mb = exp_t[mb] if exp_t is not None else None
                    app_mb = app_t[mb] if app_t is not None else None
                    adv_mb = adv_t[mb]
                    ret_mb = ret_t[mb].clamp(-VALUE_TARGET_CLIP, VALUE_TARGET_CLIP)
                    old_mb = old_logp[mb]
                    old_vb = old_val[mb]

                    if (not torch.isfinite(adv_mb).all()) or (not torch.isfinite(ret_mb).all()):
                        start = end
                        continue

                    # rebuild visibility-masked adjacency for this mini-batch (approx: use last env v)
                    # For stability/speed, reuse a dense adj with all ones; or pass a cached adj if needed.
                    v_last = vis_mb[0]  # heuristic; adj isn't used by non-GAT anyway
                    adj_mb = torch.ones((1, self.D, self.D), device=self.device)
                    if USE_GAT:
                        base_adj = build_adjacency(env, self.D).to(self.device)
                        adj_mb = masked_adjacency(base_adj, v_last)

                    with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                        out = self.net(s_mb, adj_mb)

                        per_dev_logits = torch.nan_to_num(out["per_dev_type_logits"], nan=0.0, posinf=0.0, neginf=0.0)  # [B,D,K]
                        B, D_curr, K = per_dev_logits.shape
                        per_dev_dists = [Categorical(logits=per_dev_logits[:, d, :]) for d in range(D_curr)]

                        logp_new = torch.zeros(B, device=self.device)
                        ent = torch.zeros(B, device=self.device)
                        for d, dist_d in enumerate(per_dev_dists):
                            # safe target: invisible → 0, visible → clip to valid range
                            tgt_d = torch.where(
                                vis_mb[:, d] > 0.5,
                                pd_mb[:, d].clamp(0, self.n_types - 1),
                                torch.zeros_like(pd_mb[:, d])
                            )
                            lp_d = dist_d.log_prob(tgt_d)      # [B]
                            H_d  = dist_d.entropy()            # [B]
                            m_d  = vis_mb[:, d]                # [B] 0/1
                            logp_new = logp_new + lp_d * m_d
                            ent      = ent + H_d * m_d
                        if self.E > 0:
                            exp_logits = torch.nan_to_num(out["exp_logits"], nan=0.0, posinf=0.0, neginf=0.0)
                            exp_dist = Categorical(logits=exp_logits)
                            logp_new = logp_new + exp_dist.log_prob(exp_mb)
                            ent      = ent + exp_dist.entropy()
                        if self.A > 0 and out["app_logits"] is not None:
                            app_logits = torch.nan_to_num(out["app_logits"], nan=0.0, posinf=0.0, neginf=0.0)
                            app_dist = Categorical(logits=app_logits)
                            logp_new = logp_new + app_dist.log_prob(app_mb)
                            ent      = ent + app_dist.entropy()

                        logp_new = torch.nan_to_num(logp_new, nan=0.0, posinf=0.0, neginf=0.0)
                        old_mb   = torch.nan_to_num(old_mb,   nan=0.0, posinf=0.0, neginf=0.0)
                        logp_diff = (logp_new - old_mb).clamp(-CLIP_LOGP_DIFF, CLIP_LOGP_DIFF)
                        ratio = torch.exp(logp_diff)

                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                        policy_loss = -torch.min(surr1, surr2).mean()

                        v = torch.nan_to_num(out["value"], nan=0.0, posinf=0.0, neginf=0.0)
                        v_clipped = old_vb + (v - old_vb).clamp(-VALUE_CLIP_EPS, VALUE_CLIP_EPS)
                        v_loss_unclipped = F.mse_loss(v, ret_mb)
                        v_loss_clipped   = F.mse_loss(v_clipped, ret_mb)
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped)

                        ent_mean = torch.nan_to_num(ent.mean(), nan=0.0, posinf=0.0, neginf=0.0)
                        loss = policy_loss - ENT_COEF * ent_mean + VF_COEF * v_loss

                    if torch.isfinite(loss):
                        self.opt.zero_grad(set_to_none=True)
                        if self.amp_enabled:
                            self.scaler.scale(loss).backward()
                            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                            self.scaler.step(self.opt); self.scaler.update()
                        else:
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                            self.opt.step()
                        updates_done += 1

                    start = end

            # progress print
            if steps_done - last_progress_print >= PROGRESS_EVERY_STEPS or budget_exhausted():
                if torch.is_tensor(loss):
                    last_loss = float(loss.detach().cpu()) if torch.isfinite(loss) else float("nan")
                    last_pol  = float(policy_loss.detach().cpu()) if torch.isfinite(policy_loss) else float("nan")
                    last_val  = float(v_loss.detach().cpu()) if torch.isfinite(v_loss) else float("nan")
                    last_ent  = float(ent_mean.detach().cpu()) if torch.isfinite(ent_mean) else float("nan")
                    last_advm = float(torch.abs(adv_t).mean().detach().cpu()) if torch.isfinite(torch.abs(adv_t).mean()) else float("nan")
                    print(f"[Progress] steps={steps_done} | rollouts={rollout_num} | last_loss={last_loss:.6f} "
                          f"pol={last_pol:.6f} val={last_val:.6f} ent={last_ent:.4f} | |adv|_mean={last_advm:.6f} "
                          f"| updates_done={updates_done}")
                else:
                    print(f"[Progress] steps={steps_done} | rollouts={rollout_num} | updates_done={updates_done}")
                last_progress_print = steps_done

            if budget_exhausted():
                break

        # package trained policy
        policy = MAPPOCommPolicy(self.oracle, self.role, state_dict=self.net.state_dict())
        strat = Strategy(type_mapping={"mappo": policy})
        strat.add_payoff(total_reward)
        return strat
