# hierarchical_br.py
from __future__ import annotations
import copy, random, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from strategy import Strategy

# -------------------------------
# Utils
# -------------------------------
def _safe_nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def build_visibility_mask(env, role: str) -> torch.Tensor:
    """
    Role-specific visibility mask v (length M, values in {0,1}).

    Attacker visible iff:
      Known_to_attacker AND attacker_owned AND NOT Not_yet_added

    Defender visible iff:
      (NOT Not_yet_added) AND attacker_owned
    """
    M = int(env.Max_network_size)
    v = torch.zeros(M, dtype=torch.float32)
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
    return v  # (M,)

def visible_subsets(subsets: List[List[int]], v: torch.Tensor) -> List[List[int]]:
    """
    Filter each partition subset to only visible device indices.
    """
    vis = []
    v_np = (v.detach().cpu().numpy() > 0.5)
    for s in subsets:
        vis.append([i for i in s if v_np[i]])
    return vis

# -------------------------------
# High-level: device scorers (no graph meta here)
# -------------------------------
class ScoreNet(nn.Module):
    """f_phi(s) -> raw device logit for each of M devices."""
    def __init__(self, state_dim: int, M: int, device: torch.device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, 256).to(device)
        self.fc2 = nn.Linear(256, M).to(device)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(s.to(self.device)))
        return self.fc2(x)  # (B, M)

# -------------------------------
# Low-level: two-stage end-to-end policy
# -------------------------------
class TwoStageEndToEnd(nn.Module):
    """
    Stage 1: state -> action-type logits
    Stage 2: concat([state, subset_mask_onehot]) -> M device-mask logits
      (we will *use only* the logits inside the chosen (visible) subset)
    """
    def __init__(self, state_dim: int, mask_len: int, M: int, n_types: int,
                 device: torch.device, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.mask_len  = mask_len      # here: mask_len == M (subset mask over devices)
        self.M         = M
        self.n_types   = n_types
        self.device    = device

        # action-type head
        self.act_body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.act_head = nn.Linear(hidden, n_types)

        # device head, conditioned on the chosen subset mask (length M)
        self.dev_body = nn.Sequential(
            nn.Linear(state_dim + mask_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.dev_head = nn.Linear(hidden, M)   # logits for all M devices

    def forward(self, state: torch.Tensor, subset_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # state: (B, state_dim), subset_mask: (B, M)
        x_act = self.act_body(state)
        atype_logits = self.act_head(x_act)

        x_dev = self.dev_body(torch.cat([state, subset_mask], dim=-1))
        dev_logits = self.dev_head(x_dev)

        return {
            "atype_logits": _safe_nan_to_num(atype_logits),
            "dev_logits":   _safe_nan_to_num(dev_logits),
        }

# -------------------------------
# HAGS: Hierarchical Best Response (with visibility-aware partitions)
# -------------------------------
class HierarchicalBestResponse:
    """
    High-level: choose a subset/partition (precomputed on the env graph), but
                *only* consider devices that are VISIBLE for the acting role.
    Low-level:  two-stage end-to-end policy; device logits are restricted to the
                visible devices inside the chosen subset.
    """
    def __init__(self, oracle, role: str, partition_size: Optional[int] = None):
        self.oracle = oracle
        self.role   = role
        self.env    = copy.deepcopy(oracle.env)
        self.device = oracle.device
        self.seed   = oracle.seed

        torch.manual_seed(self.seed); np.random.seed(self.seed); random.seed(self.seed)

        # dims
        self.state_dim = (self.env._get_defender_state().shape[0]
                          if role == "defender" else self.env._get_attacker_state().shape[0])
        self.M           = int(self.env.Max_network_size)
        self.num_types   = oracle.n_def_types if role == "defender" else oracle.n_att_types

        # partitions from the environment
        if partition_size is None:
            partition_size = int(math.ceil(math.sqrt(self.M)))
        # creates env.simulator.subnet.partitions: List[List[int]]
        self.env.simulator.subnet.create_partitions(partition_size)
        self.subsets: List[List[int]] = self.env.simulator.subnet.partitions

        # High-level subset selector (per-device scorer)
        self.score_net = ScoreNet(self.state_dim, self.M, self.device)
        self.hl_opt    = optim.Adam(self.score_net.parameters(), lr=1e-3)

        # Low-level two-stage policy (mask_len = M, since we pass a subset mask over M devices)
        self.low = TwoStageEndToEnd(
            state_dim=self.state_dim, mask_len=self.M, M=self.M, n_types=self.num_types,
            device=self.device, hidden=256
        ).to(self.device)
        self.low_opt = optim.Adam(self.low.parameters(), lr=3e-4)

        # loss hyperparams / stability
        self.beta_dev     = 1.0     # weight for device-mask log-prob term
        self.ent_coef_hi  = 1e-3
        self.ent_coef_at  = 1e-3
        self.ent_coef_dev = 1e-4
        self.max_grad_norm= 0.5

        # reward scaling/clipping
        self.reward_scale = 1e-2
        self.reward_clip  = 1e4

    # ----- low-level sampling restricted to a (visible) subset -----
    def _sample_low_within_subset(self, s_t: torch.Tensor, visible_subset: List[int]):
        """
        s_t: (1, state_dim)
        visible_subset: list of device indices (global) that are both in the chosen
                        partition and visible to this role right now.
        Returns (action_tuple, aux) for losses.
        """
        subset = visible_subset if len(visible_subset) > 0 else [0]

        # subset one-hot/mask over M devices
        mask = torch.zeros((1, self.M), device=self.device)
        mask[0, subset] = 1.0

        out = self.low(s_t, mask)
        at_logits  = out["atype_logits"]                  # (1, n_types)
        dev_logits = out["dev_logits"]                    # (1, M)

        # action-type categorical
        dist_at = torch.distributions.Categorical(logits=at_logits.squeeze(0))
        a_type  = dist_at.sample()
        logp_at = dist_at.log_prob(a_type)
        ent_at  = dist_at.entropy()

        # Bernoulli over only the chosen visible subset
        logits_sub = dev_logits[:, subset]                # (1, |subset|)
        probs_sub  = torch.sigmoid(logits_sub)            # (1, |subset|)
        dev_samp   = torch.bernoulli(probs_sub)           # (1, |subset|)

        # ensure at least one device in subset
        if dev_samp.sum() < 0.5:
            argm = torch.argmax(probs_sub, dim=1)
            dev_samp[0, argm] = 1.0

        # log-prob & entropy for the subset
        eps      = 1e-8
        logp_pos = torch.log(probs_sub + eps)
        logp_neg = torch.log(1.0 - probs_sub + eps)
        logp_dev = (dev_samp * logp_pos + (1.0 - dev_samp) * logp_neg).sum(dim=1).squeeze(0)
        ent_dev  = (-(probs_sub * logp_pos + (1.0 - probs_sub) * logp_neg)).sum(dim=1).squeeze(0)

        # turn subset-local indices into global device indices
        chosen_local = torch.nonzero(dev_samp[0] > 0.5, as_tuple=False).squeeze(1).tolist()
        chosen_global = np.array([subset[i] for i in chosen_local], dtype=int)
        if chosen_global.size == 0:  # extra safety
            chosen_global = np.array([subset[int(torch.argmax(probs_sub))]], dtype=int)

        action_tuple = (
            int(a_type.item()),
            np.array([0], dtype=int),            # exploit index fixed 0
            chosen_global,                       # device indices (global)
            0                                    # app index 0
        )

        aux = {
            "logp_at": logp_at,
            "ent_at":  ent_at,
            "logp_dev":logp_dev,
            "ent_dev": ent_dev,
        }
        return action_tuple, aux

    def _policy_loss(self, adv: torch.Tensor,
                     logp_hi: torch.Tensor, ent_hi: torch.Tensor,
                     low_aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        REINFORCE loss combining high-level subset choice and low-level choices.
        """
        logp_sum = logp_hi + low_aux["logp_at"] + self.beta_dev * low_aux["logp_dev"]
        ent_reg  = (self.ent_coef_hi * ent_hi
                    + self.ent_coef_at * low_aux["ent_at"]
                    + self.ent_coef_dev * low_aux["ent_dev"])
        return -(adv.detach() * logp_sum) - ent_reg

    # ----- training -----
    def train(self,
              opponent_strategies: List[Strategy],
              opponent_equilibrium: np.ndarray,
              T: int = 15_000) -> Strategy:
        env = self.env
        env.reset(from_init=True)
        # wipe counters
        for attr in ["step_num","defender_step","attacker_step","work_done",
                     "checkpoint_count","defensive_cost","clearning_cost",
                     "revert_count","scan_cnt","compromised_devices_cnt",
                     "edges_blocked","edges_added"]:
            if hasattr(env, attr): setattr(env, attr, 0)

        if self.role == "defender":
            get_my, get_oth = env._get_defender_state, env._get_attacker_state
        else:
            get_my, get_oth = env._get_attacker_state, env._get_defender_state

        running_baseline = 0.0
        steps_done = 0
        rollouts   = 0
        total_reward = 0.0

        while steps_done < T:






            turn = "defender" if (env.step_num % 2 == 0) else "attacker"
            env.mode = turn

            if turn == self.role:
                rollouts += 1

                # state + visibility
                s_np = np.asarray(get_my(), dtype=np.float32)
                s_t  = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                v    = build_visibility_mask(env, self.role).to(self.device)  # (M,)

                # High-level: device logits -> aggregate per (visible) subset -> Categorical
                dev_logits = self.score_net(s_t)[0]                      # (M,)
                vis_subsets = visible_subsets(self.subsets, v)           # List[List[int]]

                # subset scores: sum only over visible devices; empty -> -inf
                scores = []
                for s in vis_subsets:
                    if len(s) == 0:
                        scores.append(torch.tensor(-1e9, device=self.device))
                    else:
                        scores.append(dev_logits[s].sum())
                subset_scores = torch.stack(scores)                      # (num_subsets,)

                # if all are -inf (no visible devices anywhere), fallback to uniform over devices
                if torch.isneginf(subset_scores).all():
                    # treat each device as its own subset and pick any visible device
                    if v.sum() > 0:
                        probs_dev = F.softmax(dev_logits[v > 0], dim=0)
                        dev_choices = torch.nonzero(v > 0.5, as_tuple=False).squeeze(1)
                        idx_in_vis = torch.distributions.Categorical(probs_dev).sample()
                        fallback_subset = [int(dev_choices[idx_in_vis].item())]
                    else:
                        fallback_subset = [0]
                    choice = torch.tensor(0, device=self.device)  # dummy
                    logp_hi = torch.tensor(0.0, device=self.device)
                    ent_hi  = torch.tensor(0.0, device=self.device)
                    action_tuple, low_aux = self._sample_low_within_subset(s_t, fallback_subset)
                else:
                    probs   = F.softmax(subset_scores, dim=0)
                    dist_hi = torch.distributions.Categorical(probs)
                    choice  = dist_hi.sample()
                    logp_hi = dist_hi.log_prob(choice)
                    ent_hi  = dist_hi.entropy()

                    # Low-level: within the chosen *visible* subset
                    visible_subset = vis_subsets[int(choice.item())]
                    action_tuple, low_aux = self._sample_low_within_subset(s_t, visible_subset)

                # Env step (robust tuple handling)
                step_out = env.step(action_tuple)
                if len(step_out) >= 4:
                    _, raw_r, r, done, *rest = step_out if len(step_out) >= 5 else (step_out[0], 0.0, step_out[1], step_out[2], *step_out[3:])
                else:
                    raise RuntimeError("env.step returned unexpected tuple length")
                total_reward += float(raw_r)

                # Advantage with scale/clip
                rew_scaled = max(-self.reward_clip, min(self.reward_clip, float(r) * self.reward_scale))
                running_baseline = 0.99 * running_baseline + 0.01 * rew_scaled
                adv = torch.tensor([rew_scaled - running_baseline], dtype=torch.float32, device=self.device)

                # Joint update
                self.low_opt.zero_grad(set_to_none=True)
                self.hl_opt.zero_grad(set_to_none=True)

                loss = self._policy_loss(adv, logp_hi, ent_hi, low_aux)
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.low.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.score_net.parameters(), self.max_grad_norm)
                    self.low_opt.step()
                    self.hl_opt.step()

                if done:
                    # reset episode but keep networks
                    env = self.oracle.fresh_env()
                    if hasattr(env, "randomize_compromise_and_ownership"):
                        env.randomize_compromise_and_ownership()
                    for attr in ["step_num","defender_step","attacker_step","work_done",
                                 "checkpoint_count","defensive_cost","clearning_cost",
                                 "revert_count","scan_cnt","compromised_devices_cnt",
                                 "edges_blocked","edges_added"]:
                        if hasattr(env, attr): setattr(env, attr, 0)

            else:
                # Opponent turn: sample from equilibrium pool
                idx = np.random.choice(len(opponent_strategies), p=opponent_equilibrium)
                strat = opponent_strategies[idx]

                if strat.baseline_name is not None:
                    env.base_line = strat.baseline_name
                    action = None
                elif strat.actions is not None:
                    t = env.step_num
                    action = strat.actions[t % len(strat.actions)]
                else:
                    # Use the opponent's executor if available
                    if strat.type_mapping and ('mappo' in strat.type_mapping or 'marl' in strat.type_mapping):
                        state_vec = get_oth()
                        marl_agent = strat.type_mapping.get('mappo', strat.type_mapping.get('marl'))
                        try:
                            groups = marl_agent.select_action(state_vec, env=env)
                            action = groups if isinstance(groups, list) else [groups]
                        except Exception:
                            action = None
                    elif strat.type_mapping and 'hierarchical' in strat.type_mapping:
                        # another HAGS policy
                        action = HierarchicalBestResponse(self.oracle, turn).execute(strat, get_oth())
                    elif strat.type_mapping and 'meta' in strat.type_mapping:
                        from meta_hierarchical_br import MetaHierarchicalBestResponse as _Meta
                        action = _Meta(self.oracle, turn).execute(strat, get_oth())
                    else:
                        action = None  # fallback baseline

                env.step(action)
                if getattr(env, "time_budget_exceeded", False):
                    print("[HAGS] time budget exceeded — ending training early")
                    break

            steps_done += 1
            if steps_done % 5000 == 0:
                print(f"[HAGS] steps={steps_done} | rollouts={rollouts}")

        # Package the trained strategy
        return Strategy(
            actor_state_dict=None,
            critic_state_dict=None,
            actions=None,
            baseline_name=None,
            actor_dims=None,
            critic_dims=None,
            type_mapping={
                "hierarchical": {
                    "score_net": self.score_net.state_dict(),
                    "two_stage": self.low.state_dict(),
                    "M": self.M,
                    "partition_size": int(math.ceil(math.sqrt(self.M))),
                }
            }
        )

    # ----- execution (deterministic, visibility-aware) -----
    def execute(self, strat: Strategy, state: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, int]:
        if not strat.type_mapping or "hierarchical" not in strat.type_mapping:
            raise ValueError("Not a hierarchical strategy")

        # restore dims/nets
        self.M = int(strat.type_mapping["hierarchical"].get("M", self.M))

        score_net = ScoreNet(self.state_dim, self.M, self.device).to(self.device).eval()
        score_net.load_state_dict(strat.type_mapping["hierarchical"]["score_net"])

        low = TwoStageEndToEnd(
            state_dim=self.state_dim, mask_len=self.M, M=self.M, n_types=self.num_types,
            device=self.device, hidden=256
        ).to(self.device).eval()
        low.load_state_dict(strat.type_mapping["hierarchical"]["two_stage"])

        # ensure we have partitions in the fresh env copy
        part_size = strat.type_mapping["hierarchical"].get("partition_size", int(math.ceil(math.sqrt(self.M))))
        self.env.simulator.subnet.create_partitions(int(part_size))
        self.subsets = self.env.simulator.subnet.partitions

        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        v   = build_visibility_mask(self.env, self.role).to(self.device)
        vis_subsets = visible_subsets(self.subsets, v)

        with torch.no_grad():
            dev_logits = score_net(s_t)[0]                           # (M,)

            scores = []
            for s in vis_subsets:
                if len(s) == 0:
                    scores.append(torch.tensor(-1e9, device=self.device))
                else:
                    scores.append(dev_logits[s].sum())
            subset_scores = torch.stack(scores)                      # (num_subsets,)

            if torch.isneginf(subset_scores).all():
                # no visible devices in any subset → fallback to any visible device (or 0)
                if v.sum() > 0:
                    probs_dev = F.softmax(dev_logits[v > 0], dim=0)
                    dev_choices = torch.nonzero(v > 0.5, as_tuple=False).squeeze(1)
                    idx_in_vis = torch.argmax(probs_dev)
                    chosen_subset_vis = [int(dev_choices[idx_in_vis].item())]
                else:
                    chosen_subset_vis = [0]
            else:
                chosen_subset = int(torch.argmax(subset_scores).item())
                chosen_subset_vis = vis_subsets[chosen_subset]
                if len(chosen_subset_vis) == 0:
                    # safety: pick any visible device
                    if v.sum() > 0:
                        chosen_subset_vis = [int(torch.argmax(dev_logits * (v > 0)).item())]
                    else:
                        chosen_subset_vis = [0]

            # low-level forward on visible mask of chosen subset
            mask = torch.zeros((1, self.M), device=self.device)
            mask[0, chosen_subset_vis] = 1.0
            out = low(s_t, mask)
            atype = int(torch.argmax(out["atype_logits"], dim=-1).item())

            # devices only within chosen visible subset
            probs = torch.sigmoid(out["dev_logits"][:, chosen_subset_vis])      # (1, |subset|)
            sel   = (probs[0] > 0.5)
            if sel.sum() == 0:
                sel[torch.argmax(probs[0])] = 1
            chosen_local = torch.nonzero(sel, as_tuple=False).squeeze(1).tolist()
            dev_indices = np.array([chosen_subset_vis[i] for i in chosen_local], dtype=int)

        action = (
            atype,
            np.array([0], dtype=int),
            dev_indices,
            0
        )
        return action
