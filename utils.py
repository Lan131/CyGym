# utils.py
from __future__ import annotations
import os, copy, math, pickle, random, logging, time
from typing import List
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from timing_utils import timing

from strategy import Strategy
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv

# -------------------- Timing --------------------
HOTSPOTS = defaultdict(float)

def attach_expert_baselines(oracle):
    """
    Train and attach expert baselines (IPPO, MAPPO, HMARLExpert, HMARLMeta)
    to oracle.attacker_strategies / oracle.defender_strategies, if they do
    not already exist.

    This should be called AFTER Double Oracle has converged and
    oracle.defender_equilibrium / oracle.attacker_equilibrium are populated.
    """

    import numpy as np  # make sure numpy is available here

    # ---------- helpers ----------
    def has_baseline(strategies, name: str) -> bool:
        for s in strategies:
            if getattr(s, "baseline_name", None) == name:
                return True
        return False

    def append_strategy(oracle, role: str, strat, baseline_name: str):
        """
        Attach a new learned strategy to the oracle for the given role.
        We mirror the pattern you already use in run_game:
          - append to attacker_strategies / defender_strategies
          - push actor/critic state_dict into saved_* lists
          - set .baseline_name so compute_tabular_results can find it
        """
        if strat is None:
            return

        strat.baseline_name = baseline_name

        if role == "attacker":
            oracle.attacker_strategies.append(strat)
            actor_sd = getattr(strat, "actor_state_dict", None)
            critic_sd = getattr(strat, "critic_state_dict", None)
            if hasattr(oracle, "saved_attacker_actors"):
                oracle.saved_attacker_actors.append(actor_sd)
            if hasattr(oracle, "saved_attacker_critics"):
                oracle.saved_attacker_critics.append(critic_sd)
        else:
            oracle.defender_strategies.append(strat)
            actor_sd = getattr(strat, "actor_state_dict", None)
            critic_sd = getattr(strat, "critic_state_dict", None)
            if hasattr(oracle, "saved_defender_actors"):
                oracle.saved_defender_actors.append(actor_sd)
            if hasattr(oracle, "saved_defender_critics"):
                oracle.saved_defender_critics.append(critic_sd)

    # ---------- ensure we have an equilibrium ----------
    # We'll also rebuild payoff matrices once to get correct shapes.
    D_mat, A_mat = oracle.build_payoff_matrices(sparsify=False)

    if getattr(oracle, "defender_equilibrium", None) is None or \
       getattr(oracle, "attacker_equilibrium", None) is None:
        p, q = oracle.solve_nash_equilibrium(D_mat, A_mat, prune=False)
        oracle.defender_equilibrium = p
        oracle.attacker_equilibrium = q
    else:
        p = np.asarray(oracle.defender_equilibrium, dtype=float)
        q = np.asarray(oracle.attacker_equilibrium, dtype=float)

    # normalize just in case
    if p.sum() <= 0:
        p[:] = 1.0 / p.size
    else:
        p /= p.sum()
    if q.sum() <= 0:
        q[:] = 1.0 / q.size
    else:
        q /= q.sum()

    def_pool = list(oracle.defender_strategies)
    att_pool = list(oracle.attacker_strategies)

    # ==========================================================
    # 1) IPPO baselines (attacker + defender)
    # ==========================================================
    try:
        from IPPO import IPPOCommBestResponse
    except ImportError:
        IPPOCommBestResponse = None

    if IPPOCommBestResponse is not None:
        # ---- attacker IPPO baseline ----
        if not has_baseline(oracle.attacker_strategies, "IPPO"):
            print("[attach_expert_baselines] Training IPPO attacker baseline...")
            ippo_att = IPPOCommBestResponse(oracle, "attacker").train(
                def_pool, p,
                T=15000,
                budget_type="updates",
                budget=15000,
                single_update_per_rollout=True,
            )
            append_strategy(oracle, "attacker", ippo_att, "IPPO")

        # ---- defender IPPO baseline ----
        if not has_baseline(oracle.defender_strategies, "IPPO"):
            print("[attach_expert_baselines] Training IPPO defender baseline...")
            ippo_def = IPPOCommBestResponse(oracle, "defender").train(
                att_pool, q,
                T=15000,
                budget_type="updates",
                budget=15000,
                single_update_per_rollout=True,
            )
            append_strategy(oracle, "defender", ippo_def, "IPPO")

    # ==========================================================
    # 2) MAPPO baselines (attacker + defender)
    # ==========================================================
    try:
        from MAPPO import MAPPOCommBestResponse
    except ImportError:
        MAPPOCommBestResponse = None

    if MAPPOCommBestResponse is not None:
        # ---- attacker MAPPO baseline ----
        if not has_baseline(oracle.attacker_strategies, "MAPPO"):
            print("[attach_expert_baselines] Training MAPPO attacker baseline...")
            mappo_att = MAPPOCommBestResponse(oracle, "attacker").train(
                def_pool, p,
                T=7500,
                rollout_len=1,
                ppo_epochs=1,
                minibatch_size=256,
                budget_type="updates",
                budget=7500,
                single_update_per_rollout=True,
            )
            append_strategy(oracle, "attacker", mappo_att, "MAPPO")

        # ---- defender MAPPO baseline ----
        if not has_baseline(oracle.defender_strategies, "MAPPO"):
            print("[attach_expert_baselines] Training MAPPO defender baseline...")
            mappo_def = MAPPOCommBestResponse(oracle, "defender").train(
                att_pool, q,
                T=7500,
                rollout_len=1,
                ppo_epochs=1,
                minibatch_size=256,
                budget_type="updates",
                budget=7500,
                single_update_per_rollout=True,
            )
            append_strategy(oracle, "defender", mappo_def, "MAPPO")

    # ==========================================================
    # 3) HMARLExpert baselines (attacker + defender)
    # ==========================================================
    try:
        from HMARL import (
            HMARLExpertBestResponse,
            ExpertRuleMaster,
            FrozenSubPolicy,
        )
        import torch.nn as nn
        import torch
    except ImportError:
        HMARLExpertBestResponse = None

    if HMARLExpertBestResponse is not None:
        class DummySubpolicyNet(nn.Module):
            def __init__(self, obs_dim: int):
                super().__init__()
                self.fc = nn.Linear(obs_dim, 8)

            def forward(self, x):
                return self.fc(x)

        # ---- attacker HMARLExpert ----
        if not has_baseline(oracle.attacker_strategies, "HMARLExpert"):
            print("[attach_expert_baselines] Training HMARLExpert attacker baseline...")
            attacker_state_vec = oracle.env._get_attacker_state()
            attacker_state_dim = attacker_state_vec.shape[0]

            dummy_net_lowcost  = DummySubpolicyNet(attacker_state_dim)
            dummy_net_highcost = DummySubpolicyNet(attacker_state_dim)

            # NOTE: these action_type indices are based on your earlier attacker mapping
            lowcost_action_types  = [2, 3]  # cheap probe / scan
            highcost_action_types = [1]     # expensive exploit / spread

            subpolicies_att = [
                FrozenSubPolicy(
                    policy_net=dummy_net_lowcost,
                    device=oracle.device,
                    name="LowCostInvestigate",
                    role="attacker",
                    allowed_action_types=lowcost_action_types,
                ),
                FrozenSubPolicy(
                    policy_net=dummy_net_highcost,
                    device=oracle.device,
                    name="HighCostExploit",
                    role="attacker",
                    allowed_action_types=highcost_action_types,
                ),
            ]

            expert_master_att = ExpertRuleMaster(
                cheaplocal_idx=0,
                costlylocal_idx=1,
                global_idx=0,
                global_prob=0.0,
            )

            hmarlexpert_att_br = HMARLExpertBestResponse(
                oracle=oracle,
                role="attacker",
                subpolicies=subpolicies_att,
                expert_master=expert_master_att,
                device=oracle.device,
            )
            hmarlexpert_att = hmarlexpert_att_br.train(
                def_pool, p, T=15000, return_meta=True
            )
            append_strategy(oracle, "attacker", hmarlexpert_att, "HMARLExpert")

        # ---- defender HMARLExpert ----
        if not has_baseline(oracle.defender_strategies, "HMARLExpert"):
            print("[attach_expert_baselines] Training HMARLExpert defender baseline...")
            defender_state_vec = oracle.env._get_defender_state()
            defender_state_dim = defender_state_vec.shape[0]

            dummy_net_lowcost_d  = DummySubpolicyNet(defender_state_dim)
            dummy_net_highcost_d = DummySubpolicyNet(defender_state_dim)

            # TODO: adjust these indices to match your defender action mapping
            lowcost_def_action_types  = [0, 2, 3]
            highcost_def_action_types = [1]

            subpolicies_def = [
                FrozenSubPolicy(
                    policy_net=dummy_net_lowcost_d,
                    device=oracle.device,
                    name="LowCostDefend",
                    role="defender",
                    allowed_action_types=lowcost_def_action_types,
                ),
                FrozenSubPolicy(
                    policy_net=dummy_net_highcost_d,
                    device=oracle.device,
                    name="HighCostDefend",
                    role="defender",
                    allowed_action_types=highcost_def_action_types,
                ),
            ]

            expert_master_def = ExpertRuleMaster(
                cheaplocal_idx=0,
                costlylocal_idx=1,
                global_idx=0,
                global_prob=0.0,
            )

            hmarlexpert_def_br = HMARLExpertBestResponse(
                oracle=oracle,
                role="defender",
                subpolicies=subpolicies_def,
                expert_master=expert_master_def,
                device=oracle.device,
            )
            hmarlexpert_def = hmarlexpert_def_br.train(
                att_pool, q, T=15000, return_meta=True
            )
            append_strategy(oracle, "defender", hmarlexpert_def, "HMARLExpert")

    # ==========================================================
    # 4) HMARLMeta baselines (attacker + defender)
    # ==========================================================
    try:
        from HMARL import (
            HMARLMetaBestResponse,
            FrozenSubPolicy,
        )
        import torch.nn as nn
        import torch
    except ImportError:
        HMARLMetaBestResponse = None

    if HMARLMetaBestResponse is not None:
        class DummySubpolicyNetMeta(nn.Module):
            def __init__(self, obs_dim: int):
                super().__init__()
                self.fc = nn.Linear(obs_dim, 8)

            def forward(self, x):
                return self.fc(x)

        # ---- attacker HMARLMeta ----
        if not has_baseline(oracle.attacker_strategies, "HMARLMeta"):
            print("[attach_expert_baselines] Training HMARLMeta attacker baseline...")
            attacker_state_vec = oracle.env._get_attacker_state()
            attacker_state_dim = attacker_state_vec.shape[0]

            dummy_net_lowcost  = DummySubpolicyNetMeta(attacker_state_dim)
            dummy_net_highcost = DummySubpolicyNetMeta(attacker_state_dim)

            lowcost_action_types  = [2, 3]
            highcost_action_types = [1]

            subpolicies_att = [
                FrozenSubPolicy(
                    policy_net=dummy_net_lowcost,
                    device=oracle.device,
                    name="LowCostExpert",
                    role="attacker",
                    allowed_action_types=lowcost_action_types,
                ),
                FrozenSubPolicy(
                    policy_net=dummy_net_highcost,
                    device=oracle.device,
                    name="HighCostExpert",
                    role="attacker",
                    allowed_action_types=highcost_action_types,
                ),
            ]

            hmeta_att_br = HMARLMetaBestResponse(
                oracle=oracle,
                role="attacker",
                subpolicies=subpolicies_att,
                state_dim=attacker_state_dim,
                device=oracle.device,
            )
            hmeta_att = hmeta_att_br.train(
                def_pool, p, T=15000, return_meta=True
            )
            append_strategy(oracle, "attacker", hmeta_att, "HMARLMeta")

        # ---- defender HMARLMeta ----
        if not has_baseline(oracle.defender_strategies, "HMARLMeta"):
            print("[attach_expert_baselines] Training HMARLMeta defender baseline...")
            defender_state_vec = oracle.env._get_defender_state()
            defender_state_dim = defender_state_vec.shape[0]

            dummy_net_lowcost_d  = DummySubpolicyNetMeta(defender_state_dim)
            dummy_net_highcost_d = DummySubpolicyNetMeta(defender_state_dim)

            # TODO: adjust for your defender action mapping
            lowcost_def_action_types  = [0, 2, 3]
            highcost_def_action_types = [1]

            subpolicies_def = [
                FrozenSubPolicy(
                    policy_net=dummy_net_lowcost_d,
                    device=oracle.device,
                    name="LowCostDefMeta",
                    role="defender",
                    allowed_action_types=lowcost_def_action_types,
                ),
                FrozenSubPolicy(
                    policy_net=dummy_net_highcost_d,
                    device=oracle.device,
                    name="HighCostDefMeta",
                    role="defender",
                    allowed_action_types=highcost_def_action_types,
                ),
            ]

            hmeta_def_br = HMARLMetaBestResponse(
                oracle=oracle,
                role="defender",
                subpolicies=subpolicies_def,
                state_dim=defender_state_dim,
                device=oracle.device,
            )
            hmeta_def = hmeta_def_br.train(
                att_pool, q, T=15000, return_meta=True
            )
            append_strategy(oracle, "defender", hmeta_def, "HMARLMeta")

    print("[attach_expert_baselines] Done attaching expert baselines.")
    print("=== Baseline index maps ===")
    print("Defender strategies:")
    for i, s in enumerate(oracle.defender_strategies):
        print(f"  idx={i:2d}, name={getattr(s, 'baseline_name', None)!r}, type={type(s).__name__}")
    def_index = {
        getattr(s, "baseline_name", None): i
        for i, s in enumerate(oracle.defender_strategies)
        if getattr(s, "baseline_name", None) is not None
    }
    print("def_index:", def_index)

    print("Attacker strategies:")
    for j, s in enumerate(oracle.attacker_strategies):
        print(f"  idx={j:2d}, name={getattr(s, 'baseline_name', None)!r}, type={type(s).__name__}")
    att_index = {
        getattr(s, "baseline_name", None): j
        for j, s in enumerate(oracle.attacker_strategies)
        if getattr(s, "baseline_name", None) is not None
    }
    print("att_index:", att_index)


def perform_tab_results(info_path, defender_ddpg, attacker_ddpg, TIMECHECK, double_oracle, seed, tabular_sims, steps_per_episode, output_dir ,BR_type , env):
    if info_path is not None:
        with timing("write ddpg hyperparams to info_path", enabled=TIMECHECK):
            with open(info_path, "a") as f:
                for side, ddpg in [("defender", defender_ddpg),
                                ("attacker", attacker_ddpg)]:
                    a_lr = ddpg["actor_optimizer"].param_groups[0]["lr"]
                    c_lr = ddpg["critic_optimizer"].param_groups[0]["lr"]
                    buf  = ddpg["replay_buffer"].buffer
                    critic = ddpg["critic"]
                    layers = [
                        ("fc1", critic.fc1.in_features, critic.fc1.out_features),
                        ("fc2", critic.fc2.in_features, critic.fc2.out_features),
                        ("fc3", critic.fc3.in_features, critic.fc3.out_features),
                    ]
                    f.write(f"\n# {side.upper()} DDPG\n")
                    f.write(f"actor_lr: {a_lr}\n")
                    f.write(f"critic_lr: {c_lr}\n")
                    try:
                        f.write(f"replay_buffer_capacity: {buf.maxlen}\n")
                    except Exception:
                        f.write(f"replay_buffer_size: {len(buf)}\n")
                    f.write("critic_architecture:\n")
                    for name, inp, out in layers:
                        f.write(f"  {name}: Linear({inp} → {out})\n")

                    f.write("— greedy_device_coord_ascent hyper-parameters —\n")
                    f.write(f"  K         = {double_oracle.coord_K}\n")
                    f.write(f"  τ (tau)   = {double_oracle.coord_tau}\n")
                    f.write(f"  noise_std = {double_oracle.coord_noise_std}\n\n")
                    f.write("— network evolution parameters —\n")
                    f.write(f"  λ_events  = {env.lambda_events}\n")
                    f.write(f"  p_add     = {env.p_add}\n")
                    f.write(f"  p_attacker= {env.p_attacker}\n\n")

    with timing("compute_tabular_results", enabled=TIMECHECK):
        atk_latex, def_latex, extended_metrics = compute_tabular_results(
            oracle=double_oracle,
            tabular_seeds=[seed],
            tabular_sims=tabular_sims,
            steps_per_episode=steps_per_episode,
            save_dir=os.path.join(output_dir, "models"),
            BR_type=BR_type
        )

    with timing("append LaTeX tables to info_path", enabled=TIMECHECK):
        with open(info_path, "a") as f:
            f.write("\n=== PAIRWISE TABULAR RESULTS (LaTeX) ===\n\n")
            f.write("% Attacker payoffs\n")
            f.write(atk_latex + "\n\n")
            f.write("% Defender payoffs\n")
            f.write(def_latex + "\n")
            f.write("\n=== Extended Metrics (LaTeX) ===\n\n")
            f.write(extended_metrics + "\n")

    print("Simulation Complete")
    exit(0)    


def _safe_simplex_rows(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # logits: (..., K)
    logp = F.log_softmax(logits, dim=dim)
    probs = logp.exp()
    # zero out non-finite
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    # row sums
    s = probs.sum(dim=dim, keepdim=True)
    # rows with zero mass → uniform
    K = probs.size(dim)
    bad = (s <= 0)
    if bad.any():
        uniform = torch.full_like(probs, 1.0 / K)
        probs = torch.where(bad, uniform, probs)
        s = probs.sum(dim=dim, keepdim=True)
    probs = probs / s
    return probs, logp


def report_hotspots(top_k: int = 30, sink=print, header: str = "=== HOTSPOTS (cum sec) ==="):
    if not sink:
        return
    sink(header)
    if not HOTSPOTS:
        sink("(no timings recorded)")
        return
    for label, total in sorted(HOTSPOTS.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        sink(f"{total:9.3f}s  {label}")

# -------------------- LaTeX helpers --------------------

def generate_extended_metrics_table(attacker_bases, defender_bases, all_metrics, num_dev, *, seeds=None):
    metric_names = [
        "reset_count", "checkpoint_count", "compromised", "jobs",
        "defensive_cost", "scan_count", "edges_blocked", "edges_added"
    ]
    metric_titles = {
        "reset_count": "Total checkpoint + revert count",
        "checkpoint_count": "Total checkpoint count",
        "compromised": "Average compromised devices per timestep",
        "jobs": "Average workloads executed",
        "defensive_cost": "Total defensive cost",
        "scan_count": "Total scans performed",
        "edges_blocked": "Total edges blocked",
        "edges_added": "Total edges added"
    }
    metric_indices = {
        "reset_count": None, "checkpoint_count": 6, "compromised": 2, "jobs": 3,
        "scan_count": 4, "defensive_cost": 5, "edges_blocked": 8, "edges_added": 9
    }

    seed_line = (
        r"\noindent\textit{Seeds: }" + ", ".join(str(s) for s in seeds) + "\n"
        if seeds is not None and len(seeds) > 0 else ""
    )

    tables = []
    for metric in metric_names:
        cap = metric_titles[metric]
        lines = [
            seed_line,
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{{cap}}}",
            r"\begin{tabular}{l" + "c" * len(defender_bases) + r"}",
            r"\toprule",
            r"& \multicolumn{" + str(len(defender_bases)) + r"}{c}{\textbf{Defender strategies}} \\",
            r"\cmidrule{2-" + str(len(defender_bases) + 1) + r"}",
            r"\textbf{Attacker $\downarrow$ \quad Defender $\rightarrow$} & " +
                " & ".join(f"\\textbf{{{d}}}" for d in defender_bases) + r" \\",
            r"\midrule",
        ]
        for a in attacker_bases:
            row = [f"\\textbf{{{a}}}"]
            for d in defender_bases:
                key = (a, d)
                if key in all_metrics and len(all_metrics[key]) > 0:
                    if metric == "reset_count":
                        vals = [m[6] + m[7] for m in all_metrics[key]]
                    else:
                        idx = metric_indices[metric]
                        vals = [m[idx] for m in all_metrics[key]]
                    n = len(vals)
                    mean = float(sum(vals)) / n
                    std_err = (np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    row.append(f"${mean:.3f}\\pm{std_err:.3f}$")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tables.append("\n".join(lines))
    return "\n\n".join(tables)

def generate_latex_table(attacker_bases, defender_bases, all_results, num_dev, payoff_type="attacker", *, seeds=None):
    cap = f"{payoff_type.capitalize()} average payoffs (mean\\,$\\pm$\\,std)"
    seed_line = (
        r"\noindent\textit{Seeds: }" + ", ".join(str(s) for s in seeds) + "\n"
        if seeds is not None and len(seeds) > 0 else ""
    )
    lines = [
        seed_line,
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{cap}}}",
        r"\begin{tabular}{l" + "c" * len(defender_bases) + "}",
        r"\toprule",
        r"& \multicolumn{" + str(len(defender_bases)) + r"}{c}{\textbf{Defender strategies}} \\",
        r"\cmidrule(lr){2-" + str(len(defender_bases)+1) + r"}",
        r"\textbf{Attacker $\downarrow$ \quad Defender $\rightarrow$} & " +
            " & ".join(f"\\textbf{{{d}}}" for d in defender_bases) + r" \\",
        r"\midrule",
    ]
    for atk in attacker_bases:
        row = [f"\\textbf{{{atk}}}"]
        for deff in defender_bases:
            arr = np.array(all_results[(atk, deff)])
            if arr.size == 0:
                row.append("--")
                continue
            d_vals = arr[:, 0] / max(1, num_dev)
            a_vals = arr[:, 1] / max(1, num_dev)
            vals = a_vals if payoff_type == "attacker" else d_vals
            m = float(vals.mean())
            s = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row.append(f"${m:.3f}\\pm{s:.3f}$")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

# -------------------- Repro --------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smooth_rewards(rewards, window=10):
    out = []
    for i in range(len(rewards)):
        s = max(0, i - window + 1)
        out.append(float(np.mean(rewards[s:i+1])))
    return out

def initialize_reward_dict(steps_per_episode, episodes):
    max_steps = steps_per_episode * episodes + 1
    return {i: [] for i in range(0, max_steps, 100)}

# -------------------- Mixture-aware tabulation (FIX) --------------------

def _payoff_do_vs_att_baseline(D_mat, A_mat, p, att_idx):
    d = float(p.dot(D_mat[:, att_idx]))
    a = float(A_mat[att_idx, :].dot(p))
    return d, a

def _payoff_def_baseline_vs_do(D_mat, A_mat, q, def_idx):
    d = float(D_mat[def_idx, :].dot(q))
    a = float(q.dot(A_mat[:, def_idx]))
    return d, a

def compute_tabular_results(
    oracle: 'DoubleOracle',
    tabular_seeds: List[int],
    tabular_sims: int,
    steps_per_episode: int,
    save_dir: str,
    BR_type: str,
    TIMECHECK: bool = False,
    *,
    only_do_entry: bool = True,  
    do_expert_baselines: bool = False, 
):
    """
    Compute tabular results. If only_do_entry=True, we will only compute the DO vs DO
    expected payoff (and mixture-expected metrics) and return tables that contain only
    the DO row/column (keeps the same return format: attacker-table, defender-table, extended-metrics).
    """

    # Optionally ensure expert baselines are attached
    if do_expert_baselines and not only_do_entry:
        has_any_baseline = any(
            getattr(s, "baseline_name", None) in {"IPPO", "MAPPO", "HMARLExpert", "HMARLMeta"}
            for s in (list(oracle.attacker_strategies) + list(oracle.defender_strategies))
        )
        if not has_any_baseline:
            try:
                print("[compute_tabular_results] No expert baselines found; calling attach_expert_baselines(...)")
                attach_expert_baselines(oracle)
            except Exception as e:
                logging.warning(
                    "[compute_tabular_results] attach_expert_baselines failed with %r; proceeding with existing strategies.",
                    e,
                )

    if only_do_entry:
        attacker_bases = ["DO"]
        defender_bases = ["DO"]
    else:
        if do_expert_baselines:
            # "expert" MARL baselines on both sides
            attacker_bases = ["DO", "IPPO", "MAPPO", "HMARLExpert", "HMARLMeta"]
            defender_bases = ["DO", "IPPO", "MAPPO", "HMARLExpert", "HMARLMeta"]
        else:
            # original baselines (RandomInit / No Attack / No Defense / Preset)
            attacker_bases = ["DO", "RandomInit", "No Attack"]
            defender_bases = ["DO", "RandomInit", "No Defense", "Preset"]
    all_results = defaultdict(list)
    all_metrics = defaultdict(list)

    # MC per mixture-cell: lowered for large networks to avoid explosion
    S_per = 1  # keep conservative default

    for seed in tabular_seeds:
        print(f"[==== seed={seed} ====]")
        set_seed(seed)

        # Force memory-safe settings for very large networks
        try:
            if getattr(oracle.env, "numOfDevice", 0) >= 2000:
                oracle.parallel_rollouts = False
                oracle.rollout_workers = 1
                oracle.N_MC = 1
                S_per = 1
        except Exception:
            pass
        oracle.env.time_budget_deadline = False

        with timing("build_payoff_matrices", TIMECHECK):
            D_mat, A_mat = oracle.build_payoff_matrices(sparsify=False)

        if oracle.defender_equilibrium is None or oracle.attacker_equilibrium is None:
            with timing("solve_nash_equilibrium", TIMECHECK):
                oracle.solve_nash_equilibrium(D_mat, A_mat, prune=False)

        p = np.asarray(oracle.defender_equilibrium, dtype=float)
        q = np.asarray(oracle.attacker_equilibrium, dtype=float)

        # Build baseline index maps *only* if we will need them (i.e. not only_do_entry)
        def_index = {}
        att_index = {}
        if not only_do_entry:
            def_index = {
                s.baseline_name: i
                for i, s in enumerate(oracle.defender_strategies)
                if getattr(s, "baseline_name", None) is not None
            }
            att_index = {
                s.baseline_name: j
                for j, s in enumerate(oracle.attacker_strategies)
                if getattr(s, "baseline_name", None) is not None
            }

        for atk_base in attacker_bases:
            for def_base in defender_bases:
                for run_idx in range(tabular_sims):
                    set_seed(seed * 100 + run_idx)

                    # choose concrete strats only for metrics simulation
                    if def_base == "DO":
                        idx_def = np.random.choice(len(oracle.defender_strategies), p=p)
                        def_strat = oracle.defender_strategies[idx_def]
                    elif def_base == "RandomInit":
                        idx_def = def_index.get("RandomInit", None)
                        if idx_def is None:
                            logging.warning(
                                "[compute_tabular_results] Defender baseline 'RandomInit' not found; falling back to index 0"
                            )
                            idx_def = 0
                        def_strat = oracle.defender_strategies[idx_def]
                    else:
                        idx_def = def_index.get(def_base, None)
                        if idx_def is None:
                            logging.warning(
                                "[compute_tabular_results] Defender baseline %r not found; falling back to index 0",
                                def_base,
                            )
                            idx_def = 0
                        def_strat = oracle.defender_strategies[idx_def]

                    # attacker selection
                    if atk_base == "DO":
                        idx_att = np.random.choice(len(oracle.attacker_strategies), p=q)
                        att_strat = oracle.attacker_strategies[idx_att]
                    elif atk_base == "RandomInit":
                        idx_att = att_index.get("RandomInit", None)
                        if idx_att is None:
                            logging.warning(
                                "[compute_tabular_results] Attacker baseline 'RandomInit' not found; falling back to index 0"
                            )
                            idx_att = 0
                        att_strat = oracle.attacker_strategies[idx_att]
                    else:
                        idx_att = att_index.get(atk_base, None)
                        if idx_att is None:
                            logging.warning(
                                "[compute_tabular_results] Attacker baseline %r not found; falling back to index 0",
                                atk_base,
                            )
                            idx_att = 0
                        att_strat = oracle.attacker_strategies[idx_att]

                    # ---- PAYOFFS (analytical expectations when any DO appears) ----
                    # If both are DO use analytic expectation over mixture (and compute mixture metrics)
                    if def_base == "DO" and atk_base == "DO":
                        d_pay = float(p.dot(D_mat).dot(q))
                        a_pay = float(q.dot(A_mat).dot(p))

                        # mixture metrics (expectation over mixture cells)
                        exp_metrics = np.zeros(10, dtype=float)
                        D_strats = oracle.defender_strategies
                        A_strats = oracle.attacker_strategies
                        for i, pi in enumerate(p):
                            if pi == 0:
                                continue
                            for j, qj in enumerate(q):
                                if qj == 0:
                                    continue
                                with timing("simulate_game[mixture-cell]", TIMECHECK, sink=None):
                                    agg = None
                                    for _ in range(S_per):
                                        res = oracle.simulate_game(D_strats[i], A_strats[j], num_simulations=1)
                                        if agg is None:
                                            agg = np.array(res, dtype=np.float64)
                                        else:
                                            agg += np.array(res, dtype=np.float64)
                                    avg_ij = agg / float(S_per)
                                exp_metrics += pi * qj * avg_ij

                        all_results[(atk_base, def_base)].append((d_pay, a_pay))
                        all_metrics[(atk_base, def_base)].append(tuple(exp_metrics.tolist()))
                        continue

                    # Cases where one side is DO and the other is a baseline (or both baselines)
                    # For the "only_do_entry" mode we won't reach here because bases are restricted to DO only.
                    if def_base == "DO" and atk_base != "DO":
                        idx_att_fixed = att_index.get(atk_base, idx_att)
                        if idx_att_fixed is None:
                            logging.warning(
                                "[compute_tabular_results] Attacker baseline %r not found in att_index; using idx_att=%d",
                                atk_base, idx_att,
                            )
                            idx_att_fixed = idx_att
                        d_pay = float(p.dot(D_mat[:, idx_att_fixed]))
                        a_pay = float(A_mat[idx_att_fixed, :].dot(p))
                    elif def_base != "DO" and atk_base == "DO":
                        idx_def_fixed = def_index.get(def_base, idx_def)
                        if idx_def_fixed is None:
                            logging.warning(
                                "[compute_tabular_results] Defender baseline %r not found in def_index; using idx_def=%d",
                                def_base, idx_def,
                            )
                            idx_def_fixed = idx_def
                        d_pay = float(D_mat[idx_def_fixed, :].dot(q))
                        a_pay = float(q.dot(A_mat[:, idx_def_fixed]))
                    else:
                        # both non-DO: take entry directly
                        d_pay = float(D_mat[idx_def, idx_att])
                        a_pay = float(A_mat[idx_att, idx_def])

                    all_results[(atk_base, def_base)].append((d_pay, a_pay))

                    # ---- METRICS (always via simulate_game on concrete pair) ----
                    with timing("simulate_game[metrics]", TIMECHECK, sink=None):
                        d_sim, a_sim, *rest = oracle.simulate_game(def_strat, att_strat, num_simulations=1)
                    all_metrics[(atk_base, def_base)].append((d_sim, a_sim, *rest))

    # Finally produce the same three LaTeX outputs as before (they will be small if only_do_entry=True)
    return (
        generate_latex_table(
            attacker_bases,
            defender_bases,
            all_results, oracle.env.numOfDevice, payoff_type="attacker",
            seeds=tabular_seeds,
        ),
        generate_latex_table(
            attacker_bases,
            defender_bases,
            all_results, oracle.env.numOfDevice, payoff_type="defender",
            seeds=tabular_seeds,
        ),
        generate_extended_metrics_table(
            attacker_bases,
            defender_bases,
            all_metrics, oracle.env.numOfDevice,
            seeds=tabular_seeds,
        )
    )

# -------------------- State sampling --------------------
def sample_fixed_states(
    do,
    num_samples: int = 100,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    role: str = "attacker",
    use_random: bool = False,
    random_threshold: int = 500,   # auto-switch to random when many devices
) -> torch.Tensor:
    """
    Faster state sampler — optionally returns RANDOM states.

    Two modes:
      - If use_random==True OR the environment has many devices (numOfDevice >= random_threshold)
        we return `num_samples` *random* state vectors quickly.
      - Otherwise, we execute the original (more accurate) sampling which reuses env.restore/
        randomize/step for each sample.

    The random mode builds random tensors matching the shape of a single (role-specific) state.
    """

    # If huge net, prefer random states by default (fast)
    try:
        net_size = getattr(do, "env", None) and getattr(do.env, "numOfDevice", None)
    except Exception:
        net_size = None

    # If user explicitly requests random OR the net is large, use random shortcut
    if use_random or (net_size is not None and int(net_size) >= int(random_threshold)):
        # build one temp env just to get the state shape (cheap compared to stepping many times)
        env = do.fresh_env()
        env.time_budget_deadline = False
        env.tech = "DO"
        env.mode = role
        if role == "attacker":
            base_st = env._get_attacker_state()
        else:
            base_st = env._get_defender_state()
        dim = int(np.asarray(base_st).shape[0])
        # seed RNGs
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        # use normal or uniform — you can change to uniform if preferred
        states = torch.randn((int(num_samples), dim), dtype=torch.float32, device=device)
        return states

    # --- Original accurate sampling path (kept for smaller networks) ---
    if not (device.type == "cuda"):
        print("⚠️  Warning: sample_fixed_states is running on CPU — this will scale poorly with network size!")

    try:
        num_samples = int(math.ceil(float(num_samples)))
    except Exception as e:
        raise TypeError(f"num_samples must be an integer-like value, got {num_samples!r}") from e
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    if not getattr(do, "checkpoint", None):
        raise RuntimeError("sample_fixed_states: call do.checkpoint_now() first.")

    random.seed(seed); np.random.seed(seed)

    # 1) build ONE env and reuse it
    with timing("sample_fixed_states: fresh_env_once"):
        env = do.fresh_env()
        env.time_budget_deadline = False
    env.tech = "DO"
    env.mode = role

    states = []
    for _ in range(num_samples):
        # restore snapshot instead of reconstructing a new env
        with timing("sample_fixed_states: restore", sink=None):
            do.restore(env, reset_counters=True)
        # optional randomization if available
        if hasattr(env, "randomize_compromise_and_ownership"):
            with timing("sample_fixed_states: randomize", sink=None):
                env.randomize_compromise_and_ownership()
        # take a single step to get off the reset state (matches your original)
        with timing("sample_fixed_states: sample+step", sink=None):
            a = env.sample_action()
            action = a if len(a) == 4 else (a[0], a[1], a[2], 0)
            env.step(action)

        # read the role-specific observation and put it on device
        with timing("sample_fixed_states: to_tensor", sink=None):
            if role == "attacker":
                st = env._get_attacker_state()
            else:
                st = env._get_defender_state()
            states.append(torch.as_tensor(st, dtype=torch.float32, device=device))

    with timing("sample_fixed_states: stack", sink=None):
        return torch.stack(states, dim=0) if states else torch.empty((0,), device=device)


# -------------------- Diversity restart --------------------

def far_apart_ddpg_restart(
    init_ddpg_fn,
    saved_actor_dicts,
    device,
    fixed_states,
    sim_thresh: float = 0.1,
    max_restarts: int = 100,
    seed: int = None
):
    S = fixed_states.to(device)
    template = init_ddpg_fn()
    critic = template['critic'].to(device).eval()

    old_param_fps = []
    for sd in saved_actor_dicts:
        vec = torch.cat([p.flatten().to(device)
                         for p in sd.values() if isinstance(p, torch.Tensor)], dim=0)
        old_param_fps.append(F.normalize(vec, dim=0).unsqueeze(0))
    old_param_fps = torch.cat(old_param_fps, dim=0) if old_param_fps else torch.empty((0, 0), device=device)

    old_q_fps = []
    for sd in saved_actor_dicts:
        cand = init_ddpg_fn()
        actor_i = cand['actor'].to(device).eval()
        actor_i.load_state_dict(sd)
        with torch.no_grad():
            A_i = actor_i(S)
            Q_i = critic(S, A_i)
        old_q_fps.append(Q_i.mean(dim=0, keepdim=True))
    old_q_fps = torch.cat(old_q_fps, dim=0) if old_q_fps else torch.empty((0, 1), device=device)

    best_cand, best_score = None, float('inf')
    for i in range(max_restarts):
        if seed is not None:
            torch.manual_seed(seed + i); torch.cuda.manual_seed_all(seed + i)
            np.random.seed(seed + i); random.seed(seed + i)

        cand = init_ddpg_fn()
        actor = cand['actor'].to(device).eval()

        vec_new = torch.cat([p.flatten()
                             for p in actor.state_dict().values()
                             if isinstance(p, torch.Tensor)], dim=0).to(device)
        fp_param_new = F.normalize(vec_new, dim=0).unsqueeze(0)

        with torch.no_grad():
            A_new = actor(S)
            Q_new = critic(S, A_new)
        fp_q_new = Q_new.mean(dim=0, keepdim=True)

        sim_param = (F.cosine_similarity(fp_param_new, old_param_fps, dim=1).max().item()
                     if old_param_fps.numel() else 0.0)
        sim_q = (F.cosine_similarity(fp_q_new, old_q_fps, dim=1).max().item()
                 if old_q_fps.numel() else 0.0)
        worst = max(sim_param, sim_q)

        if worst < sim_thresh:
            return cand
        if worst < best_score:
            best_score, best_cand = worst, cand

    return best_cand

# -------------------- Training vs fixed opponent --------------------

def test_fixed_ddpg_training(
    env,
    initial_snapshot,
    do: 'DoubleOracle',
    train_ddpg_agent,
    fixed_role: str,
    steps_per_episode: int,
    eval_episode_len: int,
    eval_episodes: int,
    eval_feq: int,
    *,
    ActorClass,                    # required kwarg
    gamma: float = 0.95,
    batch_size: int = 512,
    burn_in: int = 500,
    σ: float = .3,
    σ_min: float = 1e-5,
    load: bool = False,
    cord_ascen: bool = True,
    TIMECHECK: bool = False,
):
    D = env.Max_network_size
    E = env.MaxExploits
    A = env.get_num_app_indices()
    env.time_budget_deadline = False

    if fixed_role == "defender":
        saved_dicts, eq_probs = do.saved_defender_actors, do.defender_equilibrium
        get_fixed, get_learn = env._get_defender_state, env._get_attacker_state
        n_fixed, n_learn = env.get_num_action_types("defender"), env.get_num_action_types("attacker")
    else:
        saved_dicts, eq_probs = do.saved_attacker_actors, do.attacker_equilibrium
        get_fixed, get_learn = env._get_attacker_state, env._get_defender_state
        n_fixed, n_learn = env.get_num_action_types("attacker"), env.get_num_action_types("defender")

    fixed_actors = []
    state_dim = get_fixed().shape[0]
    for sd in saved_dicts:
        action_dim = sd["fc3.weight"].shape[0]
        actor = ActorClass(state_dim, action_dim, do.seed, do.device)
        actor.load_state_dict(sd)
        actor.to(do.device).eval()
        fixed_actors.append(actor)

    train_actor   = train_ddpg_agent['actor']
    train_critic  = train_ddpg_agent['critic']
    target_actor  = train_ddpg_agent['target_actor']
    target_critic = train_ddpg_agent['target_critic']
    actor_opt     = train_ddpg_agent['actor_optimizer']
    critic_opt    = train_ddpg_agent['critic_optimizer']
    buffer        = train_ddpg_agent['replay_buffer']
    buffer.buffer.clear()

    decay_rate = (σ_min / σ) ** (1.0 / steps_per_episode)
    noise_std  = σ

    eval_steps, eval_rewards = [], []
    test_interval = eval_feq

    for step in range(steps_per_episode):
        turn = 'defender' if (step % 2 == 0) else 'attacker'
        env.mode = turn
        
        if turn == fixed_role:
            state, n_types = get_fixed(), n_fixed
        else:
            state, n_types = get_learn(), n_learn

        if turn == fixed_role:
            idx = np.random.choice(len(fixed_actors), p=eq_probs)
            with torch.no_grad():
                vec = fixed_actors[idx](torch.tensor(state, dtype=torch.float32)
                                        .unsqueeze(0).to(do.device)).cpu().numpy()[0]
            action_disc = do.decode_action(vec, n_fixed, D, E, A)
            action_vec = None
        else:
            st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(do.device)
            with torch.no_grad():
                raw = train_actor(st_t).detach().cpu().numpy()[0]
            noise = np.random.normal(0.0, noise_std, size=raw.shape)
            action_vec = np.clip(raw + noise, -1.0, +1.0)
            noise_std = max(σ_min, noise_std * decay_rate)
            action_disc = do.decode_action(action_vec, n_learn, D, E, A)

        _, r, done, *_ = env.step(action_disc)

        if turn != fixed_role:
            if not cord_ascen:
                next_state = get_learn()
                buffer.push(state, action_vec, r, next_state, done)
                if step > burn_in:
                    with timing("train_ddpg[raw]", TIMECHECK, sink=None):
                        train_ddpg(train_actor, train_critic, target_actor, target_critic,
                                   buffer, actor_opt, critic_opt, batch_size, gamma, do.device)
            else:
                next_state = get_learn()
                disc_vec = do.encode_action(
                    action_disc, n_learn, env.Max_network_size, env.MaxExploits, env.get_num_app_indices()
                )
                buffer.push(state, disc_vec, r, next_state, done)
                if step > burn_in:
                    with timing("train_ddpg[disc]", TIMECHECK, sink=None):
                        train_ddpg(train_actor, train_critic, target_actor, target_critic,
                                   buffer, actor_opt, critic_opt, batch_size, gamma, do.device)

        if step % test_interval == 0 and step > burn_in:
            tot_r = 0.0
            for _ in range(eval_episodes):
                # replace deepcopy of snapshot (very heavy) with fresh_env() + restore behavior
                # fresh_env() restores snapshot into do.env and returns it cheaply
                ev = do.fresh_env()
                ev.tech, ev.mode, ev.step_num = "DO", None, 0

                ep_r = 0.0
                dyn_eps = .99
                for t in range(eval_episode_len):
                    tev = 'defender' if (t % 2 == 0) else 'attacker'
                    ev.mode = tev

                    if tev == fixed_role:
                        idx2 = np.random.choice(len(fixed_actors), p=eq_probs)
                        with torch.no_grad():
                            vec2 = fixed_actors[idx2](
                                torch.tensor(get_fixed(), dtype=torch.float32)
                                .unsqueeze(0).to(do.device)
                            ).cpu().numpy()[0]
                        act = do.decode_action(vec2, n_fixed, D, E, A)
                    else:
                        st_e = get_learn()
                        st_te = torch.tensor(st_e, dtype=torch.float32).unsqueeze(0).to(do.device)
                        with torch.no_grad():
                            raw2 = train_actor(st_te).detach().cpu().numpy()[0]
                        if do.dynamic_neighbor_search and np.random.random() < dyn_eps:
                            act = do.dynamic_neighborhood_search(
                                state_tensor=st_te, raw_action=raw2, actor=train_actor, critic=train_critic,
                                k_init=3, beta_init=.05, c_k=1.0, c_beta=0.2
                            )
                            dyn_eps /= 2
                        else:
                            act = do.decode_action(raw2, n_learn, D, E, A)

                    _, r2, d2, *_ = ev.step(act)
                    if tev != fixed_role:
                        ep_r += r2
                    if d2:
                        break

                tot_r += ep_r

            avg_r = tot_r / eval_episodes
            eval_steps.append(step)
            eval_rewards.append(avg_r)
            print(f"[EVAL @ step {step:7d}] avg_reward = {avg_r:.3f}")

    return eval_steps, eval_rewards
