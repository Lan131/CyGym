
import gym
import matplotlib.pyplot as plt
import imageio
import igraph as ig
import random
import numpy as np
from collections import deque, defaultdict
from gym import spaces
from CyberDefenseEnv import CyberDefenseEnv
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv
import pickle
from CDSimulatorComponents import App, Device, OperatingSystem, Workload, os_encoder, Exploit
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import csv
import copy
import logging
from do_agent import DoubleOracle, train_ddpg, Actor, Strategy
import torch
import time
import os
from tqdm import trange
from utils import *                    # set_seed, sample_fixed_states, far_apart_ddpg_restart, compute_tabular_results, etc.
from utils import timing               # explicit import for timing ctx manager
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

# -------------------------
# Memory instrumentation (tracemalloc + psutil) -- added
# -------------------------
import tracemalloc
import psutil
import gc
import time

# start tracemalloc with a modest frame depth
tracemalloc.start(25)
_proc = psutil.Process()



# ---------- snapshot + clone helpers (place after imports) ----------
import types
from typing import Any

class SnapshotHolder:
    """
    Lightweight holder that stores pickled env bytes and implements __deepcopy__.
    When copy.deepcopy() is called on this object it returns a freshly unpickled env.
    This avoids deep-copying the large in-memory object graph.
    """
    def __init__(self, pickled_bytes: bytes):
        self._bytes = pickled_bytes

    def clone(self):
        """Return a fresh env object by unpickling stored bytes."""
        return pickle.loads(self._bytes)

    def __deepcopy__(self, memo):
        # Called by copy.deepcopy(snapshot_holder)
        return self.clone()

    def __getstate__(self):
        # allow pickling the holder itself (rare) — store bytes
        return {"_bytes": self._bytes}

    def __setstate__(self, state):
        self._bytes = state["_bytes"]


def clone_state_dict_tensors(sd: dict) -> dict:
    """
    Clone a PyTorch state_dict at tensor level (cheap compared to deepcopy of whole container).
    Accepts a dict of tensors and returns a cloned dict where each value is cloned+detached.
    """
    if sd is None:
        return None
    cloned = {}
    for k, v in sd.items():
        try:
            # v is a tensor-like
            cloned[k] = v.clone().detach()
        except Exception:
            # fallback for non-tensor values (ints, floats, nested dicts)
            # shallow copy is acceptable for small metadata
            cloned[k] = copy.deepcopy(v)
    return cloned


def clone_ddpg_from_template(double_oracle: Any, template_ddpg: dict, state_dim: int, action_dim: int):
    """
    Construct a fresh ddpg dict using the oracle's init_ddpg, then copy actor/critic weights
    from template_ddpg via state_dict. Does NOT deep-copy replay buffers or optimizer state.
    """
    # create a fresh ddpg structure via the same factory — this gives fresh weights/optimizers
    new_ddpg = double_oracle.init_ddpg(state_dim, action_dim)

    # Load actor / critic weights if available in template
    try:
        if 'actor' in template_ddpg and 'actor' in new_ddpg:
            new_ddpg['actor'].load_state_dict(template_ddpg['actor'].state_dict())
    except Exception:
        # if template stored actor as state_dict already:
        if 'actor' in template_ddpg and isinstance(template_ddpg['actor'], dict):
            new_ddpg['actor'].load_state_dict(template_ddpg['actor'])

    try:
        if 'critic' in template_ddpg and 'critic' in new_ddpg:
            new_ddpg['critic'].load_state_dict(template_ddpg['critic'].state_dict())
    except Exception:
        if 'critic' in template_ddpg and isinstance(template_ddpg['critic'], dict):
            new_ddpg['critic'].load_state_dict(template_ddpg['critic'])

    # Note: we intentionally do NOT copy replay buffer / optimizer state to avoid
    # duplicating huge buffers. If you need them, implement a small shallow copy.
    return new_ddpg




def print_rss(prefix=""):
    """Print current Resident Set Size (RSS) in MB with a timestamp."""
    try:
        rss_mb = _proc.memory_info().rss / 1024**2
    except Exception:
        rss_mb = 0.0
    print(f"[MEM] {time.strftime('%H:%M:%S')} {prefix} RSS={rss_mb:.2f}MB")
    return rss_mb

def snapshot_tracemalloc(tag: str, top_n: int = 20, dump_file: str = None):
    """
    Take a tracemalloc snapshot, print the top_n allocations by line,
    and optionally dump the snapshot to a file for offline inspection.
    Returns the snapshot object.
    """
    gc.collect()
    snap = tracemalloc.take_snapshot()
    if dump_file:
        try:
            snap.dump(dump_file)
            print(f"[TRACEMALLOC] dumped snapshot -> {dump_file}")
        except Exception as e:
            print(f"[TRACEMALLOC] failed to dump snapshot: {e}")
    stats = snap.statistics('lineno')
    print(f"[TRACEMALLOC] Snapshot '{tag}' top {min(top_n, len(stats))} (by size):")
    for i, stat in enumerate(stats[:top_n], 1):
        frame = stat.traceback[0]
        print(f" {i:2d}. {frame.filename}:{frame.lineno} "
              f"{stat.size / 1024:.1f} KiB  ({stat.count} objs)  {stat}")
    return snap

def compare_snapshots(snap_after, snap_before, top_n: int = 10):
    """Compare two snapshots and print top_n diffs (after - before)."""
    try:
        diffs = snap_after.compare_to(snap_before, 'lineno')
        print(f"[TRACEMALLOC] Top {min(top_n, len(diffs))} diffs (after - before):")
        for d in diffs[:top_n]:
            frame = d.traceback[0]
            print(f"  {frame.filename}:{frame.lineno}  size_diff={d.size_diff/1024:.1f} KiB  count_diff={d.count_diff}")
    except Exception as e:
        print(f"[TRACEMALLOC] compare failed: {e}")

# -------------------------
# --- global timing flag & print helper (overridden after argparse in __main__) ---
# -------------------------
TIMECHECK = (os.getenv("TIMECHECK", "0") == "1")

tprint = print if TIMECHECK else (lambda *a, **k: None)

def sample_defender_states(env, num_samples, seed, device):
    random.seed(seed); np.random.seed(seed)
    states = []
    for _ in range(num_samples):
        double_oracle.restore(env, reset_counters=True)
        if hasattr(env, "randomize_compromise_and_ownership"):
            env.randomize_compromise_and_ownership()
        a = env.sample_action()
        env.step(a if len(a)==4 else (*a,0))
        st = env._get_defender_state()
        states.append(torch.tensor(st, dtype=torch.float32, device=device))
    return torch.stack(states, dim=0)
# -------------------------
# Main run_game (unchanged except added memory snapshots around heavy ops)
# -------------------------
def run_game(env,
             initial_env_snapshot,
             DO_iterations,
             test_episodes,
             steps_per_episode,
             seed,
             baseline,
             output_dir,
             do_DO,
             experiment_all,
             min_DO_iters,
             fixed_test_eps_length,
             experiment_num,
             dyn,
             BR_type,
             tab_results,
             tabular_sims,
             info_path,
             zero_day,
             prune,
             zero_day_submartingale,
             far_apart_restart = False,
             time_budget_days = 30):
    """
    Overall orchestration of Double-Oracle training, dynamic testing,
    and fixed-role post-training experiments using the FULL equilibrium mixture.
    """





    # ── Setup ──────────────────────────────────────────────────────────────
    dyn_def_hist, dyn_att_hist = [], []
    payoff_history = []

    # ── Histories for fixed-role experiments ───────────────────────────────
    fix_att_hist_pretrain   = []
    fix_att_hist_rand_init  = []
    fix_att_hist_preset     = []
    fix_att_hist_nodef      = []
    fix_def_hist_pretrain   = []
    fix_att_hist_rand       = []
    fix_att_hist_noattk     = []
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    # create the oracle
    double_oracle = DoubleOracle(env, 1, steps_per_episode, seed, baseline, dyn, BR_type, zero_day)
    
    # Capture a deep-copied RAM snapshot the DO will use to restore envs
    with timing("DO.checkpoint_now (guarded)", enabled=TIMECHECK):
        # make sure the env's simulator/state exists before checkpointing
        try:
            if hasattr(env, "initialize_environment"):
                env.initialize_environment()
        except Exception:
            # ignore — we'll let DoubleOracle handle it, but prefer env initialized
            pass
        try:
            double_oracle.checkpoint_now()
        except RuntimeError as e:
            # Known benign failure: env not initialized yet (simulator/state missing).
            # Continue — we'll proceed without a snapshot instead of crashing.
            if "env not initialized" in str(e):
                print(f"   skipped ({e})")
            else:
                raise


    if zero_day:
        double_oracle.N_MC = 5

    if zero_day_submartingale:
        double_oracle.defender_strategies    = []
        double_oracle.saved_defender_actors  = []
        double_oracle.saved_defender_critics = []
        double_oracle.attacker_strategies    = []
        double_oracle.saved_attacker_actors  = []
        double_oracle.saved_attacker_critics = []

    def_sdim = env._get_defender_state().shape[0]
    att_sdim = env._get_attacker_state().shape[0]
    n_def    = env.get_num_action_types(mode="defender")
    n_att    = env.get_num_action_types(mode="attacker")

    # ── 1) grab the DO “template” actor (untrained) ───────────────────
    with timing("DO.init_ddpg(template_def)", enabled=TIMECHECK):
        template_def_ddpg = double_oracle.init_ddpg(def_sdim, n_def)
    with timing("DO.init_ddpg(template_att)", enabled=TIMECHECK):
        template_att_ddpg = double_oracle.init_ddpg(att_sdim, n_att)
    def_dict = template_def_ddpg['actor'].state_dict()
    att_dict = template_att_ddpg['actor'].state_dict()

    # ── 2) fresh RandomInit policy (different seed) ───────────────────
    set_seed(seed + 1234)
    with timing("DO.init_ddpg(random_def)", enabled=TIMECHECK):
        random_def_actor = double_oracle.init_ddpg(def_sdim, n_def)['actor'].state_dict()
    with timing("DO.init_ddpg(random_att)", enabled=TIMECHECK):
        random_att_actor = double_oracle.init_ddpg(att_sdim, n_att)['actor'].state_dict()

    random_def_strat = Strategy(random_def_actor, baseline_name="RandomInit")
    random_att_strat = Strategy(random_att_actor, baseline_name="RandomInit")

    if zero_day_submartingale:
        double_oracle.defender_strategies       = []
        double_oracle.saved_defender_actors     = []
        double_oracle.attacker_strategies       = []
        double_oracle.saved_attacker_actors     = []

    # ── 2a) inject non-parametric baselines ───────────────────────────
    double_oracle.defender_strategies = [
        Strategy(None, baseline_name="No Defense"),
        Strategy(None, baseline_name="Preset"),
        random_def_strat,
        *double_oracle.defender_strategies
    ]
    double_oracle.saved_defender_actors = [
        random_def_actor,
        *double_oracle.saved_defender_actors
    ]

    double_oracle.attacker_strategies = [
        Strategy(None, baseline_name="No Attack"),
        random_att_strat,
        *double_oracle.attacker_strategies
    ]
    double_oracle.saved_attacker_actors = [
        random_att_actor,
        *double_oracle.saved_attacker_actors
    ]

    # payoff matrix for baselines + any existing DO strategies
    # ------------------- instrumentation around the expensive call -------------------
    
    #print_rss("before initial build_payoff_matrices")
    #snap_before_init = snapshot_tracemalloc("before_build_initial", top_n=8)
    with timing("DO.build_payoff_matrices(initial)", enabled=TIMECHECK):
        D_mat, A_mat = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
        #D_mat, A_mat = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
    #print_rss("after initial build_payoff_matrices")
    #snap_after_init = snapshot_tracemalloc("after_build_initial", top_n=12,
    #                                       dump_file=f"snap_after_build_initial_{int(time.time())}.tracemalloc")
    #compare_snapshots(snap_after_init, snap_before_init, top_n=12)
    
    # -------------------------------------------------------------------------------

    double_oracle.payoff_matrix = D_mat.copy()

    # ── Snapshot & reset to the initial state ─────────────────────────
    with timing("pickle.dumps(env) -> initial_snapshot_bytes (SnapshotHolder)", enabled=TIMECHECK):
        # Keep an in-memory serialized representation; unpickle lazily when a fresh env is needed.
        initial_snapshot = SnapshotHolder(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
    with timing("env.reset(from_init=False)", enabled=TIMECHECK):
        env.reset(from_init=False)

    # ── prep save paths ────────────────────────────────────────────────
    save_dir = os.path.join(output_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    def_ckpt = os.path.join(save_dir, f"defender_mixture_seed{seed}.pt")
    att_ckpt = os.path.join(save_dir, f"attacker_mixture_seed{seed}.pt")

    if far_apart_restart:
        # once, before DO loop: fixed states for diversity checks
        try:
            with timing("sample_fixed_states(attacker)", enabled=TIMECHECK):
                fixed_states_att = sample_fixed_states(
                    do=double_oracle, num_samples=100, seed=seed, device=double_oracle.device, role="attacker"
                )
            with timing("sample_fixed_states(defender)", enabled=TIMECHECK):
                fixed_states_def = sample_fixed_states(
                    do=double_oracle, num_samples=100, seed=seed+1, device=double_oracle.device, role="defender"
                )
        except:
            # Silent fallback — continue run with dummy states
            try:
                adim = int(np.asarray(env._get_attacker_state()).shape[0])
            except:
                adim = 1
            try:
                ddim = int(np.asarray(env._get_defender_state()).shape[0])
            except:
                ddim = 1

            fixed_states_att = torch.zeros((100, adim), dtype=torch.float32, device=double_oracle.device)
            fixed_states_def = torch.zeros((100, ddim), dtype=torch.float32, device=double_oracle.device)
            pass

    if do_DO:
        # ── init parameters ────────────────────────────────────────────
        prev_eq_def  = None
        no_add_att   = 0
        no_add_def   = 0
        tol          = -5
        drop_tol     = 0.10
        min_rounds   = min_DO_iters

        def_hist = []
        att_hist = []
        att_restarted = False
        def_restarted = False
        if far_apart_restart:
            try:
                # (re-sample fixed states just in case)
                with timing("sample_fixed_states(attacker)[DO loop prep]", enabled=TIMECHECK):
                    fixed_states_att = sample_fixed_states(
                        do=double_oracle, num_samples=100, seed=seed, device=double_oracle.device, role="attacker"
                    )


                #print("checkpoint sample fix state start")
                with timing("sample_fixed_states(defender)[DO loop prep]", enabled=TIMECHECK):
                    fixed_states_def = sample_fixed_states(
                        do=double_oracle, num_samples=100, seed=seed+1, device=double_oracle.device, role="defender"
                    )
                #print("checkpoint sample fix state end")
            except:
                # Silent fallback — continue run with dummy states
                try:
                    adim = int(np.asarray(env._get_attacker_state()).shape[0])
                except:
                    adim = 1
                try:
                    ddim = int(np.asarray(env._get_defender_state()).shape[0])
                except:
                    ddim = 1

                fixed_states_att = torch.zeros((100, adim), dtype=torch.float32, device=double_oracle.device)
                fixed_states_def = torch.zeros((100, ddim), dtype=torch.float32, device=double_oracle.device)
                pass

        for ep in range(DO_iterations):
            #print("check point DO iter"+str(ep))
            with timing("env.reset(from_init=True)", enabled=TIMECHECK):
                env.reset(from_init=True)

            # ── 1) Solve restricted subgame ────────────────────────────
            # instrument each in-loop build_payoff_matrices call
            print_rss(f"before build_payoff_matrices (iter={ep})")
            #snap_b = snapshot_tracemalloc(f"before_build_iter{ep}", top_n=8)
            with timing("DO.build_payoff_matrices(D_full,A_full)", enabled=TIMECHECK):
                #D_full, A_full = double_oracle.build_payoff_matrices()
                D_full, A_full = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
            #print_rss(f"after build_payoff_matrices (iter={ep})")
            #snap_a = snapshot_tracemalloc(f"after_build_iter{ep}", top_n=10,
            #                              dump_file=f"snap_after_build_iter{ep}_{int(time.time())}.tracemalloc")
            #compare_snapshots(snap_a, snap_b, top_n=10)

            do_prune = bool(prune) and (ep >= 4)

            with timing(f"DO.solve_nash_equilibrium(prune={do_prune})", enabled=TIMECHECK):
                p, q = double_oracle.solve_nash_equilibrium(D_full, A_full, prune=do_prune)

            with timing("DO.build_payoff_matrices(D_mat,A_mat)[post-solve]", enabled=TIMECHECK):
                D_mat, A_mat = double_oracle.build_payoff_matrices()
                D_mat, A_mat = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
                

            assert D_mat.shape == (len(double_oracle.defender_strategies),
                                   len(double_oracle.attacker_strategies))
            assert A_mat.shape == (len(double_oracle.attacker_strategies),
                                   len(double_oracle.defender_strategies))

            def_names = []
            for i, strat in enumerate(double_oracle.defender_strategies):
                def_names.append(strat.baseline_name if strat.baseline_name is not None else f"DO#{i}")

            att_names = []
            for j, strat in enumerate(double_oracle.attacker_strategies):
                att_names.append(strat.baseline_name if strat.baseline_name is not None else f"DO#{j}")

            D_df  = pd.DataFrame(D_mat, index=def_names, columns=att_names)
            A_df  = pd.DataFrame(A_mat, index=att_names, columns=def_names)
            p_ser = pd.Series(p, index=def_names, name="Defender mix-prob")
            q_ser = pd.Series(q, index=att_names, name="Attacker mix-prob")

            tprint("=== Defender payoff matrix ==="); tprint(D_df, "\n")
            tprint("=== Attacker payoff matrix ==="); tprint(A_df, "\n")
            tprint("=== Defender equilibrium ===");   tprint(p_ser, "\n")
            tprint("=== Attacker equilibrium ===");   tprint(q_ser)

            with timing("eq_payoff calc", enabled=TIMECHECK):
                eq_def, eq_att = p.dot(D_mat).dot(q), q.dot(A_mat).dot(p)
            print(f"[DO] eq def payoff = {eq_def:.4f}, att payoff = {eq_att:.4f}")

            backup = {
                "def_strats": list(double_oracle.defender_strategies),
                "att_strats": list(double_oracle.attacker_strategies),
                "payoff_mat": double_oracle.payoff_matrix.copy(),
                "saved_def":  list(double_oracle.saved_defender_actors),
                "saved_att":  list(double_oracle.saved_attacker_actors),
            }
            def_pool = list(double_oracle.defender_strategies)
            att_pool = list(double_oracle.attacker_strategies)

            #print("Check point starting best BR")            
            # ── 2) Attacker BR ─────────────────────────────────────────
            if BR_type.lower() == "hierarchical":
                with timing("AttackerBR:import HierarchicalBR", enabled=TIMECHECK):
                    from hierarchical_br import HierarchicalBestResponse
                with timing("AttackerBR:Hierarchical.train", enabled=TIMECHECK):
                    new_att = HierarchicalBestResponse(double_oracle, "attacker").train(def_pool, p, T=15000)

            elif BR_type.lower() == "meta":
                with timing("AttackerBR:import MetaHierarchicalBR", enabled=TIMECHECK):
                    from meta_hierarchical_br import MetaHierarchicalBestResponse
                with timing("AttackerBR:MetaHierarchical.train", enabled=TIMECHECK):
                    new_att = MetaHierarchicalBestResponse(double_oracle, "attacker").train(def_pool, p, T=15000)

            elif BR_type.lower() == "hmarlexpert":
                with timing("AttackerBR:import HMARLExpertBR", enabled=TIMECHECK):
                    from HMARL import (
                        HMARLExpertBestResponse,
                        ExpertRuleMaster,
                        FrozenSubPolicy,
                    )
                    import torch.nn as nn

                # --- tiny stub network so FrozenSubPolicy has a policy_net to wrap ---
                class DummySubpolicyNet(nn.Module):
                    def __init__(self, obs_dim: int):
                        super().__init__()
                        # simple linear head, we don't actually decode its logits yet
                        self.fc = nn.Linear(obs_dim, 8)

                    def forward(self, x):
                        return self.fc(x)

                with timing("AttackerBR:HMARLExpert.init", enabled=TIMECHECK):
                    # attacker observation size
                    attacker_state_vec = double_oracle.env._get_attacker_state()
                    attacker_state_dim = attacker_state_vec.shape[0]

                    # build dummy nets for each attacker sub-skill
                    dummy_net_lowcost  = DummySubpolicyNet(attacker_state_dim)
                    dummy_net_highcost = DummySubpolicyNet(attacker_state_dim)

                    # ---- define which env action_types belong to which expert ----
                    # "cheap"/low-cost attacker behaviors (e.g. probe / noop)
                    lowcost_action_types = [2, 3]   # adjust if attacker action_type mapping differs
                    # "expensive"/high-cost attacker behaviors (e.g. lateral exploit / spread)
                    highcost_action_types = [1]     # attacker action_type 1 = compromise/spread in your env

                    subpolicies = [
                        FrozenSubPolicy(
                            policy_net=dummy_net_lowcost,
                            device=double_oracle.device,
                            name="LowCostInvestigate",
                            role="attacker",
                            allowed_action_types=lowcost_action_types,
                        ),
                        FrozenSubPolicy(
                            policy_net=dummy_net_highcost,
                            device=double_oracle.device,
                            name="HighCostExploit",
                            role="attacker",
                            allowed_action_types=highcost_action_types,
                        ),
                    ]

                    # Rule master for attacker using new ExpertRuleMaster API:
                    # cheaplocal_idx  -> choose probing / low-cost (skill 0)
                    # costlylocal_idx -> choose spreading / high-cost (skill 1)
                    # global_idx      -> we don't really have global attacker ops, so reuse 0
                    # global_prob     -> 0.0 so it never randomly flips
                    expert_master = ExpertRuleMaster(
                        cheaplocal_idx=0,
                        costlylocal_idx=1,
                        global_idx=0,
                        global_prob=0.0,
                    )

                    br_agent = HMARLExpertBestResponse(
                        oracle=double_oracle,
                        role="attacker",
                        subpolicies=subpolicies,
                        expert_master=expert_master,
                        device=double_oracle.device,
                    )

                with timing("AttackerBR:HMARLExpert.train(return_meta=True)", enabled=TIMECHECK):
                    new_att = br_agent.train(def_pool, p, T=15000, return_meta=True)

            elif BR_type.lower() == "hmarlmeta":
                with timing("AttackerBR:import HMARLMetaBR", enabled=TIMECHECK):
                    from HMARL import (
                        HMARLMetaBestResponse,
                        FrozenSubPolicy,
                    )
                    import torch.nn as nn

                class DummySubpolicyNet(nn.Module):
                    def __init__(self, obs_dim: int):
                        super().__init__()
                        self.fc = nn.Linear(obs_dim, 8)

                    def forward(self, x):
                        return self.fc(x)

                with timing("AttackerBR:HMARLMeta.init", enabled=TIMECHECK):
                    attacker_state_vec = double_oracle.env._get_attacker_state()
                    attacker_state_dim = attacker_state_vec.shape[0]

                    dummy_net_lowcost  = DummySubpolicyNet(attacker_state_dim)
                    dummy_net_highcost = DummySubpolicyNet(attacker_state_dim)

                    lowcost_action_types  = [2, 3]
                    highcost_action_types = [1]

                    subpolicies = [
                        FrozenSubPolicy(
                            policy_net=dummy_net_lowcost,
                            device=double_oracle.device,
                            name="LowCostExpert",
                            role="attacker",
                            allowed_action_types=lowcost_action_types,
                        ),
                        FrozenSubPolicy(
                            policy_net=dummy_net_highcost,
                            device=double_oracle.device,
                            name="HighCostExpert",
                            role="attacker",
                            allowed_action_types=highcost_action_types,
                        ),
                    ]

                    br_agent = HMARLMetaBestResponse(
                        oracle=double_oracle,
                        role="attacker",
                        subpolicies=subpolicies,
                        state_dim=attacker_state_dim,
                        device=double_oracle.device,
                    )

                with timing("AttackerBR:HMARLMeta.train(return_meta=True)", enabled=TIMECHECK):
                    new_att = br_agent.train(def_pool, p, T=15000, return_meta=True)

            elif BR_type.lower() == "mappo":
                with timing("AttackerBR:import MAPPO", enabled=TIMECHECK):
                    from MAPPO import MAPPOCommBestResponse

                with timing("AttackerBR:MAPPO.train", enabled=TIMECHECK):
                    new_att = MAPPOCommBestResponse(double_oracle, "attacker").train(
                        def_pool, p,
                        T=7500, rollout_len=1, ppo_epochs=1, minibatch_size=256,
                        budget_type="updates", budget=7500, single_update_per_rollout=True
                    )

            elif BR_type.lower() == "ippo":
                with timing("AttackerBR:import IPPO", enabled=TIMECHECK):
                    from IPPO import IPPOCommBestResponse
                with timing("AttackerBR:IPPO.train", enabled=TIMECHECK):
                    new_att = IPPOCommBestResponse(double_oracle, "attacker").train(
                        att_pool, q, T=15000,
                        budget_type="updates", budget=15000, single_update_per_rollout=True
                    )

            else:
                # DDPG-style BR
                with timing("AttackerBR:DDPG.best_response", enabled=TIMECHECK):
                    new_att = double_oracle.ddpg_best_response(
                        opponent_strategies=def_pool,
                        opponent_equilibrium=p,
                        role='attacker'
                    )


            if getattr(double_oracle.env, "time_budget_exceeded", False):
                print("time_budget_exceeded:"+str(double_oracle.env.time_budget_exceeded))
                print("Time budget exceeded — ending training early")
                if tab_results:

                    env.time_budget_seconds = None
                    env.time_budget_deadline = None
                    env.time_budget_exceeded = False

                    print("Starting Tabular Rollouts…")

                    if info_path is not None:
                        perform_tab_results(info_path,double_oracle.defender_ddpg,double_oracle.attacker_ddpg, TIMECHECK, double_oracle, seed, tabular_sims, steps_per_episode, output_dir ,BR_type , env)



            with timing("evaluate new_att vs def_pool", enabled=TIMECHECK):
                att_vs_new = np.array([
                    double_oracle.simulate_game(d, new_att, double_oracle.N_MC)[1]
                    for d in def_pool
                ])
                new_att_eq = att_vs_new.dot(p)
                imp_att    = new_att_eq - eq_att

            if imp_att > tol:
                double_oracle.attacker_strategies.append(new_att)
                actor_dict = getattr(new_att, "actor_state_dict", None)
                if actor_dict is None:
                    last = double_oracle.saved_attacker_actors[-1] if len(double_oracle.saved_attacker_actors) > 0 else None
                    actor_dict = clone_state_dict_tensors(last)
                double_oracle.saved_attacker_actors.append(actor_dict)
                double_oracle.saved_attacker_critics.append(getattr(new_att, "critic_state_dict", None))

                with timing("rebuild payoff after adding att BR", enabled=TIMECHECK):
                    #double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices()
                    double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)

                print(f" → Attacker BR accepted (Δ eq = +{imp_att:.4f})")
                no_add_att = 0
            else:
                print(f" → Attacker BR skipped (Δ eq = {imp_att:.4f} < tol={tol})")
                no_add_att += 1

                if far_apart_restart and (no_add_att >= 2 and not att_restarted and BR_type.lower() in ("cord_asc", "cord-asc", "cord asc")):
                    print("↺ Attacker stalled twice; running far_apart_ddpg_restart…")
                    init_fn = lambda: double_oracle.init_ddpg(
                        env._get_attacker_state().shape[0],
                        env.get_num_action_types("attacker")
                    )
                    with timing("far_apart_ddpg_restart(attacker)", enabled=TIMECHECK):
                        cand = far_apart_ddpg_restart(
                            init_ddpg_fn=init_fn,
                            saved_actor_dicts=double_oracle.saved_attacker_actors,
                            device=double_oracle.device,
                            fixed_states=fixed_states_att,
                            sim_thresh=0.1,
                            max_restarts=5,
                            seed=seed
                        )
                    s_dim = cand['actor'].fc1.in_features
                    a_dim = cand['actor'].fc3.out_features
                    strat = Strategy(
                        actor_state_dict=cand['actor'].state_dict(),
                        critic_state_dict=cand['critic'].state_dict(),
                        actor_dims=(s_dim, a_dim),
                        critic_dims=(s_dim, a_dim)
                    )
                    double_oracle.attacker_strategies.append(strat)
                    double_oracle.saved_attacker_actors.append(cand['actor'].state_dict())
                    no_add_att = 0
                    att_restarted = True

          
            # ── 3) Defender BR ─────────────────────────────────────────
            if BR_type.lower() == "hierarchical":
                with timing("DefenderBR:import HierarchicalBR", enabled=TIMECHECK):
                    from hierarchical_br import HierarchicalBestResponse
                with timing("DefenderBR:Hierarchical.train", enabled=TIMECHECK):
                    new_def = HierarchicalBestResponse(double_oracle, "defender").train(att_pool, q, T=15000)

            elif BR_type.lower() == "meta":
                with timing("DefenderBR:import MetaHierarchicalBR", enabled=TIMECHECK):
                    from meta_hierarchical_br import MetaHierarchicalBestResponse
                with timing("DefenderBR:MetaHierarchical.train", enabled=TIMECHECK):
                    new_def = MetaHierarchicalBestResponse(double_oracle, "defender").train(att_pool, q, T=15000)

            elif BR_type.lower() == "mappo":
                with timing("DefenderBR:import MAPPO", enabled=TIMECHECK):
                    from MAPPO import MAPPOCommBestResponse
                with timing("DefenderBR:MAPPO.train", enabled=TIMECHECK):
                    new_def = MAPPOCommBestResponse(double_oracle, "defender").train(
                        att_pool, q,
                        T=7500, rollout_len=1, ppo_epochs=1, minibatch_size=256,
                        budget_type="updates", budget=7500, single_update_per_rollout=True
                    )

            elif BR_type.lower() == "ippo":
                with timing("DefenderBR:import IPPO", enabled=TIMECHECK):
                    from IPPO import IPPOCommBestResponse
                with timing("DefenderBR:IPPO.train", enabled=TIMECHECK):
                    new_def = IPPOCommBestResponse(double_oracle, "defender").train(
                        att_pool, q,
                        T=15000, budget_type="updates", budget=15000, single_update_per_rollout=True
                    )

            else:
                # DDPG-style BR (default)
                with timing("DefenderBR:DDPG.best_response", enabled=TIMECHECK):
                    new_def = double_oracle.ddpg_best_response(
                        opponent_strategies=att_pool,
                        opponent_equilibrium=q,
                        role='defender'
                    )


            with timing("evaluate new_def vs att_pool", enabled=TIMECHECK):
                def_vs_new = np.array([
                    double_oracle.simulate_game(new_def, a, double_oracle.N_MC)[0]
                    for a in att_pool
                ])
                new_def_eq = def_vs_new.dot(q)
                imp_def    = new_def_eq - eq_def

            if imp_def > tol:
                double_oracle.defender_strategies.append(new_def)
                last = double_oracle.saved_defender_actors[-1]
                actor_dict = new_def.actor_state_dict or copy.deepcopy(last)
                double_oracle.saved_defender_actors.append(actor_dict)
                double_oracle.saved_defender_critics.append(new_def.critic_state_dict)
                with timing("rebuild payoff after adding def BR", enabled=TIMECHECK):
                    #double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices()
                    double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
                print(f" → Defender BR accepted (Δ eq = +{imp_def:.4f})")
                no_add_def = 0
            else:
                print(f" → Defender BR skipped (Δ eq = {imp_def:.4f} < tol={tol})")
                no_add_def += 1
                if far_apart_restart and no_add_def >= 2 and not def_restarted and not BR_type.lower() == "hierarchical":
                    print("↺ Defender stalled twice; running far_apart_ddpg_restart…")
                    init_fn = lambda: double_oracle.init_ddpg(
                        env._get_defender_state().shape[0],
                        env.get_num_action_types("defender")
                    )
                    with timing("far_apart_ddpg_restart(defender)", enabled=TIMECHECK):
                        cand = far_apart_ddpg_restart(
                            init_ddpg_fn=init_fn,
                            saved_actor_dicts=double_oracle.saved_defender_actors,
                            device=double_oracle.device,
                            fixed_states=fixed_states_def,
                            sim_thresh=0.1,
                            max_restarts=5,
                            seed=seed
                        )
                    s_dim  = cand['actor'].fc1.in_features
                    a_dim  = cand['actor'].fc3.out_features
                    strat = Strategy(
                        actor_state_dict=cand['actor'].state_dict(),
                        critic_state_dict=cand['critic'].state_dict(),
                        actor_dims=(s_dim, a_dim),
                        critic_dims=(s_dim, a_dim)
                    )
                    double_oracle.defender_strategies.append(strat)
                    double_oracle.saved_defender_actors.append(cand['actor'].state_dict())
                    no_add_def = 0
                    def_restarted = True

            # ── 4) Re-solve on expanded game ───────────────────────────
            with timing("DO.build_payoff_matrices(D2,A2)", enabled=TIMECHECK):
                #D2, A2 = double_oracle.build_payoff_matrices()
                D2, A2 = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
            with timing("DO.solve_nash_equilibrium (final of iter)", enabled=TIMECHECK):
                p2, q2 = double_oracle.solve_nash_equilibrium(D2, A2, prune=False)

            print(" # defender_strategies:", len(double_oracle.defender_strategies))
            print(" # attacker_strategies:", len(double_oracle.attacker_strategies))
            print(" payoff matrix D2 shape:", D2.shape, " A2 shape:", A2.shape)
            print(" p2 length:", p2.shape, " q2 length:", q2.shape)
            print("p2: "+str(p2))
            print("q2: "+str(q2))

            with timing("eq_payoff calc (p2,q2)", enabled=TIMECHECK):
                eq_def2 = p2.dot(D2).dot(q2)
                eq_att2 = q2.dot(A2).dot(p2)
            print(f"[DO] new eq def payoff = {eq_def2:.4f}, att payoff = {eq_att2:.4f}")
            def_hist.append(eq_def2)
            att_hist.append(eq_att2)

            # ── 5 & 6) Convergence ─────────────────────────────────────
            if ep + 1 >= min_rounds and no_add_att >= 2 and no_add_def >= 2:
                if np.all(p2[0:3] == 0) and np.all(q2[0:2] == 0):
                    print(f"✅ Converged after {ep+1} iterations.")
                else:
                    print("↻ Equilibrium still playing a pure baseline; continuing DO search")
                    continue

                with timing("finalize equilibrium vectors", enabled=TIMECHECK):
                    #D_mat, A_mat = double_oracle.build_payoff_matrices()
                    D_mat, A_mat = double_oracle.build_payoff_matrices(n_workers=os.cpu_count()-1)
                    double_oracle.payoff_matrix = D_mat.copy()
                    p_final, q_final = p2, q2
                    p_final = np.nan_to_num(p_final, nan=1.0/p_final.size)
                    q_final = np.nan_to_num(q_final, nan=1.0/q_final.size)
                    p_final /= p_final.sum()
                    q_final /= q_final.sum()
                    double_oracle.defender_equilibrium = p_final
                    double_oracle.attacker_equilibrium = q_final
                    double_oracle.saved_defender_actors = double_oracle.saved_defender_actors
                    double_oracle.saved_attacker_actors = double_oracle.saved_attacker_actors
                break

            # ── Intermediate diagnostics every 10 iters ─────────────────
            if (ep+1) % 10 == 0:
                its = np.arange(1, len(def_hist)+1)

                with timing("plot/save defender history", enabled=TIMECHECK):
                    plt.figure(figsize=(6,4))
                    plt.plot(its, def_hist, marker=',')
                    plt.xlabel('DO Iteration'); plt.ylabel('Defender Eq Payoff')
                    plt.title(f'Seed={seed}')
                    plt.grid(True); plt.tight_layout()
                    fn_def = os.path.join(output_dir, f'DO_def_payoff_seed{seed}_iter{ep+1}.png')
                    plt.savefig(fn_def); plt.close()
                    tprint(f"  Saved defender plot → {fn_def}")

                with timing("plot/save attacker history", enabled=TIMECHECK):
                    plt.figure(figsize=(6,4))
                    plt.plot(its, att_hist, marker=',')
                    plt.xlabel('DO Iteration'); plt.ylabel('Attacker Eq Payoff')
                    plt.title(f'Seed={seed}')
                    plt.grid(True); plt.tight_layout()
                    fn_att = os.path.join(output_dir, f'DO_att_payoff_seed{seed}_iter{ep+1}.png')
                    plt.savefig(fn_att); plt.close()
                    tprint(f"  Saved attacker plot → {fn_att}")

                with timing("plot/save average history", enabled=TIMECHECK):
                    avg = 0.5*(np.array(def_hist)+np.array(att_hist))
                    plt.figure(figsize=(6,4))
                    plt.plot(its, avg, marker=',')
                    plt.xlabel('DO Iteration'); plt.ylabel('Average Eq Payoff')
                    plt.title(f'Seed={seed} – up to iter={ep+1}')
                    plt.grid(True); plt.tight_layout()
                    fn_avg = os.path.join(output_dir, f'DO_avg_payoff_seed{seed}_iter{ep+1}.png')
                    plt.savefig(fn_avg); plt.close()
                    tprint(f"  Saved average plot → {fn_avg}")

        # ── Final plots ─────────────────────────────────────────────────
        its = np.arange(1, len(def_hist)+1)

        with timing("final plot/save defender", enabled=TIMECHECK):
            plt.figure(figsize=(6,4))
            plt.plot(its, def_hist, marker=',', label='Defender')
            plt.xlabel('DO Iteration'); plt.ylabel('Eq Payoff')
            plt.title(f'Final Defender Payoff')
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'DO_def_payoff_seed{seed}.png'))
            plt.close()

        with timing("final plot/save attacker", enabled=TIMECHECK):
            plt.figure(figsize=(6,4))
            plt.plot(its, att_hist, marker=',', label='Attacker')
            plt.xlabel('DO Iteration'); plt.ylabel('Eq Payoff')
            plt.title(f'Final Attacker Payoff')
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'DO_att_payoff_seed{seed}.png'))
            plt.close()

        with timing("final plot/save expected", enabled=TIMECHECK):
            avg_hist = 0.5*(np.array(def_hist)+np.array(att_hist))
            plt.figure(figsize=(6,4))
            plt.plot(its, avg_hist, marker=',', label='Expected EQ Payoff')
            plt.xlabel('DO Iteration'); plt.ylabel('Average Eq Payoff')
            plt.title(f'Expected DO Payoff')
            plt.grid(True); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'DO_expected_payoff_seed{seed}.png'))
            plt.close()

        # save mixtures
        with timing("torch.save(def_ckpt)", enabled=TIMECHECK):
            torch.save({
                "actor_state_dicts": double_oracle.saved_defender_actors,
                "equilibrium":       double_oracle.defender_equilibrium,
                "labels":            def_names,
            }, def_ckpt)
        with timing("torch.save(att_ckpt)", enabled=TIMECHECK):
            torch.save({
                "actor_state_dicts": double_oracle.saved_attacker_actors,
                "equilibrium":       double_oracle.attacker_equilibrium,
                "labels":            att_names,
            }, att_ckpt)
        print(f"🔒 Saved DO mixtures to {def_ckpt} & {att_ckpt}")

        with timing("DO.checkpoint_now(final)", enabled=TIMECHECK):
            double_oracle.checkpoint_now()

    else:
        # load or solve one-shot
        try:
            with timing("torch.load(def_ckpt)", enabled=TIMECHECK):
                data = torch.load(def_ckpt, weights_only=False)
            double_oracle.saved_defender_actors = data["actor_state_dicts"]
            double_oracle.defender_equilibrium  = np.array(data["equilibrium"])

            with timing("torch.load(att_ckpt)", enabled=TIMECHECK):
                data = torch.load(att_ckpt, weights_only=False)
            double_oracle.saved_attacker_actors = data["actor_state_dicts"]
            double_oracle.attacker_equilibrium  = np.array(data["equilibrium"])
            print(f"🔄 Loaded DO mixtures from {def_ckpt} & {att_ckpt}")

            oracle_path = os.path.join(save_dir, f"oracle_seed{seed}.pkl")
            with timing("pickle.dump(double_oracle)", enabled=TIMECHECK):
                with open(oracle_path, "wb") as f:
                    pickle.dump(double_oracle, f, protocol=pickle.HIGHEST_PROTOCOL)

        except FileNotFoundError:
            with timing("solve one-shot Nash (no ckpt)", enabled=TIMECHECK):
                p, q = double_oracle.solve_nash_equilibrium(D_mat, A_mat, prune=False)
                double_oracle.defender_equilibrium = p
                double_oracle.attacker_equilibrium = q
            print("⚠️  No saved mixtures; solved one-shot Nash instead.")

    if tab_results:

        env.time_budget_seconds = TIME_BUDGET_SECONDS
        env.time_budget_deadline = RUN_DEADLINE
        env.time_budget_exceeded = False

        print("Starting Tabular Rollouts…")
        perform_tab_results(info_path,double_oracle.defender_ddpg,double_oracle.attacker_ddpg, TIMECHECK, double_oracle, seed, tabular_sims, steps_per_episode, output_dir ,BR_type, env )


    # ── Fixed-role experiments after convergence ──────────────────────────
    if baseline == "Nash":

        if os.path.isfile(def_ckpt):
            with timing("torch.load(def_ckpt)[fixed-role]", enabled=TIMECHECK):
                data = torch.load(def_ckpt, weights_only=False)
            sd_list = data["actor_state_dicts"]; eq_list = data["equilibrium"]
            filtered = [(sd, p) for sd, p in zip(sd_list, eq_list) if sd is not None]
            if filtered:
                sds, probs = zip(*filtered)
                double_oracle.saved_defender_actors = list(sds)
                double_oracle.defender_equilibrium  = np.array(probs)
            else:
                double_oracle.saved_defender_actors = []
                double_oracle.defender_equilibrium  = np.array([])
            print(f"🔄 Loaded defender equilibrium mix from {def_ckpt}")

        if os.path.isfile(att_ckpt):
            with timing("torch.load(att_ckpt)[fixed-role]", enabled=TIMECHECK):
                data = torch.load(att_ckpt, weights_only=False)
            sd_list = data["actor_state_dicts"]; eq_list = data["equilibrium"]
            filtered = [(sd, p) for sd, p in zip(sd_list, eq_list) if sd is not None]
            if filtered:
                sds, probs = zip(*filtered)
                double_oracle.saved_attacker_actors = list(sds)
                double_oracle.attacker_equilibrium  = np.array(probs)
            else:
                double_oracle.saved_attacker_actors = []
                double_oracle.attacker_equilibrium  = np.array([])
            print(f"🔄 Loaded attacker equilibrium mix from {att_ckpt}")

        # strip Nones & renormalize
        def_sds, def_ps = [], []
        for sd, p in zip(double_oracle.saved_defender_actors, double_oracle.defender_equilibrium):
            if sd is not None:
                def_sds.append(sd); def_ps.append(p)
        if not def_sds:
            raise RuntimeError("No valid defender actors loaded!")
        def_ps = np.array(def_ps); def_ps = def_ps / def_ps.sum()
        double_oracle.saved_defender_actors = def_sds
        double_oracle.defender_equilibrium  = def_ps

        att_sds, att_ps = [], []
        for sd, p in zip(double_oracle.saved_attacker_actors, double_oracle.attacker_equilibrium):
            if sd is not None:
                att_sds.append(sd); att_ps.append(p)
        if not att_sds:
            raise RuntimeError("No valid attacker actors loaded!")
        att_ps = np.array(att_ps); att_ps = att_ps / att_ps.sum()
        double_oracle.saved_attacker_actors = att_sds
        double_oracle.attacker_equilibrium  = att_ps

        # Precompute dims for random DDPG in every experiment:
        att_state_dim = env._get_attacker_state().shape[0]
        n_att_types   = env.get_num_action_types(mode="attacker")
        def_state_dim = env._get_defender_state().shape[0]
        n_def_types   = env.get_num_action_types(mode="defender")

        feq = 50
        trials = 3

        # 0) Attacker learns vs fixed DO-defender
        if experiment_num == 0 or experiment_all:
            print("attacker learns vs fixed DO defender")
            with timing("clone template_att_ddpg via state_dict", enabled=TIMECHECK):
                rand_att_ddpg = clone_ddpg_from_template(double_oracle, template_att_ddpg, att_state_dim, n_att_types)
            with timing("test_fixed_ddpg_training(attacker vs DO-def)", enabled=TIMECHECK):
                eval_x_att, eval_y_att = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_att_ddpg,
                    fixed_role="defender",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_pretrain.append(list(zip(eval_x_att, eval_y_att)))
            env.base_line = baseline

        # 1) Attacker learns vs fixed RandomInit defender
        if experiment_num == 1 or experiment_all:
            print("attacker learns vs fixed random defender")
            with timing("clone template_att_ddpg via state_dict", enabled=TIMECHECK):
                rand_att_ddpg = clone_ddpg_from_template(double_oracle, template_att_ddpg, att_state_dim, n_att_types)

            double_oracle.saved_defender_actors = [random_def_actor]
            double_oracle.defender_equilibrium  = [1.0]
            double_oracle.baseline              = "RandomInit"
            env.base_line                       = "RandomInit"
            with timing("test_fixed_ddpg_training(attacker vs RandomInit-def)", enabled=TIMECHECK):
                eval_x_rand, eval_y_rand = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_att_ddpg,
                    fixed_role="defender",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_rand_init.append(list(zip(eval_x_rand, eval_y_rand)))
            double_oracle.baseline = None
            env.base_line         = baseline

        # 2) Attacker vs Preset defender
        if experiment_num == 2 or experiment_all:
            print("attacker learns vs fixed preset defender")
            with timing("clone template_att_ddpg via state_dict", enabled=TIMECHECK):
                rand_att_ddpg = clone_ddpg_from_template(double_oracle, template_att_ddpg, att_state_dim, n_att_types)

            with timing("env.reset + randomize", enabled=TIMECHECK):
                env.reset(from_init=True)
                env.randomize_compromise_and_ownership()
            env.base_line = "Preset"
            with timing("test_fixed_ddpg_training(attacker vs Preset)", enabled=TIMECHECK):
                eval_x_p, eval_y_p = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_att_ddpg,
                    fixed_role="defender",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_preset.append(list(zip(eval_x_p, eval_y_p)))

        # 3) Attacker vs No-Defense defender
        if experiment_num == 3 or experiment_all:
            print("attacker learns vs fixed do-nothing defender")
            with timing("clone template_att_ddpg via state_dict", enabled=TIMECHECK):
                rand_att_ddpg = clone_ddpg_from_template(double_oracle, template_att_ddpg, att_state_dim, n_att_types)

            with timing("env.reset + randomize", enabled=TIMECHECK):
                env.reset(from_init=True)
                env.randomize_compromise_and_ownership()
            env.base_line = "No Defense"
            with timing("test_fixed_ddpg_training(attacker vs NoDefense)", enabled=TIMECHECK):
                eval_x_n, eval_y_n = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_att_ddpg,
                    fixed_role="defender",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_nodef.append(list(zip(eval_x_n, eval_y_n)))
            env.base_line = "Nash"

        # 4) Defender learns vs fixed DO-attacker
        if experiment_num == 4 or experiment_all:
            print("Defender learns vs fixed DO obtained Attacker")
            with timing("clone template_def_ddpg via state_dict", enabled=TIMECHECK):
                rand_def_ddpg = clone_ddpg_from_template(double_oracle, template_def_ddpg, def_state_dim, n_def_types)

            with timing("test_fixed_ddpg_training(def vs DO-att)", enabled=TIMECHECK):
                eval_x_def, eval_y_def = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_def_ddpg,
                    fixed_role="attacker",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_def_hist_pretrain.append(list(zip(eval_x_def, eval_y_def)))

        # 5) Defender learns vs fixed RandomInit attacker
        if experiment_num == 5 or experiment_all:
            print("Defender learns vs fixed Random Attacker")
            double_oracle.saved_attacker_actors = [random_att_actor]
            double_oracle.attacker_equilibrium  = [1.0]
            double_oracle.baseline               = "RandomInit"
            env.base_line                        = "RandomInit"
            with timing("clone template_def_ddpg via state_dict", enabled=TIMECHECK):
                rand_def_ddpg = clone_ddpg_from_template(double_oracle, template_def_ddpg, def_state_dim, n_def_types)

            with timing("test_fixed_ddpg_training(def vs RandomInit-att)", enabled=TIMECHECK):
                eval_x_ra, eval_y_ra = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_def_ddpg,
                    fixed_role="attacker",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_rand = list(zip(eval_x_ra, eval_y_ra))
            double_oracle.baseline = None
            env.base_line         = baseline

        # 6) Defender learns vs No-Attack attacker
        if experiment_num == 6 or experiment_all:
            print("Defender learns vs fixed Do nothing Attacker")
            env.base_line = "No Attack"
            with timing("clone template_def_ddpg via state_dict", enabled=TIMECHECK):
                rand_def_ddpg = clone_ddpg_from_template(double_oracle, template_def_ddpg, def_state_dim, n_def_types)

            with timing("test_fixed_ddpg_training(def vs NoAttack)", enabled=TIMECHECK):
                eval_x_dn, eval_y_dn = test_fixed_ddpg_training(
                    env, initial_env_snapshot,
                    do=double_oracle,
                    train_ddpg_agent=rand_def_ddpg,
                    fixed_role="attacker",
                    steps_per_episode=fixed_test_eps_length,
                    eval_episode_len=steps_per_episode,
                    eval_feq=feq,
                    eval_episodes=trials,
                    load=do_DO
                )
            fix_att_hist_noattk = list(zip(eval_x_dn, eval_y_dn))
            env.base_line = "Nash"

    return


# -------------------------
# Command-line entrypoint (same args as before)
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Cyber Defense simulation. (Instrumented)')
    parser.add_argument('--DO_iterations', type=int, default=10, help='Max training episodes (with early stopping)')
    parser.add_argument('--test_episodes', type=int, default=1, help='Number of fixed-role test episodes at each test interval')
    parser.add_argument('--steps_per_episode', type=int, default=100, help='Steps per episode')
    parser.add_argument('--fixed_test_eps_length', type=int, default=5500, help='Steps per episode')
    parser.add_argument('--num_of_device', type=int, default=10, help='Number of devices')
    parser.add_argument('--output_dir', type=str, default='plots', help='Where to save per-iteration and final plots')
    parser.add_argument('--experiment_num', type=int, default=0, help='experiment type')
    parser.add_argument('--experiment_all',action='store_true',help='Do all experiments at once')
    parser.add_argument('--do_DOAR',action='store_true',help='Do DO (vs loading from policy)')
    parser.add_argument('--min_DO_iters', type=int, default=1, help='minimum number of DO rounds')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--its', type=int, default=1, help='instance of network')
    parser.add_argument('--dynamic_search',action='store_true',help='Search large action spaces?')
    parser.add_argument('--tabular_results',action='store_true',help='Pairwise play')
    parser.add_argument('--BR_type', type=str, default='Cord_asc', help='Type of Best Response- ddpg or Coordinate Ascent (Cord_asc)')
    parser.add_argument('--tabular_sims', type=int, default=3,help='Number of roll-outs per pairing per seed when tabulating')
    parser.add_argument('--zero_day',action='store_true',help='Randomize over exploits (eg theres some zero)')
    parser.add_argument('--k_known', type=int, default=1, help='how many known')
    parser.add_argument('--j_private', type=int, default=2, help='how many unkonwn')
    parser.add_argument('--workscale', type=float, default=1.0, help='workscale')
    parser.add_argument('--defensive_scale', type=float, default=1.0, help='defensive_scale')
    parser.add_argument('--preknown',type=int,default=0, help="Number of zero-day exploits that the defender already ‘discovers’ at t=0")
    parser.add_argument('--Dz_size',type=int,default=4, help="Number of zero days in the universe plus other exploits")
    parser.add_argument('--prune',action='store_true',help='Prune dominated Strategies?')
    parser.add_argument('--zero_day_submartingale',action='store_true',help='Should increasing Dz increase attack coverage in expectation?')
    parser.add_argument('--max_Dz',type=int,default=6, help="Max number of Dz")
    parser.add_argument('--timecheck', action='store_true', help='Enable detailed timing to console')
    parser.add_argument('--time_budget_days', type=float, default=30, help='Stop training after N days (float). Set 0 to disable.')
    parser.add_argument('--alpha', type=float, default=1, help='alpha hyperparmeter for MetaDOAR')
    parser.add_argument('--khop', type=int, default=1, help='khop neighborhood invalidation hyperparmeter for MetaDOAR')

    args = parser.parse_args()
    
    try:
        mp.set_start_method("fork")
    except:
        print("Speed up methods require linux. This will be very slow if device count is ~10e5." )
    # pull args
    DO_iterations = args.DO_iterations
    test_eps = args.test_episodes
    steps = args.steps_per_episode
    num_dev = args.num_of_device
    output = args.output_dir
    do_DO = args.do_DOAR
    min_DO_iters = args.min_DO_iters
    fixed_test_eps_length = args.fixed_test_eps_length
    experiment_num = args.experiment_num
    experiment_all = args.experiment_all
    seed = args.seed
    its = args.its
    dyn = args.dynamic_search
    BR_type = args.BR_type
    tab_results = args.tabular_results
    tabular_sims = args.tabular_sims
    zero_day = args.zero_day
    workscale = args.workscale
    defensive_scale = args.defensive_scale
    k_known = args.k_known
    j_private = args.j_private
    preknown = args.preknown
    Dz_size = args.Dz_size
    prune = args.prune
    zero_day_submartingale = args.zero_day_submartingale
    max_Dz = args.max_Dz
    time_budget_days = args.time_budget_days
    alpha = args.alpha
    khop = args.khop

    # activate timing if flag set
    TIMECHECK = args.timecheck or (os.getenv("TIMECHECK", "0") == "1")
    tprint = print if TIMECHECK else (lambda *a, **k: None)

    print(f"DO iters:      {DO_iterations}\nTest episodes:    {test_eps}\n" +
          f"Steps per ep:     {steps}\nNum devices:      {num_dev}\nOutput dir:       {output}\n Seed:{seed}")
    print("Its: "+str(its))

    baselines = ["Nash"]
    set_seed(seed)

    for baseline in baselines:
        print(f"Running with seed={seed}, baseline={baseline}")

        snap_path = f"initial_net_DO_its{its}.pkl"
        if not os.path.isfile(snap_path):
            raise FileNotFoundError(f"Expected snapshot at {snap_path}")

        with timing(f"pickle.load({snap_path})", enabled=TIMECHECK):
            env: Volt_Typhoon_CyberDefenseEnv = pickle.load(open(snap_path, 'rb'))

        new_work_scale = workscale
        env.work_scale = new_work_scale
        env.def_scale = defensive_scale
        env.comp_scale = 30
        env.its = its
        env.preknown = args.preknown
        env.time_budget_exceeded = False
        env.alpha = alpha
  
        TIME_BUDGET_SECONDS = time_budget_days * 24.0 * 3600.0
        time_check = time.time()
        RUN_DEADLINE = time_check + TIME_BUDGET_SECONDS
        print("Run deadline:"+str(RUN_DEADLINE))
        print("current time:" +str(time_check))
        print("Seconds budget:"+str(TIME_BUDGET_SECONDS))




        # ----- zero-day configs / environment rebuilds (can be slow) -----
        if zero_day and j_private > 0:
            if not zero_day_submartingale:
                with timing("env.simulator.exploits.clear()", enabled=TIMECHECK):
                    env.simulator.exploits.clear()
                with timing("redeploy_apps_with_unique_vulns", enabled=TIMECHECK):
                    env.simulator.redeploy_apps_with_unique_vulns(
                        num_apps=10, vul_per_app=1, min_apps_per_device=1, max_apps_per_device=3
                    )
                with timing("generateExploits(random)", enabled=TIMECHECK):
                    env.simulator.generateExploits(
                        numOfExploits=Dz_size, addVul=True, minVulperExp=1, maxVulperExp=1, mode="random"
                    )
                all_ids = [e.id for e in env.simulator.exploits]
                needed = k_known + j_private
                if len(all_ids) < needed:
                    raise RuntimeError(f"Need at least {needed} exploits but only have {len(all_ids)}")
                random.seed(seed)
                known_ids   = random.sample(all_ids, k_known)
                remaining   = [eid for eid in all_ids if eid not in known_ids]
                private_ids = random.sample(remaining, j_private)
                unknown_pool = [eid for eid in remaining if eid not in private_ids]
            else:
                with timing("redeploy_apps_with_unique_vulns", enabled=TIMECHECK):
                    env.simulator.redeploy_apps_with_unique_vulns(
                        num_apps=10, vul_per_app=1, min_apps_per_device=1, max_apps_per_device=3
                    )
                all_devs = list(env.simulator.subnet.net.keys())
                DC = next((d for d, dev in env.simulator.subnet.net.items()
                           if getattr(dev, "device_type", None) == "DomainController"), None)
                if DC is None:
                    raise RuntimeError("No DomainController in the network!")
                others = [d for d in all_devs if d != DC]

                with timing("construct zero-day exploits (linear coverage)", enabled=TIMECHECK):
                    env.simulator.exploits.clear()
                    new_exploits = []
                    exploit_device_map = {}
                    n_targets = max(1, int(round(Dz_size * len(others) / float(max_Dz))))
                    for z_id in range(Dz_size):
                        chosen_others = random.sample(others, min(n_targets, len(others)))
                        chosen = chosen_others + [DC]
                        exploit_device_map[z_id] = set(chosen)
                        exp = Exploit(z_id, expType="zero-day")
                        targets = []
                        for d in chosen:
                            dev = env.simulator.subnet.net[d]
                            for app in dev.apps.values():
                                targets.extend(app.vulnerabilities.values())
                            targets.extend(dev.OS.vulnerabilities.values())
                        if not targets:
                            raise RuntimeError(f"No vulnerabilities on devices {chosen}")
                        exp.setTargetVul(targets)
                        new_exploits.append(exp)
                    env.simulator.exploits = new_exploits

                all_ids   = list(exploit_device_map.keys())
                random.seed(seed)
                known_ids = random.sample(all_ids, k_known)
                covered   = {DC}
                for eid in known_ids:
                    covered |= exploit_device_map[eid]
                candidates  = [eid for eid in all_ids if eid not in known_ids]
                private_ids = []
                for _ in range(j_private):
                    best_eid, best_gain = None, -1
                    for eid in candidates:
                        gain = len(exploit_device_map[eid] - covered)
                        if gain > best_gain:
                            best_eid, best_gain = eid, gain
                    if best_gain <= 0:
                        best_eid = random.choice(candidates)
                    private_ids.append(best_eid)
                    covered |= exploit_device_map[best_eid]
                    candidates.remove(best_eid)
                unknown_pool = candidates

                env.zero_day            = True
                env.k_known             = k_known
                env.j_private           = j_private
                env.common_exploit_ids  = known_ids
                env.private_exploit_ids = private_ids
                env.unknown_pool_ids    = unknown_pool

                print(f">>> ZERO-DAY (linear coverage): known={known_ids}, private={private_ids}, unknown={unknown_pool}")

        elif zero_day and j_private == 0:
            env.zero_day = False
            with timing("redeploy_apps_with_unique_vulns", enabled=TIMECHECK):
                env.simulator.redeploy_apps_with_unique_vulns(
                    num_apps=10, vul_per_app=1, min_apps_per_device=1, max_apps_per_device=3
                )
            with timing("generateExploits(1 random)", enabled=TIMECHECK):
                env.simulator.generateExploits(
                    numOfExploits=1, addVul=True, minVulperExp=1, maxVulperExp=1, mode="random"
                )
            env.k_known = 0; env.j_private = 0
            env.common_exploit_ids = []; env.private_exploit_ids = []; env.unknown_pool_ids = []
            with timing("env.initialize_environment()", enabled=TIMECHECK):
                env.initialize_environment()

        else:
            env.zero_day = False
            env.k_known = 1; env.j_private = 0
            env.common_exploit_ids = []; env.private_exploit_ids = []; env.unknown_pool_ids = []
            with timing("env.initialize_environment()", enabled=TIMECHECK):
                env.initialize_environment()

        print(f"[ABLATION] Overwrote work_scale → {env.work_scale}")
        print(f"[ABLATION] Overwrote def_scale → {env.def_scale}")
        print(f"[ABLATION] Overwrote num_dev → {env.numOfDevice}")
        print(f"[ABLATION] Overwrote alpha for METADOAR → {env.alpha}")



        TIME_BUDGET_SECONDS = time_budget_days * 24.0 * 3600.0
        now = time.time()
        if TIME_BUDGET_SECONDS > 0:
            RUN_DEADLINE = now + TIME_BUDGET_SECONDS
            env.time_budget_seconds = TIME_BUDGET_SECONDS
            env.time_budget_deadline = RUN_DEADLINE
        else:
            RUN_DEADLINE = None
            env.time_budget_seconds = 0.0
            env.time_budget_deadline = None
        env.time_budget_exceeded = False

        print(f"Run deadline: {RUN_DEADLINE}")
        print(f"current time: {now}")
        print(f"Seconds budget: {TIME_BUDGET_SECONDS}")

        with timing(f"pickle.dump(env -> {snap_path})", enabled=TIMECHECK):
            with open(snap_path, "wb") as f:
                pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[ABLATION] Overwrote snapshot file with new work_scale at {snap_path}")


        env.snapshot_path = snap_path

        env.base_line        = baseline
        env.tech             = "DO"
        env.mode             = "defender"
        env.Max_network_size = num_dev + 10
        
        
        
    with timing("pickle.dumps(env) -> initial_snapshot_bytes (SnapshotHolder)", enabled=TIMECHECK):
        # Keep an in-memory serialized representation; unpickle lazily when a fresh env is needed.
        initial_snapshot = SnapshotHolder(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL))
        

        info_path = os.path.join(output, "env_info.txt")
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        if not os.path.exists(info_path):
            with timing("write env header to info_path", enabled=TIMECHECK):
                with open(info_path, "a") as f:
                    f.write("=== ENVIRONMENT & DDPG HYPERPARAMS ===\n")
                    f.write(f"Number of Devices: {num_dev}\n")
                    f.write(f"Instance ID (from initial experiments): {its}\n")
                    f.write(f"Was Double Oracle run? (as opposed to loaded): {do_DO}\n")
                    f.write(f"DO Min Iterations: {min_DO_iters}\n")
                    f.write(f"DO Max Iterations: {DO_iterations}\n")
                    f.write(f"Steps Per Episode: {steps}\n")
                    f.write(f"Max_network_size: {env.Max_network_size}\n")
                    f.write(f"seed:              {seed}\n")
                    f.write(f"Zero Day:              {zero_day}\n")
                    if zero_day:
                        f.write(f"k_known:              {k_known}\n")
                        f.write(f"j_private:              {j_private}\n")
                    for attr in [
                        "defaultversion","default_mode","default_high",
                        "work_scale","comp_scale","num_attacker_owned",
                        "base_line","tech","its","intial_ratio_compromise",
                        "fast_scan","γ","def_scale"
                    ]:
                        f.write(f"{attr:<25}: {getattr(env, attr, None)}\n")
                    f.write(f"app_id_mapping     : {len(env.app_id_mapping)} entries\n")
                    f.write("\n")

        if info_path is not None:
            with timing("append DDPG hyperparams to info_path", enabled=TIMECHECK):
                with open(info_path, 'a') as f:
                    f.write("### DDPG hyperparameters ###\n")
                    f.write(f"reward_scale  = {1!r}\n")
                    f.write(f"max_grad_norm = {0.5!r}\n")
                    f.write(f"soft‐update τ = {1e-2!r}\n")
                    f.write("\n")
       
        with timing("run_game()", enabled=TIMECHECK):
            run_game(
                env, initial_snapshot, DO_iterations, test_eps, steps,
                seed, baseline, output, do_DO, experiment_all, min_DO_iters,
                fixed_test_eps_length, experiment_num, dyn, BR_type,
                tab_results, tabular_sims, info_path, zero_day, prune, zero_day_submartingale, False, time_budget_days
            )

    print("Simulation complete")
