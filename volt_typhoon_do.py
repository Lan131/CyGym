import gym
import matplotlib.pyplot as plt
import imageio
import igraph as ig
import random
import numpy as np
from collections import deque, defaultdict
from gym import spaces
from CyberDefenseEnv import CyberDefenseEnv, calculate_max_compromise_proportion
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
import os
from tqdm import trange
from utils import *
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')
import pandas as pd


print("Torch version in use: " + str(torch.__version__))
print("Cuda version in use: " + str(torch.version.cuda))


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
             zero_day_submartingale):
    """
    Overall orchestration of Double-Oracle training, dynamic testing,
    and fixed-role post-training experiments using the FULL equilibrium mixture.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    # ── Histories for plotting ─────────────────────────────────────────────
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
    template_def_ddpg = double_oracle.init_ddpg(def_sdim, n_def)
    template_att_ddpg = double_oracle.init_ddpg(att_sdim, n_att)
    def_dict = template_def_ddpg['actor'].state_dict()
    att_dict = template_att_ddpg['actor'].state_dict()

    # ── 2) now generate a truly *fresh* RandomInit policy ──────────────
    #     (reset the RNG so it doesn’t match your template exactly)
    set_seed(seed + 1234)
    random_def_actor = double_oracle.init_ddpg(def_sdim, n_def)['actor'].state_dict()
    random_att_actor = double_oracle.init_ddpg(att_sdim, n_att)['actor'].state_dict()

    random_def_strat = Strategy(random_def_actor, baseline_name="RandomInit")
    random_att_strat = Strategy(random_att_actor, baseline_name="RandomInit")

    if zero_day_submartingale:
        double_oracle.defender_strategies       = []
        double_oracle.saved_defender_actors     = []
        double_oracle.attacker_strategies       = []
        double_oracle.saved_attacker_actors     = [] 

   # ── 2a) inject our three “non-parametric” baselines at the front ────────────
    double_oracle.defender_strategies = [
        Strategy(None,            baseline_name="No Defense"),
        Strategy(None,            baseline_name="Preset"),
        random_def_strat,
        *double_oracle.defender_strategies
    ]
    double_oracle.saved_defender_actors = [
        random_def_actor,
        *double_oracle.saved_defender_actors
    ]
 
    double_oracle.attacker_strategies = [
        Strategy(None,            baseline_name="No Attack"),
        random_att_strat,
        *double_oracle.attacker_strategies
    ]
    double_oracle.saved_attacker_actors = [
        random_att_actor,
        *double_oracle.saved_attacker_actors
    ]

    # now fill in the true payoff matrix for those baselines + any DO strategies
    D_mat, A_mat = double_oracle.build_payoff_matrices()
    double_oracle.payoff_matrix = D_mat.copy()

    # ── Histories for plotting ─────────────────────────────────────────────
    dyn_def_hist, dyn_att_hist = [], []
    payoff_history = []

    # ── Snapshot & reset to the initial state ─────────────────────────────
    initial_snapshot = copy.deepcopy(env)
    env.reset(from_init=False)

    # ── Double-Oracle training ─────────────────────────────────────────────
    save_dir = os.path.join(output_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    # ← define these up front so both branches can see them
    def_ckpt = os.path.join(save_dir, f"defender_mixture_seed{seed}.pt")
    att_ckpt = os.path.join(save_dir, f"attacker_mixture_seed{seed}.pt")

    # once, before the DO loop:
    fixed_states_att = sample_fixed_states(
        env=initial_snapshot,
        num_samples=100,
        seed=seed,
        device=double_oracle.device
    )
    fixed_states_def = sample_fixed_states(
        env=initial_snapshot,
        num_samples=100,
        seed=seed+1,
        device=double_oracle.device
    )

    if do_DO:
        # ── init parameters ────────────────────────────────────────────────
        prev_eq_def  = None
        no_add_att   = 0
        no_add_def   = 0
        tol          = -5      # min improvement to accept a BR
        drop_tol     = 0.10       # >10% drop → revert
        min_rounds   = min_DO_iters

        # histories for plotting
        def_hist = []
        att_hist = []
        restarted = False
        att_restarted = False
        def_restarted = False

        fixed_states_att = sample_fixed_states(
            env=initial_snapshot,
            num_samples=100,
            seed=seed,
            device=double_oracle.device
        )
        # a tiny helper that does the same but uses defender-view states:
        def sample_defender_states(env, num_samples, seed, device):
            random.seed(seed); np.random.seed(seed)
            states = []
            for _ in range(num_samples):
                env.reset(from_init=True)
                if hasattr(env, "randomize_compromise_and_ownership"):
                    env.randomize_compromise_and_ownership()
                a = env.sample_action()
                # step once
                env.step(a if len(a)==4 else (*a,0))
                st = env._get_defender_state()
                states.append(torch.tensor(st, dtype=torch.float32, device=device))
            return torch.stack(states, dim=0)

        fixed_states_def = sample_defender_states(
            env=initial_snapshot,
            num_samples=100,
            seed=seed+1,
            device=double_oracle.device
        )
        
        for ep in range(DO_iterations):
            env.reset(from_init=True)

            # ── 1) Solve restricted subgame ────────────────────────────────
            if not prune:
                D_mat, A_mat  = double_oracle.build_payoff_matrices()
                p, q          = double_oracle.solve_nash_equilibrium(D_mat, A_mat,prune=prune)
            else:
                    # ── 1) Solve restricted subgame ────────────────────────────────
                    # build the full payoff matrices
                    D_full, A_full = double_oracle.build_payoff_matrices()
                    # solve (possibly prune) on those
                    p, q = double_oracle.solve_nash_equilibrium(D_full, A_full, prune=prune)
                
                    # if pruning actually happened, use the pruned versions
                    if prune and (double_oracle.payoff_matrix.shape != D_full.shape):
                        D_mat = double_oracle.payoff_matrix
                        A_mat = double_oracle.attacker_payoff_matrix
                    else:
                        D_mat = D_full
                        A_mat = A_full

            def_names = []
            for i, strat in enumerate(double_oracle.defender_strategies):
                if strat.baseline_name is not None:
                    def_names.append(strat.baseline_name)
                else:
                    def_names.append(f"DO#{i}")

            att_names = []
            for j, strat in enumerate(double_oracle.attacker_strategies):
                if strat.baseline_name is not None:
                    att_names.append(strat.baseline_name)
                else:
                    att_names.append(f"DO#{j}")

            # build labeled DataFrames / Series
            D_df  = pd.DataFrame(D_mat, index=def_names, columns=att_names)
            A_df  = pd.DataFrame(A_mat, index=att_names, columns=def_names)
            p_ser = pd.Series(p, index=def_names, name="Defender mix-prob")
            q_ser = pd.Series(q, index=att_names, name="Attacker mix-prob")

            print("=== Defender payoff matrix ===")
            print(D_df, "\n")

            print("=== Attacker payoff matrix ===")
            print(A_df, "\n")

            print("=== Defender equilibrium ===")
            print(p_ser, "\n")

            print("=== Attacker equilibrium ===")
            print(q_ser)

            eq_def, eq_att = p.dot(D_mat).dot(q), q.dot(A_mat).dot(p)
            print(f"[DO] eq def payoff = {eq_def:.4f}, att payoff = {eq_att:.4f}")

            # ── Backup current pools & matrix ─────────────────────────────
            backup = {
                "def_strats": list(double_oracle.defender_strategies),
                "att_strats": list(double_oracle.attacker_strategies),
                "payoff_mat": double_oracle.payoff_matrix.copy(),
                "saved_def":  list(double_oracle.saved_defender_actors),
                "saved_att":  list(double_oracle.saved_attacker_actors),
            }
            def_pool = list(double_oracle.defender_strategies)
            att_pool = list(double_oracle.attacker_strategies)

            # ── 2) Attacker best-response ──────────────────────────────────
            new_att = double_oracle.ddpg_best_response(
                opponent_strategies  = def_pool,
                opponent_equilibrium = p,
                role                 = 'attacker'
            )
            att_vs_new = np.array([
                double_oracle.simulate_game(d, new_att, double_oracle.N_MC)[1]
                for d in def_pool
            ])
            new_att_eq = att_vs_new.dot(p)
            imp_att    = new_att_eq - eq_att

            if imp_att > tol:
                double_oracle.attacker_strategies.append(new_att)
                last = double_oracle.saved_attacker_actors[-1]
                actor_dict = new_att.actor_state_dict or copy.deepcopy(last)
                double_oracle.saved_attacker_actors.append(actor_dict)
                double_oracle.saved_attacker_critics.append(new_att.critic_state_dict)
                double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices()

                print(f" → Attacker BR accepted (Δ eq = +{imp_att:.4f})")
                no_add_att = 0
            else:
                print(f" → Attacker BR skipped (Δ eq = {imp_att:.4f} < tol={tol})")
                no_add_att += 1

            if imp_att > tol:
                no_add_att = 0
            else:
                no_add_att += 1
                # if stalled twice and not yet restarted → do one far-apart restart
                if no_add_att >= 2 and not att_restarted:
                    print("↺ Attacker stalled twice; running far_apart_ddpg_restart…")
                    init_fn = lambda: double_oracle.init_ddpg(
                        env._get_attacker_state().shape[0],
                        env.get_num_action_types("attacker")
                    )
                    cand = far_apart_ddpg_restart(
                        init_ddpg_fn       = init_fn,
                        saved_actor_dicts  = double_oracle.saved_attacker_actors,
                        device             = double_oracle.device,
                        fixed_states       = fixed_states_att,
                        sim_thresh         = 0.1,
                        max_restarts       = 50,
                        seed               = seed
                    )
                    # inject into DO pool
                    strat = Strategy(cand['actor'].state_dict())
                    double_oracle.attacker_strategies.append(strat)
                    double_oracle.saved_attacker_actors.append(cand['actor'].state_dict())
                    no_add_att = 0
                    att_restarted = True

            # ── 3) Defender best-response ──────────────────────────────────
            new_def = double_oracle.ddpg_best_response(
                opponent_strategies  = att_pool,
                opponent_equilibrium = q,
                role                 = 'defender'
            )
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
                double_oracle.payoff_matrix, _ = double_oracle.build_payoff_matrices()

                print(f" → Defender BR accepted (Δ eq = +{imp_def:.4f})")
                no_add_def = 0
            else:
                print(f" → Defender BR skipped (Δ eq = {imp_def:.4f} < tol={tol})")
                no_add_def += 1

            if imp_def > tol:
                no_add_def = 0
            else:
                no_add_def += 1
                if no_add_def >= 2 and not def_restarted:
                    print("↺ Defender stalled twice; running far_apart_ddpg_restart…")
                    init_fn = lambda: double_oracle.init_ddpg(
                        env._get_defender_state().shape[0],
                        env.get_num_action_types("defender")
                    )
                    cand = far_apart_ddpg_restart(
                        init_ddpg_fn       = init_fn,
                        saved_actor_dicts  = double_oracle.saved_defender_actors,
                        device             = double_oracle.device,
                        fixed_states       = fixed_states_def,
                        sim_thresh         = 0.1,
                        max_restarts       = 50,
                        seed               = seed
                    )
                    strat = Strategy(cand['actor'].state_dict())
                    double_oracle.defender_strategies.append(strat)
                    double_oracle.saved_defender_actors.append(cand['actor'].state_dict())
                    no_add_def = 0
                    def_restarted = True

            # ── 4) Re-solve on (possibly) expanded game ────────────────────
            D2, A2     = double_oracle.build_payoff_matrices()
            p2, q2     = double_oracle.solve_nash_equilibrium(D2, A2, prune=False)

            print(" # defender_strategies:", len(double_oracle.defender_strategies))
            print(" # attacker_strategies:", len(double_oracle.attacker_strategies))
            print(" payoff matrix D2 shape:", D2.shape, " A2 shape:", A2.shape)
            print(" p2 length:", p2.shape, " q2 length:", q2.shape)
            print("p2: "+str(p2))
            print("q2: "+str(q2))

            eq_def2    = p2.dot(D2).dot(q2)
            eq_att2    = q2.dot(A2).dot(p2)
            print(f"[DO] new eq def payoff = {eq_def2:.4f}, att payoff = {eq_att2:.4f}")
            def_hist.append(eq_def2)
            att_hist.append(eq_att2)

            # ── 5 & 6) Convergence: stop after two skips each ────────────────────

            if ep + 1 >= min_rounds and no_add_att >= 2 and no_add_def >= 2:
                # only converge if neither equilibrium mixes in the pure baseline
                # (baseline “No Defense” and “No Attack” are at index 0)
                if np.all(p2[0:3] == 0) and np.all(q2[0:2] == 0): #it can still discover these via the parametric policies but I want to avoid trival solutions
                    print(f"✅ Converged after {ep+1} iterations.")
                else:
                    print("↻ Equilibrium still playing a pure baseline; continuing DO search")
                    continue

                # rebuild & re-solve so your equilibrium vector lines up exactly
                # 1) Re‐build the full payoff matrices
                D_mat, A_mat = double_oracle.build_payoff_matrices()

                # 2) Store the defender’s payoff matrix if you still need it
                double_oracle.payoff_matrix = D_mat.copy()

                # 3) Call solve_nash_equilibrium with those two matrices
                p_final, q_final = p2,q2
                print("pfinal: "+str(p_final))
                print("qfinal: "+str(q_final))
                # catch any NaNs
                p_final = np.nan_to_num(p_final, nan=1.0/p_final.size)
                q_final = np.nan_to_num(q_final, nan=1.0/q_final.size)

                # renormalize
                p_final /= p_final.sum()
                q_final /= q_final.sum()

                double_oracle.defender_equilibrium = p_final
                double_oracle.attacker_equilibrium = q_final

                # now drop any None placeholders and renormalize
                double_oracle.saved_defender_actors, double_oracle.defender_equilibrium = \
                    double_oracle.saved_defender_actors, p_final
                double_oracle.saved_attacker_actors, double_oracle.attacker_equilibrium = \
                    double_oracle.saved_attacker_actors, q_final

                break

            # ── Intermediate diagnostics every 10 iters ──────────────────
            if (ep+1) % 10 == 0:
                its = np.arange(1, len(def_hist)+1)

                # Defender payoff history
                plt.figure(figsize=(6,4))
                plt.plot(its, def_hist, marker=',')
                plt.xlabel('DO Iteration'); plt.ylabel('Defender Eq Payoff')
                plt.title(f'Seed={seed}')
                plt.grid(True); plt.tight_layout()
                fn_def = os.path.join(output_dir, f'DO_def_payoff_seed{seed}_iter{ep+1}.png')
                plt.savefig(fn_def); plt.close()
                print(f"  Saved defender plot → {fn_def}")

                # Attacker payoff history
                plt.figure(figsize=(6,4))
                plt.plot(its, att_hist, marker=',')
                plt.xlabel('DO Iteration'); plt.ylabel('Attacker Eq Payoff')
                plt.title(f'Seed={seed}')
                plt.grid(True); plt.tight_layout()
                fn_att = os.path.join(output_dir, f'DO_att_payoff_seed{seed}_iter{ep+1}.png')
                plt.savefig(fn_att); plt.close()
                print(f"  Saved attacker plot → {fn_att}")

                # Expected (average) payoff history
                avg = 0.5*(np.array(def_hist)+np.array(att_hist))
                plt.figure(figsize=(6,4))
                plt.plot(its, avg, marker=',')
                plt.xlabel('DO Iteration'); plt.ylabel('Average Eq Payoff')
                plt.title(f'Seed={seed} – up to iter={ep+1}')
                plt.grid(True); plt.tight_layout()
                fn_avg = os.path.join(output_dir, f'DO_avg_payoff_seed{seed}_iter{ep+1}.png')
                plt.savefig(fn_avg); plt.close()
                print(f"  Saved average plot → {fn_avg}")

        # ── Final plots ────────────────────────────────────────────────────
        its = np.arange(1, len(def_hist)+1)

        # Defender
        plt.figure(figsize=(6,4))
        plt.plot(its, def_hist, marker=',', label='Defender')
        plt.xlabel('DO Iteration'); plt.ylabel('Eq Payoff')
        plt.title(f'Final Defender Payoff')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'DO_def_payoff_seed{seed}.png'))
        plt.close()

        # Attacker
        plt.figure(figsize=(6,4))
        plt.plot(its, att_hist, marker=',', label='Attacker')
        plt.xlabel('DO Iteration'); plt.ylabel('Eq Payoff')
        plt.title(f'Final Attacker Payoff')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'DO_att_payoff_seed{seed}.png'))
        plt.close()

        # Expected (average)
        avg_hist = 0.5*(np.array(def_hist)+np.array(att_hist))
        plt.figure(figsize=(6,4))
        plt.plot(its, avg_hist, marker=',', label='Expected EQ Payoff')
        plt.xlabel('DO Iteration'); plt.ylabel('Average Eq Payoff')
        plt.title(f'Expected DO Payoff')
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'DO_expected_payoff_seed{seed}.png'))
        plt.close()

        # ── ensure we actually have a latest equilibrium on the full pool ───────
        # 1) Re‐build the full payoff matrices
        #D_mat, A_mat = double_oracle.build_payoff_matrices()

        # 2) Store the defender’s payoff matrix if you still need it
        #double_oracle.payoff_matrix = D_mat.copy()

        # 3) Call solve_nash_equilibrium with those two matrices
        #p_final, q_final = double_oracle.solve_nash_equilibrium(D_mat, A_mat, prune=False)
        #double_oracle.defender_equilibrium = p_final
        #double_oracle.attacker_equilibrium = q_final

        torch.save({
            "actor_state_dicts": double_oracle.saved_defender_actors,
            "equilibrium":       double_oracle.defender_equilibrium,
            "labels":            def_names,
        }, def_ckpt)
        torch.save({
            "actor_state_dicts": double_oracle.saved_attacker_actors,
            "equilibrium":       double_oracle.attacker_equilibrium,
            "labels":            att_names,
        }, att_ckpt)
        print(f"🔒 Saved DO mixtures to {def_ckpt} & {att_ckpt}")

        oracle_path = os.path.join(save_dir, f"oracle_seed{seed}.pkl")
        with open(oracle_path, "wb") as f:
            pickle.dump(double_oracle, f)

    else:
        # immediately load them (or solve a fresh Nash):
        try:
            data = torch.load(def_ckpt, weights_only=False)
            double_oracle.saved_defender_actors = data["actor_state_dicts"]
            double_oracle.defender_equilibrium  = np.array(data["equilibrium"])
            data = torch.load(att_ckpt, weights_only=False)
            double_oracle.saved_attacker_actors = data["actor_state_dicts"]
            double_oracle.attacker_equilibrium  = np.array(data["equilibrium"])
            print(f"🔄 Loaded DO mixtures from {def_ckpt} & {att_ckpt}")
            oracle_path = os.path.join(save_dir, f"oracle_seed{seed}.pkl")
            with open(oracle_path, "wb") as f:
                pickle.dump(double_oracle, f)

        except FileNotFoundError:
            # no checkpoint found: just solve a one-shot Nash
            p, q = double_oracle.solve_nash_equilibrium(D_mat, A_mat, prune=False)
            double_oracle.defender_equilibrium = p
            double_oracle.attacker_equilibrium = q
            print("⚠️  No saved mixtures; solved one-shot Nash instead.")

    if tab_results:
        print("Starting Tabular Rollouts…")

        with open(info_path, "a") as f:
            for side, ddpg in [("defender", double_oracle.defender_ddpg),
                                ("attacker", double_oracle.attacker_ddpg)]:
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

        atk_latex, def_latex, extended_metrics = compute_tabular_results(
             snapshot_path     = info_path,
             tabular_seeds     = [seed],
             tabular_sims      = tabular_sims,
             steps_per_episode = steps_per_episode,
             save_dir          = os.path.join(output_dir, "models"),   # ← point at “<output>/models”
             BR_type           = BR_type
        )
        # append LaTeX tables to same info file
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

    # ── Fixed-role experiments after convergence ──────────────────────────────
    if baseline == "Nash":

        if os.path.isfile(def_ckpt):
            data = torch.load(def_ckpt, weights_only=False)

            # strip out any None placeholders
            sd_list = data["actor_state_dicts"]
            eq_list = data["equilibrium"]
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
            data = torch.load(att_ckpt, weights_only=False)

            sd_list = data["actor_state_dicts"]
            eq_list = data["equilibrium"]
            filtered = [(sd, p) for sd, p in zip(sd_list, eq_list) if sd is not None]
            if filtered:
                sds, probs = zip(*filtered)
                double_oracle.saved_attacker_actors = list(sds)
                double_oracle.attacker_equilibrium  = np.array(probs)
            else:
                double_oracle.saved_attacker_actors = []
                double_oracle.attacker_equilibrium  = np.array([])
            print(f"🔄 Loaded attacker equilibrium mix from {att_ckpt}")

        # ── Strip out those None placeholders & re-normalize the probs ───────
        # Defender
        def_sds, def_ps = [], []
        for sd, p in zip(double_oracle.saved_defender_actors,
                        double_oracle.defender_equilibrium):
            if sd is not None:
                def_sds.append(sd)
                def_ps.append(p)
        if not def_sds:
            raise RuntimeError("No valid defender actors loaded!")
        def_ps = np.array(def_ps)
        def_ps = def_ps / def_ps.sum()
        double_oracle.saved_defender_actors = def_sds
        double_oracle.defender_equilibrium  = def_ps

        # Attacker
        att_sds, att_ps = [], []
        for sd, p in zip(double_oracle.saved_attacker_actors,
                        double_oracle.attacker_equilibrium):
            if sd is not None:
                att_sds.append(sd)
                att_ps.append(p)
        if not att_sds:
            raise RuntimeError("No valid attacker actors loaded!")
        att_ps = np.array(att_ps)
        att_ps = att_ps / att_ps.sum()
        double_oracle.saved_attacker_actors = att_sds
        double_oracle.attacker_equilibrium  = att_ps

        # Precompute dims for random DDPG in every experiment:
        att_state_dim = env._get_attacker_state().shape[0]
        n_att_types   = env.get_num_action_types(mode="attacker")
        def_state_dim = env._get_defender_state().shape[0]
        n_def_types   = env.get_num_action_types(mode="defender")

        # ── Experiment 0: Attacker learns vs fixed DO-defender ─────────────
        feq = 50
        trials = 3
        if experiment_num == 0 or experiment_all:
            print("attacker learns vs fixed DO defender")
            # 1) create fresh random attacker
            rand_att_ddpg = copy.deepcopy(template_att_ddpg)
            # 2) train this random attacker vs the fixed DO-mixture defender
            eval_x_att, eval_y_att = test_fixed_ddpg_training(
                env, initial_snapshot,
                do=double_oracle,
                train_ddpg_agent=rand_att_ddpg,   # ← random start
                fixed_role="defender",
                steps_per_episode=fixed_test_eps_length,
                eval_episode_len=steps_per_episode,
                eval_feq=feq,
                eval_episodes=trials,
                load=do_DO
            )
            fix_att_hist_pretrain.append(list(zip(eval_x_att, eval_y_att)))
            env.base_line = baseline

        # ── Experiment 1: Attacker learns vs fixed random defender ─────────
        if experiment_num == 1 or experiment_all:
            print("attacker learns vs fixed random defender")
            rand_att_ddpg = copy.deepcopy(template_att_ddpg)
            # — override DO’s pool with *your* saved RandomInit actor —
            double_oracle.saved_defender_actors = [random_def_actor]
            double_oracle.defender_equilibrium  = [1.0]
            double_oracle.baseline              = "RandomInit"
            env.base_line                       = "RandomInit"

            eval_x_rand, eval_y_rand = test_fixed_ddpg_training(
                env, initial_snapshot,
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

            # restore baseline pointer if you need DO’s mixture later
            double_oracle.baseline = None
            env.base_line         = baseline

        # ── Experiment 2 & 3: Preset & No-Defense ─────────────────────────
        if experiment_num == 2 or experiment_all:
            print("attacker learns vs fixed preset defender")
            rand_att_ddpg = double_oracle.init_ddpg(att_state_dim, n_att_types)
            env.reset(from_init=True)
            env.randomize_compromise_and_ownership()

            env.base_line = "Preset"
            eval_x_p, eval_y_p = test_fixed_ddpg_training(
                env, initial_snapshot,
                do=double_oracle,
                train_ddpg_agent=rand_att_ddpg,  # ← random start
                fixed_role="defender",
                steps_per_episode=fixed_test_eps_length,
                eval_episode_len=steps_per_episode,
                eval_feq=feq,
                eval_episodes=trials,
                load=do_DO
            )
            fix_att_hist_preset.append(list(zip(eval_x_p, eval_y_p)))

        if experiment_num == 3 or experiment_all:
            print("attacker learns vs fixed do-nothing defender")
            rand_att_ddpg = double_oracle.init_ddpg(att_state_dim, n_att_types)
            env.reset(from_init=True)
            env.randomize_compromise_and_ownership()

            env.base_line = "No Defense"
            eval_x_n, eval_y_n = test_fixed_ddpg_training(
                env, initial_snapshot,
                do=double_oracle,
                train_ddpg_agent=rand_att_ddpg,  # ← random start
                fixed_role="defender",
                steps_per_episode=fixed_test_eps_length,
                eval_episode_len=steps_per_episode,
                eval_feq=feq,
                eval_episodes=trials,
                load=do_DO
            )
            fix_att_hist_nodef.append(list(zip(eval_x_n, eval_y_n)))
            env.base_line = "Nash"

        # ── Experiment 4: Defender learns vs fixed DO-attacker ────────────
        if experiment_num == 4 or experiment_all:
            print("Defender learns vs fixed DO obtained Attacker")
            rand_def_ddpg = copy.deepcopy(template_def_ddpg)
            eval_x_def, eval_y_def = test_fixed_ddpg_training(
                env, initial_snapshot,
                do=double_oracle,
                train_ddpg_agent=rand_def_ddpg,   # ← random start
                fixed_role="attacker",
                steps_per_episode=fixed_test_eps_length,
                eval_episode_len=steps_per_episode,
                eval_feq=feq,
                eval_episodes=trials,
                load=do_DO
            )
            fix_def_hist_pretrain.append(list(zip(eval_x_def, eval_y_def)))

        # ── Experiment 5: Defender learns vs fixed Random Attacker ───────
        if experiment_num == 5 or experiment_all:
            print("Defender learns vs fixed Random Attacker")

            # — override DO’s pool with *your* saved random attacker —
            double_oracle.saved_attacker_actors = [random_att_actor]
            double_oracle.attacker_equilibrium  = [1.0]
            double_oracle.baseline               = "RandomInit"
            env.base_line                        = "RandomInit"
            rand_def_ddpg = copy.deepcopy(template_def_ddpg)
            eval_x_ra, eval_y_ra = test_fixed_ddpg_training(
                env, initial_snapshot,
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

            # restore baseline if you’ll use the DO mixture later
            double_oracle.baseline = None
            env.base_line         = baseline

        # ── Experiment 6: Defender learns vs fixed Do-Nothing Attacker ────
        if experiment_num == 6 or experiment_all:
            print("Defender learns vs fixed Do nothing Attacker")
            env.base_line = "No Attack"
            rand_def_ddpg = copy.deepcopy(template_def_ddpg)
            eval_x_dn, eval_y_dn = test_fixed_ddpg_training(
                env, initial_snapshot,
                do=double_oracle,
                train_ddpg_agent=rand_def_ddpg,   # ← random start
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Cyber Defense simulation.')
    parser.add_argument('--DO_iterations', type=int, default=10, help='Max training episodes (with early stopping)')
    parser.add_argument('--test_episodes', type=int, default=1, help='Number of fixed-role test episodes at each test interval')
    parser.add_argument('--steps_per_episode', type=int, default=30, help='Steps per episode')
    parser.add_argument('--fixed_test_eps_length', type=int, default=5500, help='Steps per episode')
    parser.add_argument('--num_of_device', type=int, default=10, help='Number of devices') #was 20
    parser.add_argument('--output_dir', type=str, default='plots', help='Where to save per-iteration and final plots')
    parser.add_argument('--experiment_num', type=int, default=0, help='experiment type')
    parser.add_argument('--experiment_all',action='store_true',help='Do all experiments at once')
    parser.add_argument('--do_DO',action='store_true',help='Do DO (vs loading from policy)')
    parser.add_argument('--min_DO_iters', type=int, default=1, help='minimum number of DO rounds')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--its', type=int, default=1, help='instance of network')
    parser.add_argument('--dynamic_search',action='store_true',help='Search large action spaces?')
    parser.add_argument('--tabular_results',action='store_true',help='Pairwise play')
    parser.add_argument('--BR_type', type=str, default='Cord_asc', help='Type of Best Response- ddpg or Coordinate Ascent (Cord_asc)')
    parser.add_argument('--tabular_sims', type=int, default=10,help='Number of roll-outs per pairing per seed when tabulating')
    parser.add_argument('--zero_day',action='store_true',help='Randomize over exploits (eg theres some zero)')
    parser.add_argument('--k_known', type=int, default=1, help='how many known')
    parser.add_argument('--j_private', type=int, default=2, help='how many unkonwn')
    parser.add_argument('--workscale', type=float, default=1.0, help='workscale')
    parser.add_argument('--defensive_scale', type=float, default=1.0, help='defensive_scale')
    parser.add_argument('--preknown',type=int,default=0, help="Number of zero-day exploits that the defender already ‘discovers’ at t=0")
    parser.add_argument('--Dz_size',type=int,default=4, help="Number of zero days in the universe plus other exploits")
    parser.add_argument('--prune',action='store_true',help='Prune dominated Strategies?')
    parser.add_argument('--zero_day_submartingale',action='store_true',help='Should increasing Dz increase attack coverage in expection?')
    parser.add_argument('--max_Dz',type=int,default=6, help="Max number of Dz")
   
    


    args = parser.parse_args()

    DO_iterations = args.DO_iterations
    test_eps = args.test_episodes
    steps = args.steps_per_episode
    num_dev = args.num_of_device
    output = args.output_dir
    do_DO = args.do_DO
    min_DO_iters = args.min_DO_iters
    fixed_test_eps_length = args.fixed_test_eps_length
    experiment_num = args.experiment_num
    experiment_all = args.experiment_all
    seed = args.seed
    its = args.its
    dyn = args.dynamic_search
    BR_type = args.BR_type
    tab_results = args.tabular_results
    tabular_sims     = args.tabular_sims
    zero_day     = args.zero_day
    workscale = args.workscale
    defensive_scale = args.defensive_scale
    k_known = args.k_known
    j_private = args.j_private
    preknown = args.preknown
    Dz_size = args.Dz_size 
    prune = args.prune 
    zero_day_submartingale = args.zero_day_submartingale
    max_Dz = args.max_Dz


    
    

    print(f"DO iters:      {DO_iterations}\nTest episodes:    {test_eps}\n" +
          f"Steps per ep:     {steps}\nNum devices:      {num_dev}\nOutput dir:       {output}\n Seed:{seed}")
    print("Its: "+str(its))


    baselines = ["Nash"]



    set_seed(seed)
       
    

    for baseline in baselines:
        print(f"Running with seed={seed}, baseline={baseline}")

        # instead of building a fresh env, load your pre‑built snapshot
        snap_path = f"initial_net_DO_its{its}.pkl"

        
        if not os.path.isfile(snap_path):
            raise FileNotFoundError(f"Expected snapshot at {snap_path}")

        # load the pickled environment
        env: Volt_Typhoon_CyberDefenseEnv = pickle.load(open(snap_path, 'rb'))

        # MODIFY WORK SCALE HERE
        env.snapshot_path=snap_path
        new_work_scale = workscale # ← Set this to the desired ablation value
        env.work_scale = new_work_scale
        env.def_scale = defensive_scale
        env.comp_scale = 30
        env.its = its
        env.preknown = args.preknown


        if zero_day and j_private > 0:
            if not zero_day_submartingale:
                # ── full random regeneration ────────────────────────────────
                env.simulator.exploits.clear()
                env.simulator.redeploy_apps_with_unique_vulns(
                    num_apps=10, vul_per_app=1,
                    min_apps_per_device=1, max_apps_per_device=3
                )
                env.simulator.generateExploits(
                    numOfExploits = Dz_size,
                    addVul        = True,
                    minVulperExp  = 1,
                    maxVulperExp  = 1,
                    mode          = "random"
                )
                all_ids = [e.id for e in env.simulator.exploits]
                needed = k_known + j_private
                if len(all_ids) < needed:
                    raise RuntimeError(
                        f"Need at least {needed} exploits but only have {len(all_ids)}"
                    )
                random.seed(seed)
                known_ids     = random.sample(all_ids, k_known)
                remaining     = [eid for eid in all_ids if eid not in known_ids]
                private_ids   = random.sample(remaining, j_private)
                unknown_pool  = [eid for eid in remaining if eid not in private_ids]

            else:
                # --- zero-day submartingale wrt |Dz| ---
                # 1) redeploy apps so every device has at least one vuln
                env.simulator.redeploy_apps_with_unique_vulns(
                    num_apps=10, vul_per_app=1,
                    min_apps_per_device=1, max_apps_per_device=3
                )

                # 2) find the DomainController (always compromisable)
                all_devs = list(env.simulator.subnet.net.keys())
                DC = next((d for d, dev in env.simulator.subnet.net.items()
                        if getattr(dev, "device_type", None) == "DomainController"),
                        None)
                if DC is None:
                    raise RuntimeError("No DomainController in the network!")
                others = [d for d in all_devs if d != DC]

                # 3) build exactly Dz_size exploits, each covering
                #    floor(Dz_size*(num_devices-1)/max_Dz) of the *other* devices + DC
                env.simulator.exploits.clear()
                new_exploits = []
                exploit_device_map = {}
                n_targets = max(
                    1,
                    int(round(Dz_size * len(others) / float(max_Dz)))
                )

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

                # 4) install & sync the env’s exploit‐count & action‐spaces ONCE:
                env.simulator.exploits = new_exploits
                

                # 5) split into known / private / unknown exactly as before
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

                # 6) finalize zero-day flags
                env.zero_day            = True
                env.k_known             = k_known
                env.j_private           = j_private
                env.common_exploit_ids  = known_ids
                env.private_exploit_ids = private_ids
                env.unknown_pool_ids    = unknown_pool

                print(f">>> ZERO-DAY (linear coverage): "
                    f"known={known_ids}, private={private_ids}, unknown={unknown_pool}")

        elif zero_day and j_private == 0:
            # turn off zero-day entirely
            env.zero_day = False
            # (re)generate a single exploit if you still want one in the snapshot
            env.simulator.redeploy_apps_with_unique_vulns(
                num_apps=10, vul_per_app=1,
                min_apps_per_device=1, max_apps_per_device=3
            )
            env.simulator.generateExploits(
                numOfExploits = 1,
                addVul        = True,
                minVulperExp  = 1,
                maxVulperExp  = 1,
                mode          = "random"
            )
            # clear all pools
            env.k_known              = 0
            env.j_private            = 0
            env.common_exploit_ids   = []
            env.private_exploit_ids  = []
            env.unknown_pool_ids     = []
            env.initialize_environment()

        else:
            # fallback non–zero-day behavior
            env.zero_day             = False
            env.k_known              = 1
            env.j_private            = 0
            env.common_exploit_ids   = []
            env.private_exploit_ids  = []
            env.unknown_pool_ids     = []
            env.initialize_environment()


        print(f"[ABLATION] Overwrote work_scale → {env.work_scale}")
        print(f"[ABLATION] Overwrote def_scale → {env.def_scale}")
        print(f"[ABLATION] Overwrote num_dev → {env.numOfDevice}")

        # Overwrite the pickle file with the modified environment
        with open(snap_path, "wb") as f:
            pickle.dump(env, f)
        print(f"[ABLATION] Overwrote snapshot file with new work_scale at {snap_path}")


        # tell the env which file to reload on reset(from_init=True)
        env.snapshot_path = snap_path  


        # set the rest of the run parameters
        env.base_line        = baseline
        env.tech             = "DO"
        env.mode             = "defender"


        # will reload exactly that pickled snapshot.
        initial_snapshot = copy.deepcopy(env)



        info_path = os.path.join(output, "env_info.txt")
        #only write header once
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        if not os.path.exists(info_path):
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
        #Some hardcodes

        if info_path is not None:
            with open(info_path, 'a') as f:
                f.write("### DDPG hyperparameters ###\n")
                f.write(f"reward_scale  = {1!r}\n")
                f.write(f"max_grad_norm = {0.5!r}\n")
                f.write(f"soft‐update τ = {1e-2!r}\n")
                f.write("\n")

        run_game(
            env, initial_snapshot, DO_iterations, test_eps, steps,
            seed, baseline, output, do_DO,experiment_all, min_DO_iters, fixed_test_eps_length, experiment_num,dyn,BR_type,tab_results,tabular_sims,info_path,zero_day,prune,zero_day_submartingale
        )


    


    print("Simulation complete")
