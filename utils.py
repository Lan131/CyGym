import random
import numpy as np
from do_agent import DoubleOracle, train_ddpg, Actor , Strategy
from tqdm import trange
import torch
import copy
import torch.nn.functional as F
from collections import defaultdict
import pickle
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv
import pandas as pd
import os
import logging
from do_agent import DoubleOracle, Strategy  # ensure we can instantiate or load

def generate_extended_metrics_table(attacker_bases, defender_bases, all_metrics, num_dev):
    """
    Returns a single string containing all LaTeX tables for:
    - total checkpoint+revert count (reset_count)
    - total checkpoint count (checkpoint_count)
    - avg number of compromised devices per timestep (compromised)
    - avg number of workloads executed (jobs)
    - total defensive cost (defensive_cost)
    - total scans performed (scan_count)
    - total edges blocked (edges_blocked)
    - total edges added (edges_added)

    All metrics are scaled by num_dev (where appropriate).
    """
    metric_names = [
        "reset_count", "checkpoint_count",
        "compromised", "jobs", "defensive_cost", "scan_count",
        "edges_blocked", "edges_added"
    ]
    metric_titles = {
        "reset_count":      "Total checkpoint + revert count",
        "checkpoint_count": "Total checkpoint count",
        "compromised":      "Average compromised devices per timestep",
        "jobs":             "Average workloads executed",
        "defensive_cost":   "Total defensive cost",
        "scan_count":       "Total scans performed",
        "edges_blocked":    "Total edges blocked",
        "edges_added":      "Total edges added"
    }

    # all_metrics entries: (d_pay=0, a_pay=1, compromised=2, jobs=3, scan_cnt=4,
    # defensive_cost=5, checkpoint_cnt=6, revert_cnt=7, edges_blocked=8, edges_added=9)
    metric_indices = {
        "reset_count":      None,  # special case: sum of indices 6 + 7
        "checkpoint_count": 6,
        "compromised":      2,
        "jobs":             3,
        "scan_count":       4,
        "defensive_cost":   5,
        "edges_blocked":    8,
        "edges_added":      9
    }

    all_tables = []

    for metric in metric_names:
        caption = metric_titles[metric]
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\begin{tabular}{l" + "c" * len(defender_bases) + r"}",
            r"\toprule",
            # Removed "(lr)" from \cmidrule
            r"& \multicolumn{" + str(len(defender_bases)) + r"}{c}{\textbf{Defender strategies}} \\",
            r"\cmidrule{2-" + str(len(defender_bases) + 1) + r"}",
            r"\textbf{Attacker $\downarrow$ \quad Defender $\rightarrow$} & " +
                " & ".join(f"\\textbf{{{d}}}" for d in defender_bases) + r" \\",
            r"\midrule",
        ]

        for a in attacker_bases:
            row = [f"\\textbf{{{a}}}"]
            for d in defender_bases:
                if (a, d) in all_metrics and len(all_metrics[(a, d)]) > 0:
                    if metric == "reset_count":
                        # reset_count = checkpoint_count (idx 6) + revert_count (idx 7)
                        vals = [entry[6] + entry[7] for entry in all_metrics[(a, d)]]
                    else:
                        idx = metric_indices[metric]
                        vals = [entry[idx] for entry in all_metrics[(a, d)]]

                    mean = sum(vals) / len(vals)
                    std_err = (sum((x - mean)**2 for x in vals) / len(vals))**0.5 / (len(vals)**0.5)
                    row.append(f"${mean:.3f}\\pm{std_err:.3f}$")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ]

        all_tables.append("\n".join(lines))

    return "\n\n".join(all_tables)



def generate_latex_table(attacker_bases, defender_bases, all_results, num_dev, payoff_type="attacker"):
    """
    attacker_bases: list of attacker‐baseline names (strings)
    defender_bases: list of defender‐baseline names (strings)
    all_results: dict mapping (atk_base, def_base) → list of (def_payoff, att_payoff)
    payoff_type: either "attacker" or "defender"
    """
    caption = f"{payoff_type.capitalize()} average payoffs (mean\\,$\\pm$\\,std)"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
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
        for defb in defender_bases:
            runs = np.array(all_results[(atk, defb)])  # shape (N,2)
            d_mean, d_std = runs[:,0].mean(), (runs[:,0]/num_dev).std()
            a_mean, a_std = runs[:,1].mean(), (runs[:,1]/num_dev).std()
            if payoff_type == "attacker":
                m, s = a_mean/num_dev, a_std / np.sqrt(len(runs))
            else:
                m, s = d_mean/num_dev, d_std / np.sqrt(len(runs))
            row.append(f"${m:.3f}\\pm{s:.3f}$")
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ]
    return "\n".join(lines)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you want full determinism:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def smooth_rewards(rewards, window=10):
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i+1]))
    return smoothed

def initialize_reward_dict(steps_per_episode, episodes):
    max_steps = steps_per_episode * episodes + 1
    return {i: [] for i in range(0, max_steps, 100)}

def compute_tabular_results(
    snapshot_path: str,
    tabular_seeds: list[int],
    tabular_sims: int,
    steps_per_episode: int,
    save_dir: str,
    BR_type: str
):

    attacker_bases = ["DO", "RandomInit", "No Attack"]
    defender_bases = ["DO", "RandomInit", "No Defense", "Preset"]
    all_results = defaultdict(list)
    all_metrics = defaultdict(list)

    # how many sims per support pair to get mixture-expectation statistics
    S_per = 5

    for seed in tabular_seeds:
        print(f"[==== seed={seed} ====]")
        set_seed(seed)

        # load oracle
        oracle_path = os.path.join(save_dir, f"oracle_seed{seed}.pkl")
        with open(oracle_path, "rb") as f:
            oracle = pickle.load(f)

        # rebuild payoff matrices and equilibrium mixtures
        D_mat, A_mat = oracle.build_payoff_matrices()
        p, q = oracle.defender_equilibrium, oracle.attacker_equilibrium

        # index lookup for named baselines
        def_index = {s.baseline_name: i for i, s in enumerate(oracle.defender_strategies) if s.baseline_name}
        att_index = {s.baseline_name: j for j, s in enumerate(oracle.attacker_strategies) if s.baseline_name}

        # for each pairing
        for atk_base in attacker_bases:
            for def_base in defender_bases:
                for run_idx in range(tabular_sims):
                    set_seed(seed * 100 + run_idx)
                    logging.info(f"=== Starting {def_base} vs {atk_base}, run {run_idx} ===")

                    # prepare the two Strategy objects
                    if def_base == "DO":
                        idx_def = np.random.choice(len(oracle.defender_strategies), p=p)
                        def_strat = oracle.defender_strategies[idx_def]
                    elif def_base == "RandomInit":
                        idx_def = def_index["RandomInit"]
                        def_strat = oracle.defender_strategies[idx_def]
                    else:
                        idx_def = def_index[def_base]
                        def_strat = Strategy(None, baseline_name=def_base)

                    if atk_base == "DO":
                        idx_att = np.random.choice(len(oracle.attacker_strategies), p=q)
                        att_strat = oracle.attacker_strategies[idx_att]
                    elif atk_base == "RandomInit":
                        idx_att = att_index["RandomInit"]
                        att_strat = oracle.attacker_strategies[idx_att]
                    else:
                        idx_att = att_index["No Attack"]
                        att_strat = Strategy(None, baseline_name="No Attack")

                    # decide whether to use matrix payoff or simulation
                    use_matrix = not (atk_base != "DO" and def_base != "DO")

                    # --- handle DO vs DO extended-metrics exactly via mixture expectation ---
                    if def_base == "DO" and atk_base == "DO":
                        # expected payoff (already handled in all_results)
                        d_pay = p.dot(D_mat).dot(q)
                        a_pay = q.dot(A_mat).dot(p)

                        # now expected extended metrics
                        # we'll accumulate a 10-element vector:
                        exp_metrics = np.zeros(10)
                        D_strats = oracle.defender_strategies
                        A_strats = oracle.attacker_strategies

                        for i, pi in enumerate(p):
                            for j, qj in enumerate(q):
                                # simulate S_per episodes for each support pair
                                batch = np.array([
                                    oracle.simulate_game(D_strats[i], A_strats[j], num_simulations=1)
                                    for _ in range(S_per)
                                ])  # shape (S_per, 10)
                                avg_ij = batch.mean(axis=0)
                                exp_metrics += pi * qj * avg_ij

                        # record one tuple of expected metrics
                        all_metrics[(atk_base, def_base)].append(tuple(exp_metrics.tolist()))
                        all_results[(atk_base, def_base)].append((d_pay, a_pay))
                        continue

                    # otherwise use your standard simulate_vs or matrix lookup
                    d_sim, a_sim, *rest = oracle.simulate_game(def_strat, att_strat, num_simulations=1)

                    if use_matrix and not (def_base=="DO" and atk_base=="DO"):
                        d_pay = D_mat[idx_def, idx_att]
                        a_pay = A_mat[idx_att, idx_def]
                    else:
                        d_pay = d_sim
                        a_pay = a_sim

                    # record
                    all_results[(atk_base, def_base)].append((d_pay, a_pay))
                    all_metrics[(atk_base, def_base)].append((
                        d_pay, a_pay, *rest
                    ))

    # return LaTeX tables
    return (
        generate_latex_table(attacker_bases, defender_bases, all_results,
                             oracle.env.numOfDevice, payoff_type="attacker"),
        generate_latex_table(attacker_bases, defender_bases, all_results,
                             oracle.env.numOfDevice, payoff_type="defender"),
        generate_extended_metrics_table(attacker_bases, defender_bases,
                                        all_metrics, oracle.env.numOfDevice)
)


def sample_fixed_states(env, num_samples: int = 100, seed: int = 0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Run the env forward a little under random actions to collect a set of
    states for contrastive‐learning. Always pads/truncates the action returned
    by env.sample_action() into a 4‐tuple.
    Returns a tensor of shape (num_samples, state_dim) on `device`.
    """
    random.seed(seed)
    np.random.seed(seed)

    states = []
    for _ in range(num_samples):
        # reset to a fresh snapshot
        env.reset(from_init=True)
        # if your env supports randomizing who is compromised, do that:
        if hasattr(env, "randomize_compromise_and_ownership"):
            env.randomize_compromise_and_ownership()

        # sample an action of arbitrary length
        a = env.sample_action()

        # pad or truncate into (atype, exps, devs, app_idx)
        if len(a) == 4:
            action = a
        elif len(a) == 3:
            # assume missing app_index → zero
            action = (a[0], a[1], a[2], 0)
        else:
            raise ValueError(f"sample_action returned unexpected tuple of length {len(a)}: {a}")

        # step once (mode should already be correct)
        _state, _, _, *_ = env.step(action)

        # grab the attacker‐view state
        st = env._get_attacker_state()
        st_t = torch.tensor(st, dtype=torch.float32, device=device)
        states.append(st_t)

    return torch.stack(states, dim=0)





def far_apart_ddpg_restart(
    init_ddpg_fn,             # () -> dict with 'actor' and 'critic'
    saved_actor_dicts,        # list of actor.state_dict() from previous policies
    device,                   # torch.device
    fixed_states,             # Tensor [B, state_dim]
    sim_thresh: float = 0.1,  # max allowed cosine similarity
    max_restarts: int = 100,
    seed: int = None
):
    """
    Try up to `max_restarts` fresh inits, returning the first one whose
    param-space *and* Q-value fingerprints both stay below sim_thresh.
    If none qualify, returns the candidate whose *maximum* similarity
    to the old set is *smallest* (i.e. most different).
    """
    S = fixed_states.to(device)

    # build a fixed critic for Q-value comparisons
    template = init_ddpg_fn()
    critic   = template['critic'].to(device).eval()

    # precompute old param-space fingerprints
    old_param_fps = []
    for sd in saved_actor_dicts:
        vec = torch.cat([p.flatten().to(device)
                         for p in sd.values() if isinstance(p, torch.Tensor)], dim=0)
        old_param_fps.append(F.normalize(vec, dim=0).unsqueeze(0))  # (1, P)
    old_param_fps = torch.cat(old_param_fps, dim=0) if old_param_fps else torch.empty((0,0), device=device)

    # precompute old Q-value fingerprints
    old_q_fps = []
    for sd in saved_actor_dicts:
        cand = init_ddpg_fn()
        actor_i = cand['actor'].to(device).eval()
        actor_i.load_state_dict(sd)
        with torch.no_grad():
            A_i = actor_i(S)
            Q_i = critic(S, A_i)
        old_q_fps.append(Q_i.mean(dim=0, keepdim=True))  # (1,1)
    old_q_fps = torch.cat(old_q_fps, dim=0) if old_q_fps else torch.empty((0,1), device=device)

    # keep track of best candidate so far
    best_cand = None
    best_score = float('inf')  # lower is better (max similarity)

    for i in range(max_restarts):
        # reseed everything
        if seed is not None:
            torch.manual_seed(seed + i)
            torch.cuda.manual_seed_all(seed + i)
            np.random.seed(seed + i)
            random.seed(seed + i)

        cand = init_ddpg_fn()
        actor = cand['actor'].to(device).eval()

        # param fingerprint
        vec_new = torch.cat([p.flatten()
                             for p in actor.state_dict().values() if isinstance(p, torch.Tensor)],
                            dim=0).to(device)
        fp_param_new = F.normalize(vec_new, dim=0).unsqueeze(0)  # (1,P)

        # Q-value fingerprint
        with torch.no_grad():
            A_new = actor(S)
            Q_new = critic(S, A_new)
        fp_q_new = Q_new.mean(dim=0, keepdim=True)  # (1,1)

        # compute worst‐case similarity for this candidate
        sim_param = old_param_fps.numel() and F.cosine_similarity(fp_param_new, old_param_fps, dim=1).max().item() or 0.0
        sim_q     = old_q_fps.numel()     and F.cosine_similarity(fp_q_new,     old_q_fps,     dim=1).max().item() or 0.0
        worst_sim = max(sim_param, sim_q)

        # if fully under threshold, return immediately
        if worst_sim < sim_thresh:
            return cand

        # otherwise track if this is the most “different” so far
        if worst_sim < best_score:
            best_score = worst_sim
            best_cand  = cand

    # none passed the threshold: return the single best
    return best_cand



    
def test_fixed_ddpg_training(env,
                             initial_snapshot,
                             do: DoubleOracle,
                             train_ddpg_agent,
                             fixed_role: str,
                             steps_per_episode: int,
                             eval_episode_len: int,
                             eval_episodes: int,
                             eval_feq: int,
                             gamma: float = 0.95,
                             batch_size: int = 512,
                             #burn_in: int = 1000,
                             burn_in: int = 500,
                             σ: float = .3,
                             σ_min: float = 1e-5,
                             load: bool = False,
                             cord_ascen: bool = True):
    """
    Train a DDPG agent against a fixed opponent strategy (sampled from do.saved_*_actors
    according to do.*_equilibrium).  Optionally uses Dynamic Neighborhood Construction
    whenever do.dynamic_neighbor_search is True.
    Returns lists of (step, avg_reward) at every eval_feq steps.
    """
    D = env.Max_network_size
    E = env.MaxExploits
    A = env.get_num_app_indices()

    # choose which side is fixed vs learning
    if fixed_role == "defender":
        saved_dicts, eq_probs = do.saved_defender_actors, do.defender_equilibrium
        get_fixed, get_learn = env._get_defender_state, env._get_attacker_state
        n_fixed, n_learn = env.get_num_action_types("defender"), env.get_num_action_types("attacker")
    else:
        saved_dicts, eq_probs = do.saved_attacker_actors, do.attacker_equilibrium
        get_fixed, get_learn = env._get_attacker_state, env._get_defender_state
        n_fixed, n_learn = env.get_num_action_types("attacker"), env.get_num_action_types("defender")

    # build fixed-policy actor nets
    fixed_actors = []
    state_dim = get_fixed().shape[0]
    for sd in saved_dicts:
        action_dim = sd["fc3.weight"].shape[0]
        actor = Actor(state_dim, action_dim, do.seed, do.device)
        actor.load_state_dict(sd)
        actor.to(do.device).eval()
        fixed_actors.append(actor)

    # unpack learner’s DDPG components
    train_actor   = train_ddpg_agent['actor']
    train_critic  = train_ddpg_agent['critic']
    target_actor  = train_ddpg_agent['target_actor']
    target_critic = train_ddpg_agent['target_critic']
    actor_opt     = train_ddpg_agent['actor_optimizer']
    critic_opt    = train_ddpg_agent['critic_optimizer']
    buffer        = train_ddpg_agent['replay_buffer']
    buffer.buffer.clear()

    # noise schedule
    decay_rate = (σ_min / σ) ** (1.0 / steps_per_episode)
    noise_std  = σ

    eval_steps, eval_rewards = [], []
    test_interval = eval_feq
    

    for step in range(steps_per_episode):
        turn = 'defender' if (step % 2 == 0) else 'attacker'
        env.mode = turn

        # get current state and action-space size
        if turn == fixed_role:
            state, n_types = get_fixed(), n_fixed
        else:
            state, n_types = get_learn(), n_learn

        # ─── action selection ───────────────────────────────────────
        if turn == fixed_role:
            # sample from fixed strategy mixture
            idx = np.random.choice(len(fixed_actors), p=eq_probs)
            with torch.no_grad():
                vec = fixed_actors[idx](
                    torch.tensor(state, dtype=torch.float32)
                          .unsqueeze(0).to(do.device)
                ).cpu().numpy()[0]
            action_disc = do.decode_action(vec, n_fixed, D, E, A)
            action_vec = None

        else:
            # learner’s raw actor output
            st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(do.device)
            with torch.no_grad():
                raw = train_actor(st_t).detach().cpu().numpy()[0]

            # add exploration noise for replay buffer
            noise      = np.random.normal(0.0, noise_std, size=raw.shape)
            action_vec = np.clip(raw + noise, -1.0, +1.0)
            noise_std  = max(σ_min, noise_std * decay_rate)

            #If dynamic neighbor is used this needs fixed
            action_disc = do.decode_action(action_vec, n_learn, D, E, A)

        # ─── **now** step the env, so we have real r and done ─────────
        _, r, done, *_ = env.step(action_disc)

        # if this was the learner’s turn, only then push & train
        if turn != fixed_role:
            if not cord_ascen:
                next_state = get_learn()
                buffer.push(state, action_vec, r, next_state, done)
                if step > burn_in:
                    train_ddpg(train_actor, train_critic,
                            target_actor, target_critic,
                            buffer, actor_opt, critic_opt,
                            batch_size, gamma, do.device)
            if cord_ascen:
                next_state = get_learn()

                # --- encode the discrete action you actually took ---
                # action_disc is the Cord_asc / DNC pick
                disc_vec = do.encode_action(
                    action_disc,
                    n_learn,            # number of action types for the learner
                    env.Max_network_size,
                    env.MaxExploits,
                    env.get_num_app_indices()
                )

                buffer.push(state, disc_vec, r, next_state, done)
                if step > burn_in:
                    train_ddpg(train_actor, train_critic,
                            target_actor, target_critic,
                            buffer, actor_opt, critic_opt,
                            batch_size, gamma, do.device)

        # ─── periodic evaluation ──────────────────────────────────────
        if step % test_interval == 0 and step > burn_in:
            tot_r = 0.0
            for _ in range(eval_episodes):
                ev = copy.deepcopy(initial_snapshot)
                ev.reset(from_init=True)
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
                                state_tensor=st_te,
                                raw_action=raw2,
                                actor=train_actor,
                                critic=train_critic,
                                k_init=3,
                                beta_init=.05,
                                c_k=1.0,
                                c_beta=0.2
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
