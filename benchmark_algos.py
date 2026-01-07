# benchmark_algos.py
from __future__ import annotations

import os, gc, time, random, statistics, csv, warnings, pickle, shutil, sys, threading, copy, math, argparse
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# ======================================================
# User toggle: run ONLY the HMARL baselines (Expert/Meta)
# ======================================================
ONLY_HMARL = False  # set True to benchmark ONLY HMARLExpert / HMARLMeta

# ============= Silence AMP deprecation & shim old autocast ============
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(.*\)` is deprecated"
)

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

if HAS_TORCH:
    try:
        def _compat_autocast(*args, **kwargs):
            return torch.amp.autocast(device_type="cuda", **kwargs)
        if hasattr(torch, "amp") and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            # monkey-patch for older code paths that import torch.cuda.amp.autocast
            torch.cuda.amp.autocast = _compat_autocast  # type: ignore[attr-defined]
    except Exception:
        pass

# try importing psutil for reliable process memory; fallback allowed
try:
    import psutil
    HAS_PSUTIL = True
    _PS_PROC = psutil.Process(os.getpid())
except Exception:
    HAS_PSUTIL = False
    _PS_PROC = None

# ======================= Core env & agents ============================
from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv
from do_agent import DoubleOracle, Strategy

# Optional controllers (skip gracefully if missing)
try:
    from meta_hierarchical_br import MetaHierarchicalBestResponse
    HAS_META = True
except Exception:
    HAS_META = False

try:
    from hierarchical_br import HierarchicalBestResponse  # HAGS
    HAS_HAGS = True
except Exception:
    HAS_HAGS = False

# ---- IPPO / MAPPO / HMARL discovery ----
HAS_IPPO = False
HAS_MAPPO = False
HAS_HMARL_EXPERT = False
HAS_HMARL_META = False

IPPOCommBestResponse = IPPOCommPolicy = None
MAPPOTrainer = None

HMARLExpertBestResponse_cls = None
HMARLMetaBestResponse_cls = None
ExpertRuleMaster = None
FrozenSubPolicy = None

# Try IPPO
try:
    from IPPO import IPPOCommBestResponse, IPPOCommPolicy  # type: ignore
    HAS_IPPO = True
except Exception:
    pass

# Try MAPPO
for modpath, clsname in [
    ("MAPPO", "MAPPOCommBestResponse"),
    ("mappo", "MAPPOCommBestResponse"),
    ("marl",  "MAPPOCommBestResponse"),
    ("MAPPO", "MAPPOBestResponse"),
    ("mappo", "MAPPOBestResponse"),
    ("marl",  "MAPPOBestResponse"),
]:
    try:
        mod = __import__(modpath, fromlist=[clsname])
        MAPPOTrainer = getattr(mod, clsname)
        HAS_MAPPO = True
        break
    except Exception:
        continue

# Try HMARLExpert
try:
    from HMARL import (
        HMARLExpertBestResponse,
        ExpertRuleMaster as _ExpertRuleMaster,
        FrozenSubPolicy as _FrozenSubPolicy,
    )
    HMARLExpertBestResponse_cls = HMARLExpertBestResponse
    ExpertRuleMaster = _ExpertRuleMaster
    FrozenSubPolicy = _FrozenSubPolicy
    HAS_HMARL_EXPERT = True
except Exception:
    pass

# Try HMARLMeta
try:
    from HMARL import (
        HMARLMetaBestResponse,
        FrozenSubPolicy as _FrozenSubPolicyMeta,
    )
    HMARLMetaBestResponse_cls = HMARLMetaBestResponse
    if FrozenSubPolicy is None:
        FrozenSubPolicy = _FrozenSubPolicyMeta
    HAS_HMARL_META = True
except Exception:
    pass

# ============================ CONFIG ==================================
# SIZES will be overridden by --size at runtime (we keep it as a list)
SIZES         = [20000]
ROLE          = "defender"   # training role (timing averages over both roles)
BASELINE      = "Nash"
ZERO_DAY      = False
SEED          = 0

TRAIN_STEPS   = 100
N_TRAIN_RUNS  = 2

# ⬇️ default timing steps now 4, averaged over defender+attacker
N_STEPS_EXEC  = 4
WARMUP_STEPS  = 0

OUT_DIR       = "bench_results"
TRIAL_TAG     = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
TRIAL_DIR     = os.path.join(OUT_DIR, "trial_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
SNAP_DIR      = os.path.join(TRIAL_DIR, "snapshots")
HAGS_DIR      = os.path.join(TRIAL_DIR, "hags")
CLEANUP_SNAPSHOTS = True

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(HAGS_DIR, exist_ok=True)

# Global selected CUDA device (int) or None
SELECTED_CUDA_DEVICE: Optional[int] = None

# ======================= Utilities / helpers ==========================
def clear_caches_and_memory():
    try:
        plt.close('all')
    except Exception:
        pass
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                try:
                    if SELECTED_CUDA_DEVICE is not None:
                        torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                    else:
                        torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    gc.collect()

def select_cuda_device_from_env():
    """
    Chooses a CUDA device index to use for the run, based on environment variables.
    """
    global SELECTED_CUDA_DEVICE
    if not HAS_TORCH or not torch.cuda.is_available():
        SELECTED_CUDA_DEVICE = None
        return

    cand_names = ["BENCH_GPU", "CUDA_DEVICE", "CUDA_DEVICE_INDEX"]
    pick = None
    for n in cand_names:
        v = os.environ.get(n)
        if not v:
            continue
        v = v.strip()
        if v == "":
            continue
        try:
            pick = int(v)
            break
        except Exception:
            continue

    if pick is None:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            pick = 0
        else:
            pick = 0

    try:
        torch.cuda.set_device(pick)
        SELECTED_CUDA_DEVICE = pick
        print(f"[GPU] selected CUDA device index = {pick} (torch device)", flush=True)
    except Exception as e:
        print(f"[GPU] failed to set CUDA device to {pick}: {e}", flush=True)
        SELECTED_CUDA_DEVICE = None

def print_memory_snapshot(label: str = "") -> None:
    """Print a compact memory snapshot (host RAM + swap, and CUDA if present)."""
    tag = f"[MEM{(':' + label) if label else ''}]"
    # Host memory
    try:
        if HAS_PSUTIL:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            print(f"{tag} total={vm.total:,} used={vm.used:,} avail={vm.available:,} percent={vm.percent}%", flush=True)
            print(f"{tag} swap_total={swap.total:,} swap_used={swap.used:,} swap_percent={swap.percent}%", flush=True)
        else:
            with open("/proc/meminfo", "r") as f:
                lines = f.read().splitlines()
            print(f"{tag} /proc/meminfo (top):", flush=True)
            for L in lines[:8]:
                print(f"    {L}", flush=True)
    except Exception as e:
        print(f"{tag} failed to get host memory info: {e}", flush=True)

    # CUDA memory
    if HAS_TORCH:
        try:
            if torch.cuda.is_available():
                try:
                    if SELECTED_CUDA_DEVICE is not None:
                        torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                    else:
                        torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    dev = SELECTED_CUDA_DEVICE
                    alloc = torch.cuda.memory_allocated(device=dev)
                    reserved = torch.cuda.memory_reserved(device=dev)
                    max_alloc = torch.cuda.max_memory_allocated(device=dev)
                    print(f"{tag} CUDA(device={dev}) allocated={alloc:,} reserved={reserved:,} max_allocated={max_alloc:,}", flush=True)
                    try:
                        summary = torch.cuda.memory_summary(device=dev, abbreviated=True) if hasattr(torch.cuda, "memory_summary") else torch.cuda.memory_summary(abbreviated=True)  # type: ignore[arg-type]
                        for i, line in enumerate(summary.splitlines()):
                            if i > 40:
                                print(f"    ... (truncated)", flush=True)
                                break
                            print(f"    {line}", flush=True)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"{tag} CUDA query failed: {e}", flush=True)
        except Exception as e:
            print(f"{tag} torch/CUDA check failed: {e}", flush=True)

def bytes_to_mb(b: Optional[int]) -> float:
    if b is None:
        return float('nan')
    return float(b) / (1024.0 * 1024.0)

def run_with_memory_monitor(fn, *args, poll_interval: float = 0.1, **kwargs):
    """
    Run fn(*args, **kwargs) while polling process RSS and CUDA memory for the SELECTED_CUDA_DEVICE.
    Returns (ret, mem_stats)
    """
    stop_event = threading.Event()
    mem = {
        'host_rss_max': 0 if HAS_PSUTIL else None,
        'cuda_alloc_max': 0 if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None,
        'cuda_reserved_max': 0 if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None,
    }

    def monitor_loop():
        while not stop_event.is_set():
            try:
                if HAS_PSUTIL and _PS_PROC is not None:
                    rss = _PS_PROC.memory_info().rss
                    if mem['host_rss_max'] is None or rss > mem['host_rss_max']:
                        mem['host_rss_max'] = rss
            except Exception:
                pass
            if HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
                try:
                    try:
                        torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                    except Exception:
                        pass
                    a = torch.cuda.memory_allocated(device=SELECTED_CUDA_DEVICE)
                    r = torch.cuda.memory_reserved(device=SELECTED_CUDA_DEVICE)
                    if mem['cuda_alloc_max'] is None or a > mem['cuda_alloc_max']:
                        mem['cuda_alloc_max'] = a
                    if mem['cuda_reserved_max'] is None or r > mem['cuda_reserved_max']:
                        mem['cuda_reserved_max'] = r
                except Exception:
                    pass
            time.sleep(poll_interval)

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        try:
            if HAS_PSUTIL and _PS_PROC is not None:
                rss = _PS_PROC.memory_info().rss
                if mem['host_rss_max'] is None or rss > mem['host_rss_max']:
                    mem['host_rss_max'] = rss
        except Exception:
            pass
        if HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
            try:
                try:
                    torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                except Exception:
                    pass
                a = torch.cuda.memory_allocated(device=SELECTED_CUDA_DEVICE)
                r = torch.cuda.memory_reserved(device=SELECTED_CUDA_DEVICE)
                if mem['cuda_alloc_max'] is None or a > mem['cuda_alloc_max']:
                    mem['cuda_alloc_max'] = a
                if mem['cuda_reserved_max'] is None or r > mem['cuda_reserved_max']:
                    mem['cuda_reserved_max'] = r
            except Exception:
                pass
        stop_event.set()
        monitor_thread.join()
    return result, mem

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def build_env(M: int) -> Volt_Typhoon_CyberDefenseEnv:
    env = Volt_Typhoon_CyberDefenseEnv()
    env.numOfDevice      = M
    env.Max_network_size = max(M, M + 10)
    env.zero_day         = ZERO_DAY
    env.base_line        = BASELINE
    env.tech             = "DO"
    env.debug            = False
    env.mode             = "defender"
    env.initialize_environment()
    return env

def create_snapshot_for_size(M: int, tag: str) -> str:
    env = build_env(M)
    snap_path = os.path.join(SNAP_DIR, f"initial_net_{tag}_M{M}.pkl")
    with open(snap_path, "wb") as f:
        pickle.dump(env, f)
    return snap_path

def load_env_from_snapshot(snap_path: str) -> Volt_Typhoon_CyberDefenseEnv:
    with open(snap_path, "rb") as f:
        env = pickle.load(f)
    if not getattr(env, "mode", None):
        env.mode = "defender"
    if getattr(env, "base_line", None) is None:
        env.base_line = BASELINE
    try:
        env.initialize_environment()
    except Exception:
        pass
    try:
        # touch a state method to ensure tensors are wired
        _ = env._get_defender_state()
    except Exception:
        pass
    return env

def ensure_hags_paths(oracle: DoubleOracle, env: Volt_Typhoon_CyberDefenseEnv, size_label: str):
    setattr(env, "snapshot_path", getattr(env, "snapshot_path", None))
    setattr(oracle, "output_dir", HAGS_DIR)
    setattr(oracle, "info_path", os.path.join(HAGS_DIR, f"info_{size_label}.txt"))
    try:
        with open(oracle.info_path, "a"):
            pass
    except Exception:
        pass

def _ensure_env_initialized_on_do(do: DoubleOracle) -> None:
    env = getattr(do, "env", None) or getattr(do, "env_template", None)
    if env is None:
        return
    try:
        if not getattr(env, "mode", None):
            env.mode = "defender"
        if getattr(env, "base_line", None) is None:
            env.base_line = BASELINE
    except Exception:
        pass
    try:
        if hasattr(env, "initialize_environment"):
            env.initialize_environment()
    except Exception:
        pass
    try:
        if hasattr(env, "_get_defender_state"):
            _ = env._get_defender_state()
        elif hasattr(env, "_get_attacker_state"):
            _ = env._get_attacker_state()
    except Exception:
        pass

def safe_do_checkpoint(do: DoubleOracle) -> None:
    _ensure_env_initialized_on_do(do)
    try:
        do.checkpoint_now()
    except Exception as e:
        msg = str(e)
        if "env not initialized (simulator/state missing)" in msg:
            print(f"   skipped ({msg})")
            return
        raise

# ===================== Strategy execution ============================

def _get_cached(obj, name, factory):
    v = getattr(obj, name, None)
    if v is None:
        v = factory()
        setattr(obj, name, v)
    return v

def execute_one_step(oracle: DoubleOracle,
                     env: Volt_Typhoon_CyberDefenseEnv,
                     strat: Strategy,
                     role: str):
    """
    Execute exactly one decision for the given strategy in this env.
    Handles HMARL Expert/Meta (both roles), MetaHierarchical, HAGS, IPPO, MAPPO, and DOAR fallback.
    """
    # Allow baseline override to switch env baseline behavior
    if getattr(strat, "baseline_name", None):
        env.base_line = strat.baseline_name
        return None

    tm = getattr(strat, "type_mapping", {}) or {}

    # ---- HMARL Expert / Meta routing (works for defender and attacker) ----
    if ("hmarl_expert" in tm or "hmarl_meta" in tm) and (HAS_HMARL_EXPERT or HAS_HMARL_META) and FrozenSubPolicy is not None:
        def _mk_subs():
            obs = env._get_defender_state() if role == 'defender' else env._get_attacker_state()
            obs_dim = int(obs.shape[0])

            import torch.nn as nn_local
            class TinyNet(nn_local.Module):
                def __init__(self, d): 
                    super().__init__()
                    self.fc = nn_local.Linear(d, 8)
                def forward(self, x): 
                    return self.fc(x)

            if role == "defender":
                cheap  = FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "CheapLocal", role, [1,5,6,7,9,11])
                costly = FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "CostlyLocal", role, [4,12,13])
                globalp= FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "Global",     role, [2,3,8,10])
                return [cheap, costly, globalp]
            else:
                spread = FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "Spread", role, [1])
                probe  = FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "Probe",  role, [2])
                noop   = FrozenSubPolicy(TinyNet(obs_dim), oracle.device, "Noop",   role, [3])
                return [spread, probe, noop]

        subs = _get_cached(oracle, f"_hmarl_subs_{role}", _mk_subs)
        state_vec = env._get_defender_state() if role == 'defender' else env._get_attacker_state()

        if "hmarl_expert" in tm and HAS_HMARL_EXPERT and ExpertRuleMaster is not None and HMARLExpertBestResponse_cls is not None:
            def _mk_exp():
                master = ExpertRuleMaster(cheaplocal_idx=0, costlylocal_idx=1, global_idx=2, global_prob=0.1)
                return HMARLExpertBestResponse_cls(oracle, role, subs, master, device=oracle.device)
            br = _get_cached(oracle, f"_hmarl_expert_{role}", _mk_exp)
            return br.execute(strat, state_vec, env=env)

        if "hmarl_meta" in tm and HAS_HMARL_META and HMARLMetaBestResponse_cls is not None:
            def _mk_meta():
                st = env._get_defender_state() if role == 'defender' else env._get_attacker_state()
                return HMARLMetaBestResponse_cls(oracle, role, subs, state_dim=int(st.shape[0]), device=oracle.device)
            br = _get_cached(oracle, f"_hmarl_meta_{role}", _mk_meta)
            return br.execute(strat, state_vec, env=env)

    # ---- MetaHierarchical ----
    if tm and 'meta' in tm and HAS_META:
        state_vec = env._get_defender_state() if role=='defender' else env._get_attacker_state()
        br = MetaHierarchicalBestResponse(oracle, role)
        return br.execute(strat, state_vec)

    # ---- HAGS (hierarchical) ----
    if tm and 'hierarchical' in tm and HAS_HAGS:
        state_vec = env._get_defender_state() if role=='defender' else env._get_attacker_state()
        br = HierarchicalBestResponse(oracle, role)
        return br.execute(strat, state_vec)

    # ---- IPPO / MAPPO / generic MARL wrappers ----
    if tm and ('ippo' in tm or 'mappo' in tm or 'marl' in tm):
        state_vec = env._get_defender_state() if role=='defender' else env._get_attacker_state()
        agent = (tm.get('ippo') or tm.get('mappo') or tm.get('marl'))
        return agent.select_action(state_vec, env=env)

    # ---- Fallback: DOAR actor decode path ----
    actor  = strat.load_actor(oracle.defender_ddpg['actor'].__class__, seed=getattr(oracle, "seed", None), device=getattr(oracle, "device", None))
    critic = strat.load_critic(oracle.defender_ddpg['critic'].__class__, seed=getattr(oracle, "seed", None), device=getattr(oracle, "device", None))
    state_vec = env._get_defender_state() if role=='defender' else env._get_attacker_state()
    if HAS_TORCH and actor is not None:
        with torch.no_grad():
            s_t = torch.tensor(state_vec, dtype=torch.float32, device=getattr(oracle, "device", "cpu")).unsqueeze(0)
            raw_vec = actor(s_t).cpu().numpy()[0]
    else:
        raw_vec = np.zeros_like(state_vec, dtype=np.float32)
        s_t = None
    action = oracle.decode_action(
        raw_vec,
        num_action_types    = (oracle.n_def_types if role=='defender' else oracle.n_att_types),
        num_device_indices  = oracle.D_init,
        num_exploit_indices = oracle.E_init,
        num_app_indices     = oracle.A_init,
        state_tensor        = s_t if HAS_TORCH else None,
        actor               = actor,
        critic              = critic
    )
    return action

# ====================== Training wrappers ============================
def train_doar_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    pool = oracle.attacker_strategies if role == 'defender' else oracle.defender_strategies
    mix = np.ones(len(pool), dtype=float); mix /= mix.sum()
    t0 = time.perf_counter()
    strat = oracle.ddpg_best_response(
        opponent_strategies  = pool,
        opponent_equilibrium = mix,
        role                 = role,
        training_steps       = steps
    )
    t1 = time.perf_counter()
    return strat, (t1 - t0)

def train_ippo_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    if not HAS_IPPO:
        raise RuntimeError("IPPO modules not available")
    pool = oracle.attacker_strategies if role == 'defender' else oracle.defender_strategies
    mix = np.ones(len(pool), dtype=float); mix /= mix.sum()
    t0 = time.perf_counter()
    trainer = IPPOCommBestResponse(oracle, role)
    strat = trainer.train(
        opponent_strategies  = pool,
        opponent_equilibrium = mix,
        T                    = steps,
        budget_type          = "steps",
        budget               = steps,
    )
    t1 = time.perf_counter()
    return strat, (t1 - t0)

def train_mappo_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    if not HAS_MAPPO or MAPPOTrainer is None:
        raise RuntimeError("MAPPO trainer not available")
    pool = oracle.attacker_strategies if role == 'defender' else oracle.defender_strategies
    mix = np.ones(len(pool), dtype=float); mix /= mix.sum()
    t0 = time.perf_counter()
    trainer = MAPPOTrainer(oracle, role)
    strat = trainer.train(
        opponent_strategies  = pool,
        opponent_equilibrium = mix,
        T                    = steps,
        budget_type          = "steps",
        budget               = steps,
    )
    t1 = time.perf_counter()
    return strat, (t1 - t0)

# NEW: force MetaHierarchical to use DOAR (no IPPO)
def train_meta_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    if not HAS_META:
        raise RuntimeError("MetaHierarchicalBestResponse not available")

    try:
        setattr(oracle, "BR_type", "ddpg")
    except Exception:
        pass

    pool = oracle.attacker_strategies if role == 'defender' else oracle.defender_strategies
    mix = np.ones(len(pool), dtype=float); mix /= mix.sum()
    meta = MetaHierarchicalBestResponse(oracle, role=role)

    t0 = time.perf_counter()
    last_err = None
    for kwargs in [
        dict(opponent_strategies=pool, opponent_equilibrium=mix, T=steps, T_low=steps, low_level="doar"),
        dict(opponent_strategies=pool, opponent_equilibrium=mix, T=steps, T_low=steps, mode="doar"),
        dict(opponent_strategies=pool, opponent_equilibrium=mix, T=steps, T_low=steps, br_type="ddpg"),
        dict(opponent_strategies=pool, opponent_equilibrium=mix, T=steps, T_low=steps),
        dict(opponent_strategies=pool, opponent_equilibrium=mix, T=steps),
    ]:
        try:
            strat = meta.train(**kwargs)  # type: ignore[arg-type]
            break
        except TypeError as e:
            last_err = e
            continue
    else:
        raise last_err if last_err else RuntimeError("MetaHierarchicalBestResponse.train() call failed")

    t1 = time.perf_counter()
    return strat, (t1 - t0)

# ----------------- Gym 4-return compatibility proxy -------------------
class _Step4EnvProxy:
    """
    Wraps an env so that step/reset look like the older Gym 4-return API.
      - step: (obs, reward, done, info)
      - reset: obs
    If the underlying env already returns 4, we pass through unchanged.
    """
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, *args, **kwargs):
        out = self._env.step(*args, **kwargs)
        # 5-return -> trim to 4
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
            return obs, reward, done, info
        return out

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        # Newer Gym returns (obs, info)
        if isinstance(out, tuple) and len(out) >= 1:
            return out[0]
        return out

# (HMARL helpers kept for router completeness)
def _hmarl_build_subpolicies(oracle: DoubleOracle, role: str) -> Tuple[List, int]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available for HMARL baselines")
    if FrozenSubPolicy is None:
        raise RuntimeError("FrozenSubPolicy not available for HMARL baselines")

    import torch.nn as nn_local
    state_vec = oracle.env._get_defender_state() if role.lower() == "defender" else oracle.env._get_attacker_state()
    obs_dim = int(state_vec.shape[0])

    class DummySubpolicyNet(nn_local.Module):
        def __init__(self, d): 
            super().__init__()
            self.fc = nn_local.Linear(d, 8)
        def forward(self, x): 
            return self.fc(x)

    if role.lower() == "defender":
        subs = [
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "LowCostExpert", role, [1,5,6,7,9,11]),
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "HighCostExpert", role, [4,12,13]),
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "GlobalExpert", role, [2,3,8,10]),
        ]
    else:
        subs = [
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "Spread", role, [1]),
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "Probe",  role, [2]),
            FrozenSubPolicy(DummySubpolicyNet(obs_dim), oracle.device, "Noop",   role, [3]),
        ]
    return subs, obs_dim

def train_hmarl_expert_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    if not HAS_HMARL_EXPERT or HMARLExpertBestResponse_cls is None or ExpertRuleMaster is None:
        raise RuntimeError("HMARLExpertBestResponse or ExpertRuleMaster not available")
    opp_pool = oracle.attacker_strategies if role.lower() == "defender" else oracle.defender_strategies
    opp_mix = np.ones(len(opp_pool), dtype=float); opp_mix /= opp_mix.sum()
    subpolicies, _ = _hmarl_build_subpolicies(oracle, role=role)
    master = ExpertRuleMaster(cheaplocal_idx=0, costlylocal_idx=1, global_idx=2, global_prob=0.1)
    br_agent = HMARLExpertBestResponse_cls(oracle, role, subpolicies, master, device=oracle.device)
    t0 = time.perf_counter()
    strat_obj = br_agent.train(opp_pool, opp_mix, T=steps, return_meta=True)
    t1 = time.perf_counter()
    # Accept any return arity; always lift to Strategy
    if isinstance(strat_obj, tuple):
        strat = strat_obj[0]
    else:
        strat = strat_obj
    return strat, (t1 - t0)


def train_hmarl_meta_br(oracle: DoubleOracle, role: str, steps: int) -> Tuple[Strategy, float]:
    """
    HMARLMeta trainer with stable construction and guaranteed timing.

    Key points:
      - Do NOT monkey-patch oracle.env before constructing HMARLMetaBestResponse.
        BaseHMARLBR.__init__ deep-copies oracle.env; if env is a proxy with a
        custom __getattr__, deepcopy can recurse and explode.
      - Instead, construct HMARLMetaBestResponse against the real env, then wrap
        its *internal* env with _Step4EnvProxy so its rollouts see a 4-return API.
      - Always return (Strategy, train_time_seconds). If HMARLMeta is unusable,
        fall back to a DOAR BR so the benchmark still has a training-time entry.
    """
    if not HAS_HMARL_META or HMARLMetaBestResponse_cls is None:
        raise RuntimeError("HMARLMetaBestResponse not available")

    base_env = getattr(oracle, "env", None)
    if base_env is None:
        raise RuntimeError("Oracle has no env bound")

    # Opponent pool / mix
    opp_pool = oracle.attacker_strategies if role.lower() == "defender" else oracle.defender_strategies
    if not opp_pool:
        raise RuntimeError("Opponent strategy pool is empty for HMARLMeta")
    opp_mix = np.ones(len(opp_pool), dtype=float)
    opp_mix /= opp_mix.sum()

    # Subpolicies & obs_dim from the *real* oracle.env
    subpolicies, obs_dim = _hmarl_build_subpolicies(oracle, role=role)

    # ---- Instantiate HMARLMetaBestResponse WITHOUT env-proxy shenanigans ----
    br_agent = None
    last_err: Optional[Exception] = None
    ctor_attempts = [
        dict(oracle=oracle, role=role, subpolicies=subpolicies,
             state_dim=obs_dim, device=getattr(oracle, "device", None)),
        dict(oracle=oracle, role=role, subpolicies=subpolicies,
             state_dim=obs_dim),
        dict(oracle=oracle, role=role, subpolicies=subpolicies),
    ]

    for kw in ctor_attempts:
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        try:
            br_agent = HMARLMetaBestResponse_cls(**clean_kw)
            break
        except TypeError as e:
            # Signature mismatch, try next pattern.
            last_err = e
            continue
        except RecursionError as e:
            # Constructor itself is recursively broken; don't retry patterns that
            # just reshuffle args — bail out to fallback below.
            last_err = e
            br_agent = None
            break
        except Exception as e:
            last_err = e
            continue

    if br_agent is None:
        # As an optional baseline we *could* silently skip, but we want a time
        # entry; fall back to DOAR so CSV/table stay populated.
        print("   HMARLMeta: ctor failed; falling back to DOAR BR for timing.", flush=True)
        return train_doar_br(oracle, role, steps)

    # ---- Wrap HMARLMeta's internal env with 4-return proxy (safe now) ----
    try:
        if hasattr(br_agent, "env") and br_agent.env is not None:
            br_agent.env = _Step4EnvProxy(br_agent.env)
    except Exception:
        # Not fatal; worst case HMARLMeta sees 5-return and may raise cleanly.
        pass

    # ---- Train HMARLMeta and measure time ----
    t0 = time.perf_counter()
    ret = None
    last_err = None

    train_attempts = [
        dict(opponent_strategies=opp_pool, opponent_equilibrium=opp_mix,
             T=steps, return_meta=True),
        dict(opponent_strategies=opp_pool, opponent_equilibrium=opp_mix,
             T=steps),
    ]

    for kw in train_attempts:
        clean_kw = {k: v for k, v in kw.items() if v is not None}
        try:
            ret = br_agent.train(**clean_kw)
            break
        except TypeError as e:
            # Signature mismatch; try next calling pattern.
            last_err = e
            continue
        except RecursionError as e:
            # Training loop went recursive (e.g., env misuse). Fall back but keep timing.
            last_err = e
            print("   HMARLMeta: recursion during training; falling back to DOAR BR for this baseline.", flush=True)
            # Measure DOAR fallback as this algo's time estimate.
            return train_doar_br(oracle, role, steps)
        except Exception as e:
            last_err = e
            continue

    t1 = time.perf_counter()

    if ret is None:
        # Could not train HMARLMeta successfully; fall back while still giving a time.
        print(f"   HMARLMeta: train() failed ({last_err}); falling back to DOAR BR.", flush=True)
        return train_doar_br(oracle, role, steps)

    # Normalize output to Strategy
    if isinstance(ret, tuple) and len(ret) >= 1:
        strat = ret[0]
    else:
        strat = ret

    # Prefer an explicit time if HMARLMeta ever returns (strat, time)
    if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], (int, float)):
        train_time = float(ret[1])
    else:
        train_time = float(t1 - t0)

    return strat, train_time


# ========================= Actor discovery for timing =================
def _try_get_actor_from_strategy(strat: Strategy, oracle: DoubleOracle, verbose: bool = False):
    tm = getattr(strat, "type_mapping", {}) or {}
    if "hmarl_expert" in tm or "hmarl_meta" in tm:
        return None, None, "hmarl-strategy-no-actor"
    if not HAS_TORCH:
        return None, None, "no-torch"

    for dev_try in ("cpu", getattr(oracle, "device", None), None):
        try:
            if hasattr(strat, "load_actor"):
                dev_arg = torch.device("cpu") if dev_try == "cpu" else dev_try
                if dev_arg is None:
                    actor = strat.load_actor(oracle.defender_ddpg['actor'].__class__, seed=getattr(oracle, "seed", None))
                else:
                    try:
                        actor = strat.load_actor(oracle.defender_ddpg['actor'].__class__, seed=getattr(oracle, "seed", None), device=dev_arg)
                    except TypeError:
                        actor = strat.load_actor(oracle.defender_ddpg['actor'].__class__, seed=getattr(oracle, "seed", None))
                if actor is not None:
                    try:
                        actor_cpu = actor.to(torch.device("cpu"))
                    except Exception:
                        actor_cpu = actor
                    cls = oracle.defender_ddpg['actor'].__class__ if getattr(oracle, "defender_ddpg", None) else actor.__class__
                    return actor_cpu, cls, f"loaded-via-load_actor(dev_try={dev_try})"
        except Exception as e:
            if verbose:
                print(f"[actor-find] load_actor(dev={dev_try}) failed: {e}", flush=True)

    for attr in ("actor", "policy", "model"):
        try:
            maybe = getattr(strat, attr, None)
            if maybe is not None:
                if HAS_TORCH and isinstance(maybe, torch.nn.Module):
                    try:
                        actor_cpu = maybe.to(torch.device("cpu"))
                    except Exception:
                        actor_cpu = maybe
                    return actor_cpu, type(maybe), f"found-attr-{attr}"
                if isinstance(maybe, dict) and getattr(oracle, "defender_ddpg", None):
                    actor_cls = oracle.defender_ddpg['actor'].__class__
                    try:
                        inst = actor_cls()
                        inst.load_state_dict(maybe)
                        if HAS_TORCH:
                            inst.to(torch.device("cpu"))
                        return inst, actor_cls, f"reconstructed-from-{attr}-state_dict"
                    except Exception as e:
                        if verbose:
                            print(f"[actor-find] reconstruct from {attr} failed: {e}", flush=True)
        except Exception:
            pass

    try:
        actor_cls = oracle.defender_ddpg['actor'].__class__ if getattr(oracle, "defender_ddpg", None) else None
        for name in ("actor_state_dict", "state_dict", "policy_state_dict"):
            sd = getattr(strat, name, None)
            if isinstance(sd, dict) and actor_cls is not None:
                try:
                    inst = actor_cls()
                    inst.load_state_dict(sd)
                    if HAS_TORCH:
                        inst.to(torch.device("cpu"))
                    return inst, actor_cls, f"reconstructed-from-strat.{name}"
                except Exception as e:
                    if verbose:
                        print(f"[actor-find] reconstruct from strat.{name} failed: {e}", flush=True)
    except Exception:
        pass

    return None, None, "no-actor-found"

# ============== Forward-pass timing averaged across roles ==============
def _nanmean(lst):
    vals = [v for v in lst if (isinstance(v, (float, int)) and not (math.isnan(v) or math.isinf(v)))]
    return statistics.mean(vals) if vals else float('nan')

def _measure_role_forward(env: Volt_Typhoon_CyberDefenseEnv,
                          oracle: DoubleOracle,
                          strat: Strategy,
                          role: str,
                          n_steps: int,
                          verbose: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], int]:
    env.mode = role
    try:
        env.initialize_environment()
    except Exception:
        pass

    actor_cpu, actor_cls, _reason = _try_get_actor_from_strategy(strat, oracle, verbose=verbose)
    use_gpu = (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None)
    actor_gpu = None
    if actor_cpu is not None and use_gpu:
        try:
            state = None
            try:
                state = actor_cpu.state_dict()
            except Exception:
                state = None
            dev = torch.device(f"cuda:{SELECTED_CUDA_DEVICE}")
            if state is not None and actor_cls is not None:
                try:
                    actor_gpu = actor_cls(); actor_gpu.load_state_dict(state); actor_gpu.to(dev)
                except Exception:
                    actor_gpu = copy.deepcopy(actor_cpu).to(dev)
            else:
                try:
                    actor_gpu = copy.deepcopy(actor_cpu).to(dev)
                except Exception:
                    try:
                        actor_gpu = actor_cpu.to(dev)
                    except Exception:
                        actor_gpu = None
            if actor_gpu is not None:
                actor_gpu.eval()
        except Exception:
            actor_gpu = None

    env_ms_list, cpu_ms_list, gpu_ms_list, e2e_cpu_list, e2e_gpu_list = [], [], [], [], []
    if WARMUP_STEPS > 0:
        for _ in range(WARMUP_STEPS):
            env.mode = role
            _ = execute_one_step(oracle, env, strat, role)

    n_eff = 0
    for _ in range(n_steps):
        env.mode = role

        # env feature assembly timing
        t0 = time.perf_counter()
        try:
            state_vec = env._get_defender_state() if role == 'defender' else env._get_attacker_state()
        except Exception:
            state_vec = np.zeros(1, dtype=np.float32)
        t1 = time.perf_counter()
        env_ms = (t1 - t0) * 1e3

        cpu_ms = float('nan'); gpu_ms = float('nan'); e2e_cpu_ms = float('nan'); e2e_gpu_ms = float('nan')

        if actor_cpu is None:
            # No plain actor — time execute_one_step path (covers HMARL etc.)
            try:
                if HAS_TORCH and torch.cuda.is_available():
                    try:
                        if SELECTED_CUDA_DEVICE is not None:
                            torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                        else:
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                t0x = time.perf_counter()
                _ = execute_one_step(oracle, env, strat, role)
                if HAS_TORCH and torch.cuda.is_available():
                    try:
                        if SELECTED_CUDA_DEVICE is not None:
                            torch.cuda.synchronize(SELECTED_CUDA_DEVICE)
                        else:
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                t1x = time.perf_counter()
                cpu_ms = (t1x - t0x) * 1e3
                e2e_cpu_ms = env_ms + cpu_ms
            except Exception:
                pass
        else:
            # CPU forward
            try:
                actor_cpu.eval()
                s_cpu = torch.tensor(state_vec, dtype=torch.float32, device=torch.device("cpu")).unsqueeze(0)
                t0c = time.perf_counter()
                with torch.no_grad():
                    _ = actor_cpu(s_cpu)
                t1c = time.perf_counter()
                cpu_ms = (t1c - t0c) * 1e3
                e2e_cpu_ms = env_ms + cpu_ms
            except Exception:
                pass

            # GPU forward (only if we cloned to GPU)
            if actor_gpu is not None and use_gpu:
                try:
                    dev = torch.device(f"cuda:{SELECTED_CUDA_DEVICE}")
                    s_gpu = torch.tensor(state_vec, dtype=torch.float32, device=dev).unsqueeze(0)
                    try:
                        torch.cuda.synchronize(dev)
                    except Exception:
                        pass
                    t0g = time.perf_counter()
                    with torch.no_grad():
                        _ = actor_gpu(s_gpu)
                    try:
                        torch.cuda.synchronize(dev)
                    except Exception:
                        pass
                    t1g = time.perf_counter()
                    gpu_ms = (t1g - t0g) * 1e3
                    e2e_gpu_ms = env_ms + gpu_ms
                except Exception:
                    pass

        env_ms_list.append(env_ms)
        cpu_ms_list.append(cpu_ms)
        gpu_ms_list.append(gpu_ms)
        e2e_cpu_list.append(e2e_cpu_ms)
        e2e_gpu_list.append(e2e_gpu_ms)
        n_eff += 1

    return env_ms_list, cpu_ms_list, gpu_ms_list, e2e_cpu_list, e2e_gpu_list, n_eff

def time_forward_pass(env: Volt_Typhoon_CyberDefenseEnv,
                      oracle: DoubleOracle,
                      strat: Strategy,
                      n_steps_total: int,
                      verbose: bool = False) -> Tuple[Tuple[float,float,float,float,float], int]:
    """
    Average timing over BOTH roles.
    Runs floor(N/2) steps defender and ceil(N/2) steps attacker, combines, and returns
    (env_ms, cpu_forward_ms, gpu_forward_ms, e2e_cpu_ms, e2e_gpu_ms) means (NaN-robust) and n_effective.
    """
    n_steps_total = max(1, int(n_steps_total))
    n_def = n_steps_total // 2
    n_att = n_steps_total - n_def

    d_env, d_cpu, d_gpu, d_e2e_cpu, d_e2e_gpu, n_d = _measure_role_forward(env, oracle, strat, "defender", n_def, verbose=verbose)
    a_env, a_cpu, a_gpu, a_e2e_cpu, a_e2e_gpu, n_a = _measure_role_forward(env, oracle, strat, "attacker", n_att, verbose=verbose)

    env_all = d_env + a_env
    cpu_all = d_cpu + a_cpu
    gpu_all = d_gpu + a_gpu
    e2e_cpu_all = d_e2e_cpu + a_e2e_cpu
    e2e_gpu_all = d_e2e_gpu + a_e2e_gpu

    return (_nanmean(env_all), _nanmean(cpu_all), _nanmean(gpu_all), _nanmean(e2e_cpu_all), _nanmean(e2e_gpu_all)), (n_d + n_a)

# ======================= Algo suites (phase split) ====================
NON_MARL_ALGOS = OrderedDict([
    ("DOAR",  dict(train=lambda o, env_for_paths, size_label: train_doar_br(o, ROLE, TRAIN_STEPS))),
    ("HAGS",  dict(train=lambda o, env_for_paths, size_label: train_hags_br(o, ROLE, TRAIN_STEPS, env_for_paths=env_for_paths, size_label=size_label),
                   optional=True)),
    ("Meta",  dict(train=lambda o, env_for_paths, size_label: train_meta_br(o, ROLE, TRAIN_STEPS))),
])

MARL_ALGOS = OrderedDict([
    ("IPPO",        dict(train=lambda o, env_for_paths, size_label: train_ippo_br(o, ROLE, TRAIN_STEPS), optional=True)),
    ("MAPPO",       dict(train=lambda o, env_for_paths, size_label: train_mappo_br(o, ROLE, TRAIN_STEPS), optional=True)),
    ("HMARLExpert", dict(train=lambda o, env_for_paths, size_label: train_hmarl_expert_br(o, ROLE, TRAIN_STEPS), optional=True)),
    ("HMARLMeta",   dict(train=lambda o, env_for_paths, size_label: train_hmarl_meta_br(o, ROLE, TRAIN_STEPS), optional=True)),
])

if ONLY_HMARL:
    NON_MARL_ALGOS = OrderedDict([])
    MARL_ALGOS = OrderedDict([
        ("HMARLExpert", dict(train=lambda o, env_for_paths, size_label: train_hmarl_expert_br(o, ROLE, TRAIN_STEPS), optional=True)),
        ("HMARLMeta",   dict(train=lambda o, env_for_paths, size_label: train_hmarl_meta_br(o, ROLE, TRAIN_STEPS), optional=True)),
    ])

# ================== Snapshot-bound oracle factory =====================
def make_oracle_bound_to_snapshot(snap_path: str) -> Tuple[DoubleOracle, Volt_Typhoon_CyberDefenseEnv]:
    env = load_env_from_snapshot(snap_path)
    do = DoubleOracle(
        env=env,
        num_episodes=1,
        steps_per_episode=max(2*N_STEPS_EXEC, 64),
        seed=SEED,
        baseline=BASELINE,
        dynamic_neighbor_search=False,
        BR_type="ddpg",
        zero_day=ZERO_DAY
    )
    try:
        if HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
            dev = torch.device(f"cuda:{SELECTED_CUDA_DEVICE}")
            setattr(do, "device", dev)
    except Exception:
        pass

    def _fresh():
        return load_env_from_snapshot(snap_path)

    setattr(do, "fresh_env", _fresh)
    setattr(do, "env", env)
    setattr(env, "snapshot_path", snap_path)
    safe_do_checkpoint(do)
    return do, env

# =========================== Runner helper ============================
def _run_suite_for_algos(snap_path: str,
                         M: int,
                         algos: "OrderedDict[str, dict]",
                         phase_label: str,
                         wt, we) -> Dict[str, Tuple[float, float, int]]:
    per_algo_exec_stats: Dict[str, Tuple[float, float, int]] = {}

    print(f"\n—— {phase_label}: training + forward-pass (avg of defender & attacker) — M={M} ——", flush=True)

    for name, cfg in algos.items():
        is_optional = cfg.get("optional", False)
        print(f"\n→ Training {name} BR for {N_TRAIN_RUNS} runs…", flush=True)

        # Track training times for mean±std
        train_times_sec: List[float] = []

        # Track training memory peaks across runs
        train_host_rss_max_over_runs: Optional[int] = 0 if HAS_PSUTIL else None
        train_cuda_alloc_max_over_runs: Optional[int] = 0 if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None
        train_cuda_reserved_max_over_runs: Optional[int] = 0 if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None

        last_strat: Optional[Strategy] = None
        last_do: Optional[DoubleOracle] = None

        try:
            for r in range(N_TRAIN_RUNS):
                set_seeds(SEED + r)

                do, env_for_paths = make_oracle_bound_to_snapshot(snap_path)
                try:
                    if HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
                        setattr(do, "device", torch.device(f"cuda:{SELECTED_CUDA_DEVICE}"))
                except Exception:
                    pass

                size_label = f"M{M}_run{r+1}"

                print(f"   run {r+1}/{N_TRAIN_RUNS}: starting training (monitored)...", flush=True)
                t0 = time.perf_counter()
                ret, train_mem = run_with_memory_monitor(cfg["train"], do, env_for_paths, size_label)
                t1 = time.perf_counter()

                # Preferred: trainer returned (strategy, time_seconds)
                if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], (int, float)):
                    t_train = float(ret[1])
                    strat = ret[0]
                else:
                    # Fallback to wall-clock around the call
                    t_train = float(t1 - t0)
                    # If trainer returned (strategy, *extra) without time in slot 2, use first as strategy
                    if isinstance(ret, tuple) and len(ret) >= 1:
                        strat = ret[0]
                    else:
                        strat = ret

                train_times_sec.append(t_train)

                # update tracker maxima across runs
                if HAS_PSUTIL:
                    hr = train_mem.get('host_rss_max')
                    if hr is not None:
                        if train_host_rss_max_over_runs is None or hr > train_host_rss_max_over_runs:
                            train_host_rss_max_over_runs = hr
                if HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None:
                    ca = train_mem.get('cuda_alloc_max')
                    cr = train_mem.get('cuda_reserved_max')
                    if ca is not None:
                        if train_cuda_alloc_max_over_runs is None or ca > train_cuda_alloc_max_over_runs:
                            train_cuda_alloc_max_over_runs = ca
                    if cr is not None:
                        if train_cuda_reserved_max_over_runs is None or cr > train_cuda_reserved_max_over_runs:
                            train_cuda_reserved_max_over_runs = cr

                last_strat = strat if not (isinstance(strat, tuple) and not strat) else last_strat
                last_do = do
                print(f"   run {r+1}/{N_TRAIN_RUNS}: training finished", flush=True)

            # === NEW: write training time stats to CSV and stdout ===
            if train_times_sec:
                mean_s = statistics.mean(train_times_sec)
                std_s  = statistics.stdev(train_times_sec) if len(train_times_sec) > 1 else 0.0
                wt.writerow([f"{phase_label}:{name}", M, len(train_times_sec), f"{mean_s:.6f}", f"{std_s:.6f}"])
                print(f"   {name}: training time {mean_s:.6f} ± {std_s:.6f} s over {len(train_times_sec)} run(s)", flush=True)
            else:
                wt.writerow([f"{phase_label}:{name}", M, 0, "ERR", "ERR"])
                print(f"   {name}: training time not recorded (ERR)", flush=True)

            if last_strat is None or last_do is None:
                raise RuntimeError("No trained strategy available to time.")
            env_exec = last_do.fresh_env()
            env_exec.base_line = BASELINE

            print(f"   {name}: starting forward-pass (avg roles; monitored, {N_STEPS_EXEC} total steps)…", flush=True)
            (res_tuple, n_ok), infer_mem = run_with_memory_monitor(time_forward_pass, env_exec, last_do, last_strat, N_STEPS_EXEC)
            env_ms, cpu_forward_ms, gpu_forward_ms, e2e_cpu_ms, e2e_gpu_ms = res_tuple
            print(f"   {name}: forward-pass done (n={n_ok})", flush=True)

            we.writerow([f"{phase_label}:{name}", M, f"{env_ms:.6f}", f"{cpu_forward_ms:.6f}", f"{gpu_forward_ms:.6f}", f"{e2e_cpu_ms:.6f}", f"{e2e_gpu_ms:.6f}", n_ok])

            per_algo_exec_stats[name] = (
                e2e_cpu_ms if not math.isnan(e2e_cpu_ms) else (e2e_gpu_ms if not math.isnan(e2e_gpu_ms) else float('nan')),
                0.0,
                n_ok
            )

            # Print memory peaks (MB) like before
            infer_host_rss_max = infer_mem.get('host_rss_max') if HAS_PSUTIL else None
            infer_cuda_alloc_max = infer_mem.get('cuda_alloc_max') if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None
            infer_cuda_reserved_max = infer_mem.get('cuda_reserved_max') if (HAS_TORCH and torch.cuda.is_available() and SELECTED_CUDA_DEVICE is not None) else None

            th = bytes_to_mb(train_host_rss_max_over_runs) if train_host_rss_max_over_runs not in (None, 0) else float('nan')
            tia = bytes_to_mb(train_cuda_alloc_max_over_runs) if train_cuda_alloc_max_over_runs not in (None, 0) else float('nan')
            tir = bytes_to_mb(train_cuda_reserved_max_over_runs) if train_cuda_reserved_max_over_runs not in (None, 0) else float('nan')
            ih = bytes_to_mb(infer_host_rss_max) if infer_host_rss_max not in (None, 0) else float('nan')
            iia = bytes_to_mb(infer_cuda_alloc_max) if infer_cuda_alloc_max not in (None, 0) else float('nan')
            iir = bytes_to_mb(infer_cuda_reserved_max) if infer_cuda_reserved_max not in (None, 0) else float('nan')

            print(f"   {name} memory peaks (MB):", flush=True)
            print(f"      training  host_rss_max={th:.1f} MB  cuda_alloc_max={tia:.1f} MB  cuda_reserved_max={tir:.1f} MB", flush=True)
            print(f"      forward   host_rss_max={ih:.1f} MB  cuda_alloc_max={iia:.1f} MB  cuda_reserved_max={iir:.1f} MB", flush=True)

        except Exception as e:
            tag = "skipped" if is_optional else "FAILED"
            print(f"   {tag}: {e}", flush=True)
            wt.writerow([f"{phase_label}:{name}", M, 0, "NA" if is_optional else "ERR", "NA" if is_optional else "ERR"])
            we.writerow([f"{phase_label}:{name}", M, "NA" if is_optional else "ERR", "NA" if is_optional else "ERR", "NA", "NA", "NA", 0])

    print(f"\n=== {phase_label} summary (forward-pass e2e ms, avg roles) — M={M} ===", flush=True)
    for name, (mean_ms, _, n_ok) in per_algo_exec_stats.items():
        print(f"{name:<12}  {mean_ms:9.2f} ms  (n={n_ok})", flush=True)

    return per_algo_exec_stats

# ============================== MAIN ================================
def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark BR training + forward-pass timing (timings averaged over defender & attacker).")
    ap.add_argument("--size", type=int, default=None, help="Single network size; overrides SIZES to [size].")
    ap.add_argument("--role", type=str, default=None, choices=["defender", "attacker"], help="Role to TRAIN (timing always averages both).")
    ap.add_argument("--only-hmarl", action="store_true", help="Run only HMARLExpert/HMARLMeta baselines.")
    ap.add_argument("--steps", type=int, default=None, help="Total timing steps (N). Runs N/2 defender + N/2 attacker. Default 4.")
    return ap.parse_args()

def main():
    global SIZES, ROLE, ONLY_HMARL, N_STEPS_EXEC

    args = parse_args()
    if args.size is not None:
        SIZES = [int(args.size)]
    if args.role is not None:
        ROLE = args.role
    if args.only_hmarl:
        ONLY_HMARL = True
    if args.steps is not None:
        N_STEPS_EXEC = max(1, int(args.steps))

    set_seeds(SEED)
    select_cuda_device_from_env()
    print_memory_snapshot("startup")

    print(f"\nBenchmarking ONE best response per algorithm (training role={ROLE})")
    print(f"Forward-pass timings use N={N_STEPS_EXEC} total steps averaged over defender & attacker")
    print(f"Trial tag: {TRIAL_TAG}")
    print(f"Snapshots in: {SNAP_DIR}\n")

    size_to_snap = {}
    for M in SIZES:
        print_memory_snapshot(f"before_create_snapshot_M{M}")
        snap = create_snapshot_for_size(M, TRIAL_TAG)
        size_to_snap[M] = snap
        print(f"✅ created snapshot for M={M}: {snap}", flush=True)
        print_memory_snapshot(f"after_create_snapshot_M{M}")

    train_csv_path = os.path.join(TRIAL_DIR, "train_times.csv")
    exec_csv_path  = os.path.join(TRIAL_DIR, "exec_times.csv")

    with open(train_csv_path, "w", newline="") as ft, open(exec_csv_path, "w", newline="") as fe:
        wt = csv.writer(ft); we = csv.writer(fe)
        wt.writerow(["algo", "num_devices", "n_runs", "train_mean_s", "train_std_s"])
        we.writerow(["algo", "num_devices", "env_ms", "cpu_forward_ms", "gpu_forward_ms", "e2e_cpu_ms", "e2e_gpu_ms", "n_effective"])

        for M in SIZES:
            clear_caches_and_memory()
            print_memory_snapshot(f"after_clear_M{M}")
            print(f"\n==== M = {M} devices ====")
            snap_path = size_to_snap[M]

            if not ONLY_HMARL:
                print_memory_snapshot(f"pre_phase1_M{M}")
                _ = _run_suite_for_algos(snap_path, M, NON_MARL_ALGOS, "Phase1-NonMARL", wt, we)
                ft.flush(); fe.flush()

            print_memory_snapshot(f"pre_phase2_M{M}")
            _ = _run_suite_for_algos(snap_path, M, MARL_ALGOS, "Phase2-MARL", wt, we)
            ft.flush(); fe.flush()

            clear_caches_and_memory()
            print_memory_snapshot(f"post_run_clear_M{M}")

    try:
        data: Dict[str, List[Tuple[int,float]]] = defaultdict(list)
        with open(exec_csv_path, "r") as f:
            rd = csv.DictReader(f)
            for row in rd:
                algo = row["algo"]
                if row["e2e_cpu_ms"] in ("NA", "ERR"): 
                    continue
                m = int(row["num_devices"])
                try:
                    mean_ms = float(row["e2e_cpu_ms"])
                except Exception:
                    mean_ms = float('nan')
                data[algo].append((m, mean_ms))

        plt.figure(figsize=(8, 5))
        plotted_any = False
        for algo, pts in data.items():
            pts = sorted(pts, key=lambda x: x[0])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker='o', label=algo)
            plotted_any = True

        plt.yscale('log')
        plt.xlabel("Network size (numOfDevice)")
        plt.ylabel(f"Forward-pass e2e CPU time (ms) over {N_STEPS_EXEC} total calls (avg roles)")
        plt.title(f"Algorithm forward-pass scaling (trial {TRIAL_TAG})")
        plt.grid(True, which='both', linestyle=':')
        if plotted_any:
            plt.legend()
        plt.tight_layout()
        out_png = os.path.join(TRIAL_DIR, "exec_scaling_log.png")
        plt.savefig(out_png)
        print(f"\n↳ Saved plot to {out_png}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"\n↳ Training times CSV: {train_csv_path}")
    print(f"↳ Exec (forward-pass, avg roles) CSV: {exec_csv_path}")

    if CLEANUP_SNAPSHOTS:
        try:
            shutil.rmtree(TRIAL_DIR)
            print(f"🧹 Cleaned up snapshots and outputs for trial {TRIAL_TAG} at {TRIAL_DIR}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

# ---- HAGS placeholder (unchanged) ----
def train_hags_br(oracle: DoubleOracle, role: str, steps: int, env_for_paths=None, size_label:str="") -> Tuple[Strategy, float]:
    # Minimal fallback to DOAR BR to keep pipeline flowing.
    return train_doar_br(oracle, role, steps)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
