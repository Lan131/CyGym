# volt_typhoon_env.py
import gym
import matplotlib.pyplot as plt
import os

from collections import deque, defaultdict

import numpy as np
from gym import spaces
import random
from CyberDefenseEnv import CyberDefenseEnv
from CDSimulatorComponents import App, Device, OperatingSystem, Workload, os_encoder
import logging
import uuid
import time
import pickle
from datetime import datetime
from pathlib import Path
import copy


VOLT_CVE_ID = "ED3A999C-9184-4D27-A62E-3D8A3F0D4F27"
VOLT_DC_CVE_ID = "0A5713AE-B7C5-4599-8E4F-9C235E73E5F6"

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='cyberdefense_debug.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class Volt_Typhoon_CyberDefenseEnv(CyberDefenseEnv):

    def __init__(self):
        super().__init__()
        self.app_id_mapping = self._create_app_id_mapping()
        self.defaultOS = OperatingSystem(id=0, type="default", version="1.0")
        self.defaultversion = "1.0"
        self.default_mode = 1
        self.default_high = 3
        self.work_scale = 1
        self.comp_scale = 50
        self.num_attacker_owned = 1
        self.base_line = "Nash"
        self.tech = "DQN"
        self.its = 1
        self.intial_ratio_compromise = .4  # Across all devices in the network
        self.fast_scan = True
        self.γ = 0.99
        self._prev_potential = None
        self._prev_att_potential = None
        self.def_scale = 1
        self.scan_cnt = 0
        self.checkpoint_count = 0
        self.defensive_cost = 0
        self.clearning_cost = 0
        self.revert_count = 0
        self.compromised_devices_cnt = 0
        self.edges_blocked = 0
        self.edges_added = 0
        self.partition_size = 10
        self.alpha = 1
        self.khop = 1
        self.snapshot_path = None  # Snapshot env generated via init_experiments (controls topology)

        # --- performance / evolution control ---
        self._topology_dirty = False
        self._evolve_period = max(1, int(round(np.sqrt(self.numOfDevice))))

        # --- graph caches ---
        self._vidx = {}       # name -> vertex index
        self._name = {}       # index -> name
        self._outnbrs = {}    # name -> [neighbor names (outgoing)]
        self._innbrs = {}     # name -> [neighbor names (incoming)]
        self._blocked = set() # set of (src_name, dst_name) tuples
        self._device_ckpts = {}
        self._exp_by_id = {}
        self._rng = np.random.default_rng()

        self._in_batch = False        # are we inside step_grouped?
        self._batch_cost = 0.0        # accumulated action costs during batching

        # --- new scaling flag + tuning params ---
        self.scaling_vulnerability = True  # enable by default
        self._max_version_by_type = {}

        # Tuning parameters
        self.sv_dc_ratio = 50            # devices per DC
        self.sv_attacker_fraction = 0.05 # fraction attacker-owned
        self.sv_apps_base = 3            # base apps/device
        self.sv_apps_per_device = 0.0    # extra apps per device

        # >>> NEW/CHANGED: turbo & workload controls
        self.turbo = False                  # skip heavy ops when True
        self.workload_cap = None            # None=unlimited; 0=skip generation; N=cap per arrival tick
        self.workload_period_base = 50      # base cadence for arrivals
        self.workload_period_max  = 200     # max cadence clamp
        # Turbo throttling / ramp
        self.turbo_fraction_clients = 0.05    # ≤ 5% of active devices per step
        self.turbo_fraction_servers = 0.02    # ≤ 2% of active devices per step
        self.turbo_max_clients = 200          # absolute cap per step (clients)
        self.turbo_max_servers = 40           # absolute cap per step (servers)
        self.turbo_ramp_steps = 200           # ramp to full turbo caps over N steps

        # Tiny “bootstrap” batch so startup is snappy but not empty
        self.turbo_bootstrap_clients = 10
        self.turbo_bootstrap_servers = 2

        # --- turbo training throttle knobs ---
        self.turbo_train_max_logs = 256   # keep training cheap by capping log window
        self.turbo_train_stride   = 2     # downsample logs in turbo (every Nth record)

        # perf dict (lazy-inited as well for pickle safety)
        self._ensure_perf()

        # Threshold where we switch to sparse attacker-connect (tunable)
        self._sparse_threshold = 500

        self._busy_devices = {}


        self.time_budget_deadline = None
        # optional convenience flag:
        

    # --------------------
    # Utilities / helpers
    # --------------------
    def _ensure_perf(self):
        # lazy-init so older pickles / hot-reloads won't error
        if not hasattr(self, "_perf") or self._perf is None:
            self._perf = defaultdict(float)

    def _reindex_exploits(self):
        self._exp_by_id = {exp.id: exp for exp in self.simulator.exploits}

    def _stall(self, low=None, high=None):
        low = self.default_mode if low is None else low
        high = self.default_high if high is None else high
        return random.randint(low, high)

    # >>> NEW/CHANGED: adaptive arrival period & idle fraction
    def _arrival_period(self):
        n_active = sum(1 for d in self.simulator.subnet.net.values() if not d.Not_yet_added)
        base = int(self.workload_period_base)
        # Slow the cadence as network grows, clamp to max
        return min(self.workload_period_max, max(10, int(base + 0.5 * np.sqrt(max(1, n_active)))))


    def _train_detector(self, data):
        """
        Always train, but throttle in turbo mode:
          - Clip to last `turbo_train_max_logs`
          - Downsample by `turbo_train_stride`
        Falls back gracefully if data is None/empty or detector lacks train().
        """
        if not data:
            return
        # Normalize to list
        if not isinstance(data, list):
            try:
                data = list(data)
            except Exception:
                return

        # In turbo: clip + optionally downsample
        if self.turbo:
            data = data[-int(self.turbo_train_max_logs):]
            stride = max(1, int(self.turbo_train_stride))
            if stride > 1:
                data = data[::stride]

        # Finally, call the detector
        train_fn = getattr(self.simulator.detector, "train", None)
        if callable(train_fn):
            try:
                train_fn(data)
            except TypeError:
                # If train has a different signature, attempt a best-effort fallback with no args
                try:
                    train_fn()
                except Exception:
                    pass


    def _idle_fraction(self):
        idle = sum(
            1
            for d in self.simulator.subnet.net.values()
            if not d.Not_yet_added and (getattr(d, "busy_time", 0) <= 0) and not d.workload
        )
        active = sum(1 for d in self.simulator.subnet.net.values() if not d.Not_yet_added)
        return (idle / active) if active else 0.0

    def _generate_workloads_timed(self, *, numLoads: int, mode: int, high: int, wtype: str, bootstrap: bool=False):
        """
        Generate workloads with timing + turbo throttling (not all-or-nothing).
        - If turbo is on, we cap and ramp rather than returning [].
        - Respect a global workload_cap if provided.
        - Never schedule more workloads than active devices.
        """
        self._ensure_perf()
        key = f"gw_{wtype or 'unknown'}"
        self._perf[f"{key}_calls"] += 1.0

        # How many devices are currently active?
        n_active = sum(1 for d in self.simulator.subnet.net.values() if not d.Not_yet_added)
        if n_active <= 0:
            return []

        # Optional global cap (applies in all modes, including turbo)
        if isinstance(self.workload_cap, (int, float)) and self.workload_cap >= 0:
            numLoads = min(numLoads, int(self.workload_cap))

        # Bootstrap: very small initial batch on step 0
        if bootstrap:
            numLoads = self.turbo_bootstrap_clients if wtype == "client" else self.turbo_bootstrap_servers
            numLoads = max(0, min(numLoads, n_active))

        # Turbo throttling (cap + ramp), but NOT zero
        if self.turbo and not bootstrap:
            if wtype == "client":
                frac_cap = max(1, int(self.turbo_fraction_clients * n_active))
                hard_cap = self.turbo_max_clients
            else:
                frac_cap = max(1, int(self.turbo_fraction_servers * n_active))
                hard_cap = self.turbo_max_servers

            # Start small and ramp up toward min(frac_cap, hard_cap)
            ramp_alpha = min(1.0, max(0.0, float(self.step_num) / float(max(1, self.turbo_ramp_steps))))
            turbo_cap = max(1, int(round(min(frac_cap, hard_cap) * ramp_alpha)))

            numLoads = min(numLoads, turbo_cap)

        # Extra sanity: don't exceed #active devices
        numLoads = max(0, min(int(numLoads), n_active))

        # If still zero, skip the real generator (but keep perf counters)
        if numLoads <= 0:
            return []

        t0 = time.perf_counter()
        out = self.simulator.generate_workloads(numLoads=numLoads, mode=mode, high=high, wtype=wtype)
        dt = time.perf_counter() - t0
        self._perf[f"{key}_time"] = self._perf.get(f"{key}_time", 0.0) + dt
        self._perf["gw_max_dt"] = max(self._perf.get("gw_max_dt", 0.0), dt)
        return out

    def workload_timing_stats(self):
        """
        Returns a small dict of workload generation timing stats (ms).
        """
        self._ensure_perf()
        c_calls = int(self._perf.get("gw_client_calls", 0))
        s_calls = int(self._perf.get("gw_server_calls", 0))
        c_time = float(self._perf.get("gw_client_time", 0.0))
        s_time = float(self._perf.get("gw_server_time", 0.0))
        return {
            "client_calls": c_calls,
            "client_total_ms": round(c_time * 1000.0, 2),
            "client_avg_ms": round((c_time / c_calls) * 1000.0, 2) if c_calls else 0.0,
            "server_calls": s_calls,
            "server_total_ms": round(s_time * 1000.0, 2),
            "server_avg_ms": round((s_time / s_calls) * 1000.0, 2) if s_calls else 0.0,
            "max_call_ms": round(self._perf.get("gw_max_dt", 0.0) * 1000.0, 2),
        }

    def _scaled_numloads(self, base_clients: int = 100, base_servers: int = 10) -> tuple[int, int]:
        """
        Scale requested workloads with active device count (anchor=50),
        but DO NOT factor work_scale here. Hard-cap by available device slots.
        """
        devices = list(self.simulator.subnet.net.values())
        active = [d for d in devices if not d.Not_yet_added]

        # capacity by type (free = not busy and no workload)
        free_client_slots = sum(
            1 for d in active if (d.busy_time or 0) == 0 and d.workload is None and d.wtype != 'server'
        )
        free_server_slots = sum(
            1 for d in active if (d.busy_time or 0) == 0 and d.workload is None and d.wtype == 'server'
        )

        # simple linear scaling by active devices (anchor=50)
        anchor = 50
        scale = (len(active) / anchor) if anchor > 0 else 1.0

        req_clients = max(1, int(round(base_clients * scale)))
        req_servers = max(1, int(round(base_servers * scale)))

        # hard-cap by available free device slots (cannot exceed devices)
        clients = min(req_clients, max(1, free_client_slots))
        servers = min(req_servers, max(1, free_server_slots))

        return clients, servers

    def _safe_float(self, x, default=1.0):
        try:
            return float(x)
        except Exception:
            try:
                return float(str(x).strip().split()[0])
            except Exception:
                return default

    def _build_max_version_index(self):
        """
        Build an index of max version by app *type* across the whole subnet.
        Avoid lambdas so the env is picklable.
        """
        self._max_version_by_type = {}  # plain dict (picklable)
        for d in self.simulator.subnet.net.values():
            for a in d.apps.values():
                v = self._safe_float(a.version)
                if v is None:
                    continue
                t = a.type
                cur = self._max_version_by_type.get(t, 1.0)
                if v > cur:
                    self._max_version_by_type[t] = v

    def _bump_app_version(self, app):
        """
        Increment the version using the index (by app type) without scanning all devices.
        """
        t = app.type
        cur_max = self._max_version_by_type.get(t, 1.0)
        new_v = cur_max + 1.0
        app.version = f"{new_v:.1f}"
        self._max_version_by_type[t] = new_v

    def randomize_compromise_and_ownership(self):
        """
        Reassign 'attacker_owned' and 'isCompromised' for active, non-DC devices, keeping counts.
        DomainControllers untouched. Single pass + one shuffle; apply only changed bits.
        """
        sim = self.simulator

        # Prefer a cached list if your simulator can maintain it; else build once.
        # active_non_dcs = sim.active_non_dc_devices    # if available
        # Fallback:
        non_dcs = [d for d in sim.subnet.net.values()
                if not d.Not_yet_added and d.device_type != "DomainController"]
        n = len(non_dcs)
        if n == 0:
            return

        # current counts
        k_owned = 0
        k_comp  = 0
        for d in non_dcs:
            if d.attacker_owned: k_owned += 1
            if d.isCompromised:  k_comp  += 1

        if k_owned == 0 and k_comp == 0:
            # nothing to do
            return

        # one shuffle and slice — no set() and no extra list builds
        rnd = random
        rnd.shuffle(non_dcs)                      # in-place
        owned_slice = non_dcs[:k_owned]           # attacker-owned devices
        extra_comp  = max(0, k_comp - k_owned)
        comp_slice  = non_dcs[k_owned:k_owned+extra_comp]  # additional compromised

        # Apply only when value changes to minimize Python attr writes.
        # First, clear all three flags on all non-DCs
        for d in non_dcs:
            if d.attacker_owned:
                d.attacker_owned = False
            if d.isCompromised:
                d.isCompromised = False
            if d.Known_to_attacker:
                d.Known_to_attacker = False

        # Set new owned+compromised
        for d in owned_slice:
            if not d.attacker_owned:    d.attacker_owned = True
            if not d.isCompromised:     d.isCompromised = True
            if not d.Known_to_attacker: d.Known_to_attacker = True

        # Set extra compromised (not owned)
        for d in comp_slice:
            if not d.isCompromised:     d.isCompromised = True
            if not d.Known_to_attacker: d.Known_to_attacker = True


    # ---- slim device checkpointing (performance) ----
    def _pack_workload(self, w):
        if not w:
            return None
        return {
            "processing_time": int(getattr(w, "processing_time", 0) or 0),
            "adversarial": bool(getattr(w, "adversarial", False)),
            "wtype": getattr(w, "wtype", None),
        }

    def _has_indices(self, x):
        if x is None:
            return False
        if isinstance(x, np.ndarray):
            return x.size > 0
        try:
            return len(x) > 0
        except Exception:
            return False

    def _unpack_workload(self, s):
        if not s:
            return None
        w = Workload(
            id=-1,
            processing_time=int(s.get("processing_time", 0) or 0),
            OS=None,
            version=None,
        )
        setattr(w, "adversarial", bool(s.get("adversarial", False)))
        setattr(w, "wtype", s.get("wtype", None))
        return w

    def _device_state(self, d):
        return {
            "isCompromised": bool(d.isCompromised),
            "compromised_by": list(d.compromised_by),
            "busy_time": int(d.busy_time or 0),
            "workload": self._pack_workload(d.workload),
            "Known_to_attacker": bool(d.Known_to_attacker),
            "reachable_by_attacker": bool(d.reachable_by_attacker),
            "Not_yet_added": bool(d.Not_yet_added),
        }

    def _apply_device_state(self, d, s):
        d.isCompromised = s["isCompromised"]
        d.compromised_by = set(s["compromised_by"])
        d.busy_time = int(s["busy_time"])
        d.workload = self._unpack_workload(s["workload"])
        d.Known_to_attacker = s["Known_to_attacker"]
        d.reachable_by_attacker = s["reachable_by_attacker"]
        d.Not_yet_added = s["Not_yet_added"]

    def _checkpoint_device_mem(self, device_id: int) -> bool:
        try:
            dev = self.simulator.subnet.net[device_id]
            self._device_ckpts[device_id] = self._device_state(dev)
            return True
        except Exception:
            return False

    def _restore_device_mem(self, device_id: int) -> bool:
        state = self._device_ckpts.get(device_id)
        if state is None:
            return False
        dev = self.simulator.subnet.net[device_id]
        self._apply_device_state(dev, state)
        return True

    # ---- graph cache helpers ----
    def _rebuild_graph_cache(self):
        g = self.simulator.subnet.graph
        t0 = time.perf_counter()

        # Vertex maps
        self._vidx = {v["name"]: i for i, v in enumerate(g.vs)} if "name" in g.vs.attributes() else {i: i for i in range(g.vcount())}
        self._name = {i: (g.vs[i]["name"] if "name" in g.vs.attributes() else i) for i in range(g.vcount())}

        # Neighbor lists from igraph (C side); much faster than Python loops
        self._outnbrs = {}
        self._innbrs  = {}
        for i in range(g.vcount()):
            u = self._name[i]
            out_idx = g.neighbors(i, mode="out")
            in_idx  = g.neighbors(i, mode="in")
            # map indices to names
            self._outnbrs[u] = [self._name[j] for j in out_idx]
            self._innbrs[u]  = [self._name[j] for j in in_idx]

        # Fast path: don't force-create/scan "blocked" attribute over all edges on large graphs
        self._blocked = set()
        if "blocked" in g.es.attributes():
            # Only read true edges; DO NOT iterate all edges unless attribute exists
            # (still linear in E once, but avoids writing)
            # If this is still too slow for your E, comment this block entirely.
            pass

        #self._logp("graph cache built", t0)

    def _edge_is_blocked(self, u, v):
        return (u, v) in self._blocked

    def _set_edge_blocked(self, u, v, blocked=True):
        g = self.simulator.subnet.graph
        try:
            eid = g.get_eid(self._vidx[u], self._vidx[v], directed=True, error=True)
        except Exception:
            return False
        g.es[eid]["blocked"] = bool(blocked)
        if blocked:
            self._blocked.add((u, v))
        else:
            self._blocked.discard((u, v))
        return True

    def _random_incident_unblocked_edge(self, device_name):
        out_cands = [(device_name, v) for v in self._outnbrs.get(device_name, []) if not self._edge_is_blocked(device_name, v)]
        in_cands = [(u, device_name) for u in self._innbrs.get(device_name, []) if not self._edge_is_blocked(u, device_name)]
        pool = out_cands + in_cands
        return random.choice(pool) if pool else None

    def _random_incident_blocked_edge(self, device_name):
        out_cands = [(device_name, v) for v in self._outnbrs.get(device_name, []) if self._edge_is_blocked(device_name, v)]
        in_cands = [(u, device_name) for u in self._innbrs.get(device_name, []) if self._edge_is_blocked(u, device_name)]
        pool = out_cands + in_cands
        return random.choice(pool) if pool else None

    # ---- small helpers ----
    def get_num_action_types(self, mode=None):
        if mode == 'defender':
            return 14
        elif mode == 'attacker':
            return 3
        else:
            raise ValueError("Invalid mode: must be either 'defender' or 'attacker'")

    def get_device_indices(self):
        return [device.id for device in self.simulator.subnet.net.values()]

    def get_num_exploit_indices(self):
        return len(self.simulator.exploits)

    def get_num_app_indices(self):
        unique_apps = set()
        for device in self.simulator.subnet.net.values():
            for app in device.apps.values():
                unique_apps.add(app.id)
        return len(unique_apps)

    def _create_app_id_mapping(self):
        app_ids = set()
        for device in self.simulator.subnet.net.values():
            for app in device.apps.values():
                app_ids.add(app.id)
        app_id_mapping = {app_id: idx for idx, app_id in enumerate(app_ids)}
        return app_id_mapping

    def _encode_app_id(self, app_id):
        return self.app_id_mapping.get(app_id, -1)  # Return -1 if app_id is not found

    def _checkpoint_single_device(self, device_id):
        device = self.simulator.subnet.net[device_id]
        stamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        fname = f"device_{device_id}_{stamp}.pkl"
        outdir = Path("device_checkpoints")
        outdir.mkdir(parents=True, exist_ok=True)
        fpath = outdir / fname
        with open(fpath, "wb") as f:
            pickle.dump(copy.deepcopy(device), f)
        return str(fpath)

    def _latest_device_ckpt(self, device_id):
        folder = Path("device_checkpoints")
        pattern = folder.glob(f"device_{device_id}_*.pkl")
        latest = max(pattern, default=None, key=lambda p: p.stat().st_mtime)
        return latest

    def _count_comp(self):
        """Return (#compromised non-owned active, #compromised DC among active)."""
        n_comp = 0
        n_comp_dc = 0
        for d in self.simulator.subnet.net.values():
            if d.isCompromised and not d.Not_yet_added and not d.attacker_owned:
                n_comp += 1
                if getattr(d, "device_type", None) == "DomainController":
                    n_comp_dc += 1
        return n_comp, n_comp_dc

    # -------- arrivals (shared) --------
    def _generate_arrivals_if_due(self):
        """Shared arrival logic for step() and step_grouped()."""
        if (self.step_num % self._arrival_period()) != 0:
            return
        # Skip when few idle targets (avoids useless generation)
        if self._idle_fraction() < 0.10:
            return
        if self.scaling_vulnerability:
            nC, nS = self._scaled_numloads(100, 10)
        else:
            nC, nS = 100, 10

        # cap combined arrivals if requested
        if isinstance(self.workload_cap, int) and self.workload_cap > 0:
            total = nC + nS
            if total > self.workload_cap:
                ratio = self.workload_cap / total
                nC = max(0, int(nC * ratio))
                nS = max(0, int(nS * ratio))

        self._generate_workloads_timed(numLoads=nC, mode=2, high=5, wtype='client')
        self._generate_workloads_timed(numLoads=nS, mode=2, high=5, wtype='server')

    # ----------- step logic (unchanged besides _add_cost & arrivals wiring) -----------
    def _add_cost(self, delta: float, reward_accumulator: list | None = None):
        if self._in_batch:
            self._batch_cost += float(delta)
        else:
            if reward_accumulator is None:
                return
            reward_accumulator[0] += float(delta)

    def _tick_busy_time_once(self):
        for _, device in self.simulator.subnet.net.items():
            if device.busy_time > 0:
                device.busy_time -= 1

    def _step_apply_only(self, action):
        if action is None:
            pass

        action_type, exploit_indices, device_indices, app_index = action
        if self.mode == 'defender' and action_type == 0:
            action_type = 8
        elif self.mode == 'attacker' and action_type == 0:
            action_type = 3

        reward_box = [0.0]
        if self.mode == 'defender':
            if self.base_line != "Nash":
                action_type = 8

            if action_type == 2:
                self.checkpoint_count += 1
                self.checkpoint_variables(None, reward_box[0])
                self._add_cost(-0.5 * len(device_indices) * self.def_scale, reward_box)
                self.defensive_cost += 0.5 * len(device_indices) * self.def_scale
                for _, device in self.simulator.subnet.net.items():
                    if device.busy_time and device.busy_time > 0:
                        device.busy_time += 1

            elif action_type == 3:
                try:
                    self.revert_count += 1
                    if self.checkpoint is not None:
                        variables = self.checkpoint
                        self.simulator = variables['simulator']
                        for _, device in self.simulator.subnet.net.items():
                            device.busy_time = np.ceil(self._stall(0, self.default_high))
                            device.workload = None
                        self._add_cost(-1.0 * len(device_indices) * self.def_scale, reward_box)
                        self._topology_dirty = True
                except Exception:
                    pass

            elif action_type == 10:
                self._add_cost(-1.0 * self.def_scale, reward_box)
                data = self.simulator.logger.get_logs()
                if data:
                    if isinstance(data, list):
                        data = data[-2000:] if len(data) > 2000 else data
                    else:
                        dl = list(data)
                        data = dl[-2000:] if len(dl) > 2000 else dl
                self._train_detector(data)

            elif action_type == 11:
                if device_indices is None or (hasattr(device_indices, "__len__") and len(device_indices) == 0):
                    raise ValueError("Action 11 requires exactly one device index")
                target_id = int(device_indices[0])
                _ = self._checkpoint_device_mem(target_id)
                self.checkpoint_count += 1
                self._add_cost(-0.1 * self.def_scale, reward_box)
                self.defensive_cost += 0.1 * self.def_scale

        for device_index in device_indices:
            device = self.simulator.subnet.net[device_index]
            if device.Not_yet_added:
                continue

            if self.mode == 'defender':
                if action_type == 1:
                    if not device.attacker_owned:
                        self._add_cost((0.3 if device.isCompromised else -0.01) * self.def_scale, reward_box)
                        self.clearning_cost += (0.3 if device.isCompromised else 0.01) * self.def_scale
                        self.defensive_cost += (0.3 if device.isCompromised else 0.01) * self.def_scale
                        if not self._exp_by_id:
                            self._reindex_exploits()
                        for eid in list(device.compromised_by):
                            exp = self._exp_by_id.get(eid)
                            if exp is not None:
                                exp.discovered = True
                        device.compromised_by.clear()
                        device.isCompromised = False
                        device.busy_time = np.ceil(self._stall(0, self.default_high))
                        device.workload = None
            else:
                pass

    def step_grouped(self, groups):
        assert isinstance(groups, (list, tuple)) and len(groups) > 0
        prev_batch = self._in_batch
        self._in_batch = True
        self._batch_cost = 0.0
        for g in groups:
            self._step_apply_only(g)
        self._in_batch = prev_batch

        self._tick_busy_time_once()

        current_work = 0
        att_work = 0
        for _, device_work in self.simulator.subnet.net.items():
            if device_work.busy_time == 0 and not device_work.Not_yet_added:
                if (device_work.workload is not None
                    and device_work.workload.processing_time
                    and device_work.workload.processing_time > 0
                    and not device_work.workload.adversarial):
                    device_work.workload.processing_time -= 1
                    if device_work.workload.processing_time == 0:
                        device_work.workload = None
                        self.work_done += 1
                        current_work += 1
                if (device_work.workload is not None
                    and device_work.workload.processing_time
                    and device_work.workload.processing_time > 0
                    and device_work.workload.adversarial):
                    device_work.workload.processing_time -= 1
                    if device_work.workload.processing_time == 0:
                        device_work.workload = None
                        att_work += 1

        def_work = self.work_scale * current_work

        # >>> NEW/CHANGED: adaptive arrivals (single path)
        self._generate_arrivals_if_due()

        n_comp, n_comp_dc = self._count_comp()
        shaping_bonus = 0.0
        if self.mode == 'defender':
            raw_reward = self._batch_cost + def_work - n_comp * self.comp_scale
            shaped_reward = raw_reward
        else:
            base = self._batch_cost + self.comp_scale * (n_comp + 10 * n_comp_dc)
            max_nodes = len(self.simulator.subnet.net)
            φ_a_new = (n_comp / max_nodes) if max_nodes > 0 else 0.0
            if not hasattr(self, '_prev_att_potential') or self._prev_att_potential is None:
                self._prev_att_potential = φ_a_new
            α_att = 0.1
            shaped_inc = self.γ * φ_a_new - self._prev_att_potential
            shaping_bonus = α_att * shaped_inc + shaping_bonus
            self._prev_att_potential = self.γ * φ_a_new
            raw_reward = base
            shaped_reward = raw_reward + shaping_bonus

        self.state = self._get_state()
        self.step_num += 1
        if self.mode == "attacker":
            self.attacker_step += 1
        else:
            self.defender_step += 1

        done = self._check_done(groups[-1])

        info = {
            'mode': self.mode,
            'step_count': self.step_num,
            'revert_count': self.revert_count,
            'checkpoint_count': self.checkpoint_count,
            'defensive_cost': self.defensive_cost,
            'clearning_cost': self.clearning_cost,
            'Scan_count': self.scan_cnt,
            'action_taken': groups,
            'work_done': self.work_done,
            'Compromised_devices': self.compromised_devices_cnt,
            'Edges Blocked': self.edges_blocked,
            'Edges Added': self.edges_added,
        }

        need_periodic = (self.step_num % self._evolve_period) == 0
        if self._topology_dirty or need_periodic:
            self.evolve_network()
            self._topology_dirty = False

        return self.state, raw_reward, shaped_reward, done, info, self.simulator.logger.get_logs()



    def _work_ready_rebuild(self):
        """Rebuild the ready set (devices that can advance work now)."""
        ready = set()
        for d in self.simulator.subnet.net.values():
            if d.Not_yet_added or d.busy_time != 0:
                continue
            wl = d.workload
            if wl and getattr(wl, "processing_time", 0) > 0:
                ready.add(d)
        self._work_ready = ready
        self._work_ready_epoch = getattr(self, "_work_ready_epoch", 0) + 1

    def _work_ready_maybe_rebuild(self, interval=50):
        """Periodic reconciliation to stay correct if other code mutates devices."""
        if not hasattr(self, "_work_ready") or not hasattr(self, "_work_ready_epoch"):
            self._work_ready_rebuild()
            return
        step = getattr(self, "step_num", 0)
        if interval and step % interval == 0:
            self._work_ready_rebuild()

    def _work_ready_maybe_add(self, device):
        """Call when you assign a workload or a device becomes idle."""
        if device.Not_yet_added or device.busy_time != 0:
            return
        wl = device.workload
        if wl and getattr(wl, "processing_time", 0) > 0:
            # lazy init if needed
            if not hasattr(self, "_work_ready"):
                self._work_ready = set()
                self._work_ready_epoch = 0
            self._work_ready.add(device)



    def step(self, action, agent_cnt=None):


        deadline = getattr(self, "time_budget_deadline", None)


        '''
        if deadline is not None and time.time() > deadline and not getattr(self, "time_budget_exceeded", False):
            print("Time exceeded, current time: "+str(time.time())+" current deadline: "+str(deadline) )
            self.time_budget_exceeded = True
            self.step_num = int(1e9)
            info = {"time_budget_exceeded": True}
            if not getattr(self, "_time_budget_warned", False):
                now = time.time()
                print(f"[TIME_BUDGET] now={now:.0f} configured={deadline:.0f} "
                    f"remaining={(deadline-now)/3600:.2f}h")
                self._time_budget_warned = True
            return self.state, 0.0, None, True, info, None
        '''
                    



        
        if isinstance(action, (list, tuple)) and action and isinstance(action[0], (list, tuple)):
            #print("check point grouped")
            return self.step_grouped(action)

        # ----- fill default action if None -----
        if action is None:
            if self.mode == 'defender':
                if self.base_line == "No Defense":
                    action_type = 8
                    exploit_indices = [0]
                    device_indices = [
                        d.id
                        for d in self.simulator.subnet.net.values()
                        if not d.attacker_owned and not d.Not_yet_added
                    ]
                    app_index = 0
                elif self.base_line == "Preset":
                    action_type, exploit_indices, device_indices, app_index = 7, [0], [], 0
                else:
                    action_type, exploit_indices, device_indices, app_index = 7, [0], [], 0
            elif self.mode == 'attacker':
                if self.base_line == "No Attack":
                    action_type = 3
                    exploit_indices = [0]
                    device_indices = [
                        d.id
                        for d in self.simulator.subnet.net.values()
                        if d.Known_to_attacker and not d.Not_yet_added
                    ]
                    app_index = 0
                else:
                    action_type, exploit_indices, device_indices, app_index = 2, [0], [], 0
            action = (action_type, exploit_indices, device_indices, app_index)

        action_type, exploit_indices, device_indices, app_index = action

        # ----- validate action type against the correct action space -----
        if self.mode == 'defender':
            if not (0 <= action_type < self.defender_action_space.n):
                action_type = 8  # noop
        elif self.mode == 'attacker':
            if not (0 <= action_type < self.attacker_action_space.n):
                action_type = 3  # noop

        for device_id, device in self.simulator.subnet.net.items():
            if self.debug:
                logging.debug(
                    f"Before action: Device {device_id} - isCompromised: {device.isCompromised}, "
                    f"attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, "
                    f"mode: {self.mode}, always reachable: {device.reachable_by_attacker}, "
                    f"To be Added: {device.Not_yet_added}"
                )
            pass

        reward = 0
        reward_box = [0.0]
        done = False
        def_work = 0
        att_work = 0
        shaping_bonus = 0

        #print("check point starting busy time decremation")
        for d in tuple(self._busy_devices):
            d.busy_time -= 1
            if d.busy_time <= 0:
                d.busy_time = 0
                self._busy_devices.discard(d)
        #print("check point starting busy time decremation finished")

        # ----- mode-specific meta actions (these don't need per-device iteration) -----
        if self.mode == 'defender':
            if self.base_line != "Nash":
                action_type = 8
            if self.debug:
                logging.debug(f"Defender action taken: {action_type} on device {device_indices} at step {self.step_num}")

            if action_type == 2:
                self.checkpoint_count += 1
                self.checkpoint_variables(None, reward)
                self._add_cost(-0.5 * len(device_indices) * self.def_scale, reward_box)
                self.defensive_cost += 0.5 * len(device_indices) * self.def_scale
                #print("check point busy time 2 decrement")
                for _, device in self.simulator.subnet.net.items():
                    if device.busy_time is not None and device.busy_time > 0:
                        device.busy_time += 1

            elif action_type == 3:
                try:
                    self.revert_count += 1
                    if self.checkpoint is not None:
                        variables = self.checkpoint
                        self.simulator = variables['simulator']
                        for _, device in self.simulator.subnet.net.items():
                            #print("checkpoint busy time 3")
                            device.busy_time = np.ceil(self._stall(0, self.default_high))
                            device.workload = None
                        self._add_cost(-1.0 * len(device_indices) * self.def_scale, reward_box)
                        self._topology_dirty = True
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] Revert failed: {e}")
                    pass

            elif action_type == 10:
                if self._has_indices(device_indices):
                    didx = device_indices[0]
                    d = self.simulator.subnet.net[int(didx)]
                    d.busy_time = (d.busy_time or 0) + 1
                else:
                    for d in self.simulator.subnet.net.values():
                        if getattr(d, "busy_time", 0) > 0:
                            d.busy_time += 1
                self._add_cost(-1.0 * self.def_scale, reward_box)
                data = self.simulator.logger.get_logs()
                if data:
                    if isinstance(data, list):
                        data = data[-2000:] if len(data) > 2000 else data
                    else:
                        dl = list(data)
                        data = dl[-2000:] if len(dl) > 2000 else dl
                self._train_detector(data)

            elif action_type == 11:
                if not self._has_indices(device_indices):
                    raise ValueError("Action 11 requires exactly one device index")
                target_id = int(device_indices[0])
                saved_mem = self._checkpoint_device_mem(target_id)
                self.checkpoint_count += 1
                self._add_cost(-0.1 * self.def_scale, reward_box)
                self.defensive_cost += 0.1 * self.def_scale
                if self.debug:
                    if saved_mem:
                        logging.debug(f"[Checkpoint-Single] Saved device {target_id} to in-memory stash")
                    else:
                        logging.debug(f"[Checkpoint-Single] FAILED to save device {target_id} to memory")
        else:
            if self.debug:
                logging.debug(f"Attacker action taken: {action_type} on device {device_indices} at step {self.step_num}")

        # ----- decide if we need to iterate over device_indices at all -----
        # Per-device defender actions only; attacker actions never need it here.
        must_iterate_device_indices = (
            self.mode == 'defender'
            and action_type in (1, 4, 5, 6, 7, 9, 12, 13)
        )

        # ----- per-device actions (only when needed) -----
        if must_iterate_device_indices:
            for device_index in device_indices:
                device = self.simulator.subnet.net[device_index]
                if device.Not_yet_added:
                    continue

                if self.mode == 'defender':
                    if action_type == 1:
                        if not device.attacker_owned:
                            self._add_cost((0.3 if device.isCompromised else -0.01) * self.def_scale, reward_box)
                            self.clearning_cost += (0.3 if device.isCompromised else 0.01) * self.def_scale
                            self.defensive_cost += (0.3 if device.isCompromised else 0.01) * self.def_scale
                            if not self._exp_by_id:
                                self._reindex_exploits()
                            for eid in list(device.compromised_by):
                                exp = self._exp_by_id.get(eid)
                                if exp is not None:
                                    exp.discovered = True
                            device.compromised_by.clear()
                            device.isCompromised = False
                            device.busy_time = np.ceil(self._stall(0, self.default_high))
                            if device.workload:
                                device.workload = None

                    elif action_type == 4:
                        self._add_cost(-1 * self.def_scale, reward_box)
                        if isinstance(app_index, int) and 0 <= app_index < len(device.apps):
                            app = list(device.apps.values())[app_index]
                            self._bump_app_version(app)
                            device.busy_time = np.ceil(self._stall(0, self.default_high))

                    elif action_type == 5:
                        self.scan_cnt += 1
                        logs = self.simulator.logger.get_logs()
                        if logs:
                            if isinstance(logs, list):
                                logs = logs[-512:] if len(logs) > 512 else logs
                            else:
                                ll = list(logs)
                                logs = ll[-512:] if len(ll) > 512 else ll

                        if not self.fast_scan:
                            for log in (logs[-256:] if logs else []):
                                if not self.turbo:
                                    predicted_kind, device.anomaly_score = self.simulator.detector.predict(
                                        from_device=log["from_device"], to_device=log["to_device"], return_score=True
                                    )
                                else:
                                    predicted_kind = None
                                    device.anomaly_score = 0.0
                                self._add_cost(-0.5 * self.def_scale, reward_box)
                                self.defensive_cost += 0.5 * self.def_scale
                                if predicted_kind == "A":
                                    dsrc = self.simulator.subnet.net[log["from_device"]]
                                    for eid in dsrc.compromised_by:
                                        for exp in self.simulator.exploits:
                                            if exp.id == eid:
                                                exp.discovered = True
                                                break
                                    dsrc.compromised_by.clear()
                                    dsrc.isCompromised = False
                                    dsrc.busy_time = np.ceil(self._stall(0, self.default_high))
                        else:
                            window = logs[-30:] if logs else []
                            if window:
                                log_points = [(l["from_device"], l["to_device"]) for l in window]
                                predictions = [] if self.turbo else self.simulator.detector.batch_predict(log_points)
                                num_anoms = sum(1 for p in predictions if p == "A")
                                majority = len(predictions) // 2 + 1
                                self._add_cost(-0.5 * self.def_scale, reward_box)
                                self.defensive_cost += 0.5 * self.def_scale
                                if num_anoms >= majority:
                                    flagged_senders = {
                                        l["from_device"]
                                        for l, p in zip(window, predictions)
                                        if p == "A"
                                    }
                                    for dev_id in flagged_senders:
                                        dev = self.simulator.subnet.net[dev_id]
                                        dev.isCompromised = False
                                        dev.busy_time = np.ceil(self._stall(0, self.default_high))

                    elif action_type == 6:
                        self._add_cost(-0.5 * self.def_scale, reward_box)
                        self.defensive_cost += 0.5 * self.def_scale
                        u = device.id
                        cand = self._random_incident_unblocked_edge(u)
                        if cand:
                            su, sv = cand
                            if self._set_edge_blocked(su, sv, True):
                                self.edges_blocked += 1
                                self._topology_dirty = True

                    elif action_type == 7:
                        self._add_cost(-0.5 * self.def_scale, reward_box)
                        device.Not_yet_added = True
                        device.isCompromised = False
                        device.compromised_by.clear()
                        if device.workload:
                            device.workload = None
                        self._topology_dirty = True

                    elif action_type == 9:
                        self._add_cost(-0.5 * self.def_scale, reward_box)
                        self.defensive_cost += 0.5 * self.def_scale
                        u = device.id
                        cand = self._random_incident_blocked_edge(u)
                        if cand:
                            su, sv = cand
                            if self._set_edge_blocked(su, sv, False):
                                self.edges_added += 1
                                self._topology_dirty = True

                    elif action_type == 12:
                        if not self._has_indices(device_indices):
                            raise ValueError("Action 12 needs one device index")
                        dev_id = int(device_indices[0])
                        restored = self._restore_device_mem(dev_id)
                        if restored:
                            self._add_cost(-1 * self.def_scale, reward_box)
                            self.defensive_cost += 1.0 * self.def_scale

                    elif action_type == 13:
                        if device_indices is None or len(device_indices) == 0:
                            raise ValueError("Action 13 needs one device index")
                        dev_id = device_indices[0]
                        dev = self.simulator.subnet.net[dev_id]
                        dev.isCompromised = False
                        dev.compromised_by.clear()
                        if dev.workload:
                            dev.workload = None
                        dev.busy_time = np.ceil(self._stall(3, self.default_high + 3))
                        self._add_cost(-3.0 * self.def_scale, reward_box)
                        self.clearning_cost += 3.0 * self.def_scale
                        self.defensive_cost += 3.0 * self.def_scale

        # ----- attacker actions that don’t require per-device iteration -----
        if self.mode == 'attacker':
            compromised_devices = [idx for idx, device in self.simulator.subnet.net.items()
                                if device.isCompromised or device.attacker_owned]

            if action_type == 1 and self.base_line != "No Attack":
                if self.zero_day:
                    owned_indices = list(self.common_exploit_indices | self.private_exploit_indices)

                for raw in exploit_indices:
                    if self.zero_day and raw not in owned_indices:
                        raw = random.choice(owned_indices)

                    if 0 <= raw < len(self.simulator.exploits):
                        exploit = self.simulator.exploits[raw]
                    else:
                        exploit = next((e for e in self.simulator.exploits if e.id == raw), None)

                    if exploit is None:
                        continue
                    if self.zero_day and raw not in owned_indices:
                        continue

                    for device_id in compromised_devices:
                        device = self.simulator.subnet.net[device_id]
                        neighbor_ids = self._outnbrs.get(device_id, [])
                        if not neighbor_ids:
                            continue

                        for neighbor_id in neighbor_ids:
                            if self._edge_is_blocked(device_id, neighbor_id):
                                if self.debug:
                                    logging.debug(f"Edge from {device_id} to {neighbor_id} is blocked. Skipping communication.")
                                continue

                            neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                            self.simulator.log_communication(device_id, neighbor_id, 'A')

                            if device.device_type == "DomainController":
                                neighbor_device.isCompromised = True
                                neighbor_device.compromised_by.add(exploit.id)
                                if self.debug:
                                    logging.debug(f"Device {neighbor_id} compromised by Domain Controller {device_id}.")
                                break

                            if neighbor_device.reachable_by_attacker:
                                neighbor_device.isCompromised = True
                                break

                            if neighbor_device and not neighbor_device.isCompromised and neighbor_device.Known_to_attacker:
                                for app in neighbor_device.apps.values():
                                    for vul in app.vulnerabilities.values():
                                        if exploit is not None and vul.id in exploit.target:
                                            if self.debug:
                                                logging.debug(
                                                    f"Compromising device {neighbor_id} via vuln {vul.id} using exploit {exploit.id}"
                                                )
                                            neighbor_device.isCompromised = True
                                            break
                                if neighbor_device.isCompromised:
                                    break

            if action_type == 2 and self.base_line != "No Attack":
                if compromised_devices:
                    probe_from_device_id = random.choice(compromised_devices)
                    neighbor_ids = self._outnbrs.get(probe_from_device_id, [])
                    for neighbor_id in neighbor_ids:
                        if self._edge_is_blocked(probe_from_device_id, neighbor_id):
                            if self.debug:
                                logging.debug(f"Edge from {probe_from_device_id} to {neighbor_id} is blocked. Skipping probe.")
                            continue
                        neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                        if neighbor_device and not neighbor_device.Known_to_attacker:
                            neighbor_device.Known_to_attacker = True
                            self._add_cost(+0.1, reward_box)
                            if self.debug:
                                logging.debug(f"Probed device {neighbor_id} from {probe_from_device_id}.")
                            break

            # action_type == 3 or "No Attack" is pass

        # ----- workload advancement + arrivals -----
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt:
            current_work = 0

            if self.numOfDevice > 5000:
                self._work_ready_maybe_rebuild(interval=5)

                current_work = 0
                att_work = 0

                for d in tuple(self._work_ready):
                    if d.Not_yet_added or d.busy_time != 0 or not d.workload:
                        self._work_ready.discard(d)
                        continue

                    wl = d.workload
                    pt = getattr(wl, "processing_time", 0)
                    if pt <= 0:
                        self._work_ready.discard(d)
                        continue

                    wl.processing_time = pt - 1
                    if wl.processing_time > 0:
                        continue

                    adv = getattr(wl, "adversarial", False)
                    d.workload = None
                    self._work_ready.discard(d)

                    if adv:
                        att_work += 1
                    else:
                        self.work_done += 1
                        current_work += 1

            else:
                for _, device_work in self.simulator.subnet.net.items():
                    if device_work.busy_time == 0 and not device_work.Not_yet_added:
                        if (device_work.workload is not None and
                            device_work.workload.processing_time is not None and
                            device_work.workload.processing_time > 0 and
                            not device_work.workload.adversarial):
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0:
                                device_work.workload = None
                                self.work_done += 1
                                current_work += 1

                        if (device_work.workload is not None and
                            device_work.workload.processing_time is not None and
                            device_work.workload.processing_time > 0 and
                            device_work.workload.adversarial):
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0:
                                device_work.workload = None
                                att_work += 1

            def_work = self.work_scale * current_work
            self._generate_arrivals_if_due()

        # ----- reward & bookkeeping -----
        self.compromised_devices_cnt += len([
            d for d in self.simulator.subnet.net.values()
            if d.isCompromised and not d.attacker_owned and not d.Not_yet_added
        ])

        info = {
            'mode': self.mode,
            'step_count': self.step_num,
            'revert_count': self.revert_count,
            'checkpoint_count': self.checkpoint_count,
            'defensive_cost': self.defensive_cost,
            'clearning_cost': self.clearning_cost,
            'Scan_count': self.scan_cnt,
            'action_taken': action,
            'work_done': self.work_done,
            'Compromised_devices': self.compromised_devices_cnt,
            'Edges Blocked': self.edges_blocked,
            'Edges Added': self.edges_added,
        }

        reward += reward_box[0]
        n_comp, n_comp_dc = self._count_comp()

        if self.mode == 'defender':
            raw_reward = reward + def_work - n_comp * self.comp_scale
            shaped_reward = raw_reward
        else:
            reward = reward + self.comp_scale * (n_comp + 10 * n_comp_dc)
            max_nodes = len(self.simulator.subnet.net)
            φ_a_new = (n_comp / max_nodes) if max_nodes > 0 else 0.0
            if not hasattr(self, '_prev_att_potential') or self._prev_att_potential is None:
                self._prev_att_potential = φ_a_new
            α_att = 0.1
            shaped_inc = self.γ * φ_a_new - self._prev_att_potential
            shaping_bonus = α_att * shaped_inc + shaping_bonus
            self._prev_att_potential = self.γ * φ_a_new
            raw_reward = reward
            shaped_reward = raw_reward + shaping_bonus

        self.state = self._get_state()
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt:
            self.step_num += 1
            if self.mode == "attacker":
                self.attacker_step += 1
            else:
                self.defender_step += 1

        done = self._check_done(action)

        for device_id, device in self.simulator.subnet.net.items():
            if self.debug:
                print("checkpoint logging")
                logging.debug(
                    f"After action: Device {device_id} - isCompromised: {device.isCompromised}, "
                    f"attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, "
                    f"done: {done}, reward for step: {reward}, To be Added: {device.Not_yet_added}"
                )
            pass

        need_periodic = (self.step_num % self._evolve_period) == 0
        if self._topology_dirty or need_periodic:
            self.evolve_network()
            self._topology_dirty = False
        self._busy_devices = {d for d in self.simulator.subnet.net.values() if d.busy_time > 0}

        info['executed_atype'] = action_type
        return self.state, raw_reward, shaped_reward, done, info, self.simulator.logger.get_logs()

    def _logp(self, msg, t0=None):
        """Lightweight progress print with optional delta timing."""
        if t0 is not None:
            dt = time.perf_counter() - t0
            print(f"[init] {msg} ({dt:.2f}s)", flush=True)
        else:
            print(f"[init] {msg}", flush=True)

    # ---- new helper: sparse connect attacker-owned devices ----
    def _sparse_connect_attacker_owned(self, g, attacker_owned_devices, sample_k=3, verbose=False):
        """
        Bulk-add a small set of edges for each attacker-owned device:
          - ensure each attacker-owned node has an edge to every DomainController (DC)
          - add `sample_k` extra random edges (non-DC, non-self) to avoid isolation
        Uses bulk g.add_edges(list_of_pairs) where possible to minimize igraph call overhead.
        Accepts attacker_owned_devices as iterable of device ids (ints or strings).
        """
        t0 = time.perf_counter()
        # Ensure caches exist
        try:
            self._rebuild_graph_cache()
        except Exception:
            # best-effort rebuild using graph vs
            try:
                self._vidx = {v["name"]: i for i, v in enumerate(g.vs)}
                self._name = {i: v["name"] for i, v in enumerate(g.vs)}
            except Exception:
                # give up gracefully
                if verbose:
                    print("[_sparse_connect] failed to build vidx/name caches")
                return

        # Collect DC names based on device_type on subnet
        dc_names = [str(d.id) if not isinstance(d.id, str) else d.id
                    for d in self.simulator.subnet.net.values()
                    if getattr(d, "device_type", None) == "DomainController"]
        if verbose:
            print(f"[_sparse_connect] attacker_owned={len(attacker_owned_devices)} DCs={len(dc_names)} sample_k={sample_k}")

        all_names = list(self._vidx.keys())
        name_set = set(all_names)

        # normalize attacker names into graph naming format (string or raw id)
        att_names = []
        for aid in attacker_owned_devices:
            an = aid if isinstance(aid, str) else str(aid)
            if an in name_set:
                att_names.append(an)
            else:
                # try raw id if graph stored ints as names
                if aid in name_set:
                    att_names.append(aid)

        edges_to_add = []
        sampled = 0
        # Precompute non-DC candidates
        non_dc_candidates = [n for n in all_names if n not in dc_names]

        for an in att_names:
            try:
                aidx = self._vidx[an]
            except KeyError:
                continue

            # Connect to every DC if not already connected
            for dcn in dc_names:
                try:
                    didx = self._vidx[dcn]
                except KeyError:
                    continue
                try:
                    eid = g.get_eid(aidx, didx, directed=True, error=False)
                    if eid == -1:
                        edges_to_add.append((aidx, didx))
                except Exception:
                    edges_to_add.append((aidx, didx))

            # Add a few random edges to non-DCs
            if non_dc_candidates:
                k = min(len(non_dc_candidates), max(0, int(sample_k)))
                if k > 0:
                    picks = random.sample(non_dc_candidates, k)
                    for p in picks:
                        try:
                            pidx = self._vidx[p]
                        except KeyError:
                            continue
                        try:
                            eid = g.get_eid(aidx, pidx, directed=True, error=False)
                            if eid == -1:
                                edges_to_add.append((aidx, pidx))
                                sampled += 1
                        except Exception:
                            edges_to_add.append((aidx, pidx))
                            sampled += 1

        # Bulk add unique edges
        if edges_to_add:
            try:
                # deduplicate
                edges_to_add = list(set(edges_to_add))
                g.add_edges(edges_to_add)
            except Exception as e:
                # fallback: add one-by-one (slower but robust)
                if verbose:
                    print(f"[_sparse_connect] bulk add failed ({e}), falling back to per-edge add")
                for u, v in edges_to_add:
                    try:
                        g.add_edges([(u, v)])
                    except Exception:
                        pass

        # ensure 'blocked' edge attribute exists to keep invariants
        try:
            if "blocked" not in g.es.attributes():
                g.es["blocked"] = [False] * g.ecount()
        except Exception:
            pass

        # rebuild caches cheaply
        try:
            self._rebuild_graph_cache()
        except Exception:
            pass

        if verbose:
            dt = time.perf_counter() - t0
            print(f"[_sparse_connect] done; added ~{len(edges_to_add)} edges (sampled {sampled}) in {dt:.2f}s")

    # ---- tiny shim: choose dense vs sparse based on size ----
    def _connect_attacker_owned_smart(self, g, attacker_owned_devices):
        """
        Use dense simulator helper for small graphs; switch to sparse for large ones.
        Mirrors logic already used in initialize_environment().
        """
        net_size = len(self.simulator.subnet.net)
        if net_size >= getattr(self, "_sparse_threshold", 500):
            sample_k = max(1, int(round(np.log2(max(2, net_size)) / 2)))
            print(f"[init] Large net ({net_size} nodes): using sparse attacker-connect (sample_k={sample_k})")
            try:
                self._sparse_connect_attacker_owned(g, attacker_owned_devices, sample_k=sample_k, verbose=True)
            except Exception as e:
                print(f"[init] sparse connect failed: {e} - marking attacker-owned flags without edges")
            return
        # small graphs → try dense helper, fall back to sparse(3)
        try:
            self.simulator.subnet.connectAttackerOwnedDevices(g, attacker_owned_devices)
        except Exception:
            self._sparse_connect_attacker_owned(g, attacker_owned_devices, sample_k=3, verbose=True)

    def initialize_environment(self):
        """
        Initialize environment with progress prints & timing. Uses sparse attacker-connect
        for very large graphs to drastically reduce initialization time.
        """
        self._ensure_perf()
        t0_total = time.perf_counter()
        print("[init] starting environment initialization")

        # 1) reset simulator subnet
        t0 = time.perf_counter()
        self.simulator.resetAllSubnet()
        print(f"[init] resetAllSubnet: {time.perf_counter() - t0:.3f}s")

        # 2) generate target apps
        t0 = time.perf_counter()
        targetApps = self.simulator.generateApps(3, True, 1, vul_to=VOLT_CVE_ID)
        print(f"[init] generateApps: {time.perf_counter() - t0:.3f}s")

        # ZERO-DAY bookkeeping (fast ops)
        self.common_exploit_ids = []
        self.unknown_pool_ids = []
        self.private_exploit_ids = []
        if not isinstance(self.k_known, int) or self.k_known < 0:
            self.k_known = 1
        if not isinstance(self.j_private, int) or self.j_private < 0:
            self.j_private = 0

        minVulperExp = 1
        maxVulperExp = 1

        # 3) exploits (clip to small number typically)
        t0 = time.perf_counter()
        if self.zero_day:
            total_to_generate = (self.k_known + self.j_private)
            if total_to_generate > 0:
                self.simulator.generateExploits(
                    total_to_generate,
                    addVul=True,
                    minVulperExp=minVulperExp,
                    maxVulperExp=maxVulperExp,
                    mode="random"
                )

            # map ids efficiently
            sim_exs = getattr(self.simulator, "exploits", [])
            for idx in range(min(self.k_known, len(sim_exs))):
                self.common_exploit_ids.append(sim_exs[idx].id)
            self.common_exploit_indices = set(range(min(self.k_known, len(sim_exs))))

            start_idx = self.k_known
            end_idx = min(self.k_known + self.j_private, len(sim_exs))
            for idx in range(start_idx, end_idx):
                self.unknown_pool_ids.append(sim_exs[idx].id)

            if self.unknown_pool_ids:
                self.prior_pi = {z: 1.0 / len(self.unknown_pool_ids) for z in self.unknown_pool_ids}
            else:
                self.prior_pi = {}

            pool = list(self.unknown_pool_ids)
            if pool and self.j_private > 0:
                probs = [self.prior_pi.get(z, 0.0) for z in pool]
                total = sum(probs)
                if total <= 0.0:
                    probs = [1.0 / len(pool)] * len(pool)
                else:
                    probs = [p / total for p in probs]
                chosen = self._rng.choice(pool, size=min(self.j_private, len(pool)), replace=False, p=probs)
                try:
                    self.private_exploit_ids = list(chosen)
                except TypeError:
                    self.private_exploit_ids = [chosen]
            else:
                self.private_exploit_ids = []
            self.private_exploit_indices = {
                idx for idx, exp in enumerate(getattr(self.simulator, "exploits", []))
                if exp.id in set(self.private_exploit_ids)
            }
        else:
            self.simulator.generateExploits(
                1, addVul=True, minVulperExp=minVulperExp, maxVulperExp=maxVulperExp,
                mode="target", expID=VOLT_CVE_ID
            )
            self.simulator.generateExploits(
                1, addVul=True, minVulperExp=minVulperExp, maxVulperExp=maxVulperExp,
                mode="target", expID=VOLT_DC_CVE_ID
            )
            # fill common_exploit_ids quickly
            for exp_obj in getattr(self.simulator, "exploits", []):
                self.common_exploit_ids.append(exp_obj.id)
            self.unknown_pool_ids = []
            self.private_exploit_ids = []
        print(f"[init] exploit generation: {time.perf_counter() - t0:.3f}s")

        # 4) scaling knobs -> compute n_dc, n_owned, addApps
        t0 = time.perf_counter()
        if self.scaling_vulnerability:
            n_dc = max(1, int(np.ceil(self.numOfDevice / max(1, float(getattr(self, "sv_dc_ratio", 1))))))
            n_owned = max(1, int(round(self.numOfDevice * float(getattr(self, "sv_attacker_fraction", 0.05)))))
            addApps = int(getattr(self, "sv_apps_base", 3) + np.floor(self.numOfDevice * float(getattr(self, "sv_apps_per_device", 0.0))))
            addApps = max(1, addApps)
        else:
            n_dc = 3
            n_owned = 5
            addApps = 3
        print(f"[init] scaling knobs computed: {time.perf_counter() - t0:.3f}s (n_dc={n_dc}, n_owned={n_owned}, addApps={addApps})")

        # 5) generate subnet (this can be heavy)
        t0 = time.perf_counter()
        maxVulperApp = 1
        self.simulator.generateSubnet(self.Max_network_size, addApps, 0, maxVulperApp + 1)
        t_subnet = time.perf_counter() - t0
        print(f"[init] generateSubnet: {t_subnet:.3f}s; nodes={len(self.simulator.subnet.net)}")

        # 6) graph initialization
        t0 = time.perf_counter()
        g = self.simulator.subnet.initializeVoltTyGraph(self.Max_network_size)
        t_graph = time.perf_counter() - t0
        print(f"[init] initializeVoltTyGraph: {t_graph:.3f}s")

        # 7) optional partitions & rebuild caches
        t0 = time.perf_counter()
        if hasattr(self.simulator.subnet, "create_partitions") and getattr(self, "partition_size", None) is not None:
            try:
                self.simulator.subnet.create_partitions(self.partition_size)
            except Exception:
                pass
        # rebuild graph cache (fast)
        try:
            self._rebuild_graph_cache()
        except Exception:
            pass
        print(f"[init] partitions + rebuild_graph_cache: {time.perf_counter() - t0:.3f}s")

        # 8) assign target apps to devices (vectorized style, single loop)
        t0 = time.perf_counter()
        net_vals = list(self.simulator.subnet.net.values())
        for device in net_vals:
            device.addApps(targetApps)
        print(f"[init] assign targetApps to devices: {time.perf_counter() - t0:.3f}s")

        # 9) choose most-connected devices and set active set
        t0 = time.perf_counter()
        all_devices = sorted(net_vals, key=lambda d: d.id)
        # neighbors lookup via igraph is expensive for each node; use cached mapping if possible
        try:
            # use degree from graph if available
            degrees = {int(self._name[i]): len(g.neighbors(i)) for i in range(len(g.vs))}
            most_connected_devices = sorted(all_devices, key=lambda d: degrees.get(d.id, 0), reverse=True)[:max(3, n_dc)]
        except Exception:
            most_connected_devices = sorted(all_devices, key=lambda d: len(g.neighbors(d.id)), reverse=True)[:max(3, n_dc)]

        forced_active_ids = {dev.id for dev in most_connected_devices}
        starting_compromised = self.starting_compromised or []
        forced_active_ids = forced_active_ids.union(set(starting_compromised))
        initial_active_ids = {dev.id for dev in all_devices[: self.numOfDevice]}
        active_set_ids = initial_active_ids.union(forced_active_ids)
        for device in all_devices:
            device.Not_yet_added = device.id not in active_set_ids
        print(f"[init] active-set assignment: {time.perf_counter() - t0:.3f}s")

        # 10) DomainController assignment + targeted vuln generation (optimize by limiting vuln generation hits)
        t0 = time.perf_counter()
        app_types = ['VPN', 'RDP', 'ActiveDirectory', 'AdminPasswordService', 'FortiOS']
        app_versions = ['1.0', '2.0', '3.0']
        fortios_version = '3.1'
        fortios_count = 0

        # assign DC apps + vuln generation only for DCs (small set)
        for dc_device in most_connected_devices[:n_dc]:
            dc_device.addApps([
                App(id=f"ActiveDirectory_{dc_device.id}", type="ActiveDirectory", version="1.0"),
                App(id=f"Windows_Server_2019_{dc_device.id}", type="Windows_Server_2019", version="2019")
            ])
            dc_device.device_type = "DomainController"
            for app in dc_device.getApps().values():
                if app.type == 'Windows_Server_2019' and app.version == "2019":
                    try:
                        vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_DC_CVE_ID)
                        for vul, prob in vulnerabilities:
                            if random.random() < prob:
                                app.addVulnerability(vul)
                    except Exception:
                        pass
        print(f"[init] DC assignment & DC vuln gen: {time.perf_counter() - t0:.3f}s")

        # 11) add random apps to non-DCs and attach FortiOS vulnerabilities (single pass, limited fortios special-case)
        t0 = time.perf_counter()
        fortios_limit = 5
        for device in net_vals:
            if getattr(device, "device_type", None) == "DomainController":
                continue
            apps = []
            for app_type in app_types:
                if app_type == 'VPN':
                    device.wtype = 'server'
                if app_type == 'ActiveDirectory':
                    device.wtype = 'server'
                    continue
                app_version = random.choice(app_versions)
                if app_type == 'FortiOS' and fortios_count < fortios_limit:
                    app_version = fortios_version
                    fortios_count += 1
                apps.append(App(id=f"{app_type}_{device.id}", type=app_type, version=app_version))
            device.addApps(apps)

        # only run expensive vuln generation for FortiOS instances with the special version
        for device in net_vals:
            for app in device.getApps().values():
                if app.type == 'FortiOS' and app.version == fortios_version:
                    try:
                        vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_CVE_ID)
                        for vul, prob in vulnerabilities:
                            if random.random() < prob:
                                app.addVulnerability(vul)
                    except Exception:
                        pass
        print(f"[init] non-DC app assignment & FortiOS vuln gen: {time.perf_counter() - t0:.3f}s")

        # 12) starting attacker-owned selection (scaled)
        t0 = time.perf_counter()
        self.num_attacker_owned = n_owned
        keys = list(self.simulator.subnet.net.keys())
        if len(keys) == 0:
            keys = [d.id for d in net_vals]
        starting_attacker_owned = random.sample(keys, min(self.num_attacker_owned, len(keys)))
        self.starting_compromised = starting_attacker_owned

        # mark flags and Not_yet_added
        for device_id in self.starting_compromised:
            dev = self.simulator.subnet.net.get(device_id)
            if dev is None:
                # try by string id
                dev = next((d for d in net_vals if str(d.id) == str(device_id)), None)
            if dev is None:
                continue
            dev.isCompromised = True
            dev.attacker_owned = True
            dev.Known_to_attacker = True
            dev.Not_yet_added = False
        print(f"[init] attacker-owned selection & flagging: {time.perf_counter() - t0:.3f}s (chosen={len(self.starting_compromised)})")

        # 13) connect attacker-owned devices to graph (sparse for large nets)
        t0 = time.perf_counter()
        attacker_owned_devices = [device_id for device_id in self.starting_compromised]
        try:
            # Use size-aware connector
            self._connect_attacker_owned_smart(g, attacker_owned_devices)
        except Exception as e:
            print(f"[init] warning connecting attacker-owned devices: {e}")
        print(f"[init] attacker connect: {time.perf_counter() - t0:.3f}s")

        # 14) ensure at-least-one-reachable neighbor for each compromised device (best-effort)
        t0 = time.perf_counter()

        net = getattr(self.simulator.subnet, "net", {})    # device_id -> Device object
        g = getattr(self.simulator.subnet, "graph", None)
        starting = list(getattr(self, "starting_compromised", []))

        try:
            # Fast path: graph stored as dict mapping device_id -> neighbor list
            if isinstance(g, dict):
                for device_id in starting:
                    nbrs = g.get(device_id) or []
                    if not nbrs:
                        continue
                    neighbor_device_id = random.choice(nbrs)
                    nd = net.get(neighbor_device_id)
                    if nd:
                        nd.reachable_by_attacker = True

            else:
                # If this appears to be an igraph.Graph-like object, try to build name->index and adjlist once
                vs = getattr(g, "vs", None)
                if vs is None:
                    # graph doesn't expose vs (fallback): attempt to use graph.get(...) if available
                    for device_id in starting:
                        try:
                            neighbor_ids = g.get(device_id) if hasattr(g, "get") else []
                        except Exception:
                            neighbor_ids = []
                        if neighbor_ids:
                            neighbor_device_id = random.choice(neighbor_ids)
                            nd = net.get(neighbor_device_id)
                            if nd:
                                nd.reachable_by_attacker = True
                else:
                    # Build name->index mapping once (cheap overall)
                    name_to_idx = {}
                    for idx, v in enumerate(vs):
                        try:
                            nm = v.attributes().get("name") if hasattr(v, "attributes") else getattr(v, "name", None)
                        except Exception:
                            nm = getattr(v, "name", None)
                        if nm is not None:
                            name_to_idx[nm] = idx

                    # Build adjacency list once (prefer C-backed get_adjlist)
                    try:
                        adj = g.get_adjlist(mode="out")
                    except Exception:
                        # fallback building neighbors list in Python (still better than repeated find())
                        adj = [g.neighbors(i, mode="out") for i in range(len(vs))]

                    # Iterate starting compromised and pick a neighbor
                    for device_id in starting:
                        idx = name_to_idx.get(device_id)
                        if idx is None:
                            # device_id might already be an integer index; try that
                            try:
                                idx = int(device_id)
                                if idx < 0 or idx >= len(adj):
                                    continue
                            except Exception:
                                continue

                        nbr_idxs = adj[idx]
                        if not nbr_idxs:
                            continue
                        nbr_idx = random.choice(nbr_idxs)

                        # Resolve neighbor vertex name (if available) else fallback to index string
                        try:
                            neighbor_name = vs[nbr_idx].attributes().get("name") if hasattr(vs[nbr_idx], "attributes") else getattr(vs[nbr_idx], "name", None)
                        except Exception:
                            neighbor_name = None

                        if neighbor_name is None:
                            neighbor_name = str(nbr_idx)

                        nd = net.get(neighbor_name)
                        if nd:
                            nd.reachable_by_attacker = True

        except Exception:
            # Ultimate safe fallback: run the original slow but correct approach
            for device_id in starting:
                neighbor_ids = []
                try:
                    neighbor_ids = g.get(device_id)
                except Exception:
                    try:
                        vertex = g.vs.find(name=device_id)
                        nbrs = g.neighbors(vertex.index, mode="out")
                        neighbor_ids = [g.vs[n].attributes().get("name", None) for n in nbrs]
                        neighbor_ids = [n for n in neighbor_ids if n is not None]
                    except Exception:
                        neighbor_ids = []
                if neighbor_ids:
                    neighbor_device_id = random.choice(neighbor_ids)
                    nd = net.get(neighbor_device_id)
                    if nd:
                        nd.reachable_by_attacker = True

        finally:
            print(f"[init] ensure reachable neighbor for compromised: {time.perf_counter() - t0:.3f}s")


        # 15) optional random known-to-attacker compromise (legacy)
        t0 = time.perf_counter()
        init_ratio = getattr(self, "intial_ratio_compromise", 0.0)
        if init_ratio and init_ratio > 0.0:
            for device in net_vals:
                if not device.Not_yet_added and random.random() < init_ratio:
                    device.isCompromised = True
                    device.Known_to_attacker = True
        print(f"[init] optional random compromise: {time.perf_counter() - t0:.3f}s")

        # 16) initial workloads (tiny bootstrap)
        t0 = time.perf_counter()
        self.step_num = 0
        if (self.step_num % 10) == 0:
            if self.scaling_vulnerability:
                nC, nS = self._scaled_numloads(100, 10)
                self._generate_workloads_timed(numLoads=nC, mode=2, high=5, wtype='client', bootstrap=True)
                self._generate_workloads_timed(numLoads=nS, mode=2, high=5, wtype='server', bootstrap=True)
            else:
                self._generate_workloads_timed(numLoads=100, mode=1, high=3, wtype='client', bootstrap=True)
                self._generate_workloads_timed(numLoads=10,  mode=1, high=3, wtype='server', bootstrap=True)
        print(f"[init] initial workloads: {time.perf_counter() - t0:.3f}s")

        # 17) finalize action spaces & state
        t0 = time.perf_counter()
        num_exploits = len(getattr(self.simulator, "exploits", []))
        self.attacker_action_space = spaces.Discrete(num_exploits + 3)
        self.defender_action_space = spaces.Discrete(self.get_num_action_types(mode='defender'))
        self.state = self._get_state()
        self._prev_potential = None
        self._prev_att_potential = None
        print(f"[init] finalize & state build: {time.perf_counter() - t0:.3f}s")

        # 18) optional one-shot snapshot to disk (kept but guarded)
        t0 = time.perf_counter()
        try:
            if getattr(self, "tech", None) == "DQN":
                fname = f"initial_net_its{self.its}.pkl"
            else:
                fname = f"initial_net_DO_its{self.its}.pkl"
            if not os.path.exists(fname):
                # write only once (best-effort)
                try:
                    with open(fname, 'wb') as f:
                        pickle.dump({'simulator': self.simulator, 'state': self.state}, f)
                    print(f"[init] snapshot written: {fname}")
                except Exception:
                    # ignore I/O path failures
                    pass
        except Exception:
            pass
        print(f"[init] snapshot step: {time.perf_counter() - t0:.3f}s")

        total_dt = time.perf_counter() - t0_total
        print(f"[init] environment initialization completed in {total_dt:.2f}s")

        return self.state



    def reset(self, from_init=True, *args, **kwargs):
        try:
            if from_init:
                if not hasattr(self, "snapshot_path"):
                    raise RuntimeError("reset(from_init=True) requires env.snapshot_path to be set")
                with open(self.snapshot_path, 'rb') as f:
                    loaded = pickle.load(f)

                if isinstance(loaded, Volt_Typhoon_CyberDefenseEnv):
                    # keep a few runtime overrides you may have set just before reset
                    preserve_keys = {
                        "snapshot_path", "time_budget_deadline", "time_budget_seconds",
                        "time_budget_exceeded"
                    }
                    preserved = {k: getattr(self, k) for k in preserve_keys if hasattr(self, k)}

                    # Replace the entire object dict with the loaded snapshot
                    self.__dict__ = copy.deepcopy(loaded.__dict__)

                    # Restore preserved runtime overrides
                    for k, v in preserved.items():
                        setattr(self, k, v)

                else:
                    # Old dict snapshot path – keep your original logic
                    self.simulator = loaded.get('simulator', getattr(loaded, 'simulator', self.simulator))
                    self.state     = loaded.get('state',     getattr(loaded, 'state',     None))

                # Rebuild cheap caches and ensure perf dict exists
                try:
                    self._rebuild_graph_cache()
                except Exception:
                    pass
                self._ensure_perf()

                # Sanity: action spaces from snapshot should exist; if not, rebuild
                if not hasattr(self, "attacker_action_space") or self.attacker_action_space is None:
                    num_exploits = len(getattr(self.simulator, "exploits", []))
                    self.attacker_action_space = spaces.Discrete(num_exploits + 3)
                if not hasattr(self, "defender_action_space") or self.defender_action_space is None:
                    self.defender_action_space = spaces.Discrete(self.get_num_action_types(mode='defender'))



            else:
                # legacy behavior: rebuild a fresh topology/workloads etc.
                self.simulator.resetAllSubnet()

                targetApps = self.simulator.generateApps(3, True, 1, vul_to=VOLT_CVE_ID)

                minVulperExp = 1
                maxVulperExp = 1

                # create 2 target exploits (original behavior)
                self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_CVE_ID)
                self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_DC_CVE_ID)

                for _, device in self.simulator.subnet.net.items():
                    device.addApps(targetApps)

                num_exploits = 1
                self.attacker_action_space = spaces.Discrete(num_exploits + 3)
                self.defender_action_space = spaces.Discrete(self.get_num_action_types(mode='defender'))

                maxVulperApp = 1
                addApps = 3

                self.simulator.generateSubnet(self.Max_network_size, addApps, 0, maxVulperApp + 1)
                g = self.simulator.subnet.initializeVoltTyGraph(self.Max_network_size)

                all_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: d.id)
                most_connected_devices = sorted(all_devices, key=lambda d: len(g.neighbors(d.id)), reverse=True)[:3]
                forced_active_ids = {device.id for device in most_connected_devices}

                starting_compromised = self.starting_compromised if getattr(self, "starting_compromised", None) is not None else []
                forced_active_ids = forced_active_ids.union(set(starting_compromised))

                initial_active_ids = {device.id for device in all_devices[:self.numOfDevice]}
                active_set_ids = initial_active_ids.union(forced_active_ids)
                for device in all_devices:
                    device.Not_yet_added = device.id not in active_set_ids

                app_types = ['VPN', 'RDP', 'ActiveDirectory', 'AdminPasswordService', 'FortiOS']
                app_versions = ['1.0', '2.0', '3.0']
                fortios_version = '3.1'
                fortios_count = 1

                most_connected_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: len(g.neighbors(d.id)), reverse=True)[:3]
                for dc_device in most_connected_devices:
                    dc_device.addApps([
                        App(id=f"ActiveDirectory_{dc_device.id}", type="ActiveDirectory", version="1.0"),
                        App(id=f"Windows_Server_2019_{dc_device.id}", type="Windows_Server_2019", version="2019")
                    ])
                    dc_device.device_type = "DomainController"

                    for app in dc_device.getApps().values():
                        if app.type == 'Windows_Server_2019' and app.version == "2019":
                            vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_DC_CVE_ID)
                            for vul, prob in vulnerabilities:
                                if random.random() < prob:
                                    app.addVulnerability(vul)

                for device in self.simulator.subnet.net.values():
                    if device.device_type == "DomainController":
                        continue
                    apps = []
                    for app_type in app_types:
                        if app_type == 'VPN':
                            device.wtype = 'server'
                        if app_type == 'ActiveDirectory':
                            device.wtype = 'server'
                            continue
                        app_version = random.choice(app_versions)
                        if app_type == 'FortiOS' and fortios_count < 2:
                            app_version = fortios_version
                            fortios_count += 1
                        apps.append(App(id=f"{app_type}_{device.id}", type=app_type, version=app_version))
                    device.addApps(apps)

                for device in self.simulator.subnet.net.values():
                    for app in device.getApps().values():
                        if app.type == 'FortiOS' and app.version == fortios_version:
                            vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_CVE_ID)
                            for vul, prob in vulnerabilities:
                                if random.random() < prob:
                                    app.addVulnerability(vul)

                # new — reuse scaling knobs for attacker-owned picks
                if self.scaling_vulnerability:
                    n_owned = max(1, int(round(self.numOfDevice * float(getattr(self, "sv_attacker_fraction", 0.05)))))
                else:
                    n_owned = 5

                # starting attacker owned (scaled)
                self.num_attacker_owned = n_owned
                starting_attacker_owned = random.sample(
                    list(self.simulator.subnet.net.keys()),
                    min(self.num_attacker_owned, len(self.simulator.subnet.net))
                )
                self.starting_compromised = starting_attacker_owned

                for device_id in self.starting_compromised:
                    self.simulator.subnet.net[device_id].isCompromised = True
                    self.simulator.subnet.net[device_id].attacker_owned = True
                    self.simulator.subnet.net[device_id].Known_to_attacker = True
                    self.simulator.subnet.net[device_id].Not_yet_added = False

                attacker_owned_devices = [device_id for device_id in self.starting_compromised]
                try:
                    # Ensure graph caches exist to help sparse path if used
                    try:
                        self._rebuild_graph_cache()
                    except Exception:
                        pass
                    self._connect_attacker_owned_smart(g, attacker_owned_devices)
                except Exception:
                    pass

                # ensure at least one reachable neighbor for each compromised device
                for device_id in self.starting_compromised:
                    try:
                        neighbor_ids = self.simulator.subnet.graph.get(device_id)
                    except Exception:
                        try:
                            vertex = self.simulator.subnet.graph.vs.find(name=device_id)
                            neighbor_ids = self.simulator.subnet.graph.neighbors(vertex.index, mode="out")
                            neighbor_ids = [self.simulator.subnet.graph.vs[n].attributes()["name"] for n in neighbor_ids]
                        except Exception:
                            neighbor_ids = []
                    if neighbor_ids:
                        neighbor_device_id = random.choice(neighbor_ids)
                        self.simulator.subnet.net[neighbor_device_id].reachable_by_attacker = True

                self.step_num = 0
                if (self.step_num % 10) == 0:
                    if self.scaling_vulnerability:
                        nC, nS = self._scaled_numloads(100, 10)
                        self._generate_workloads_timed(numLoads=nC, mode=2, high=5, wtype='client', bootstrap=True)
                        self._generate_workloads_timed(numLoads=nS, mode=2, high=5, wtype='server', bootstrap=True)
                    else:
                        self._generate_workloads_timed(numLoads=100, mode=1, high=3, wtype='client', bootstrap=True)
                        self._generate_workloads_timed(numLoads=10,  mode=1, high=3, wtype='server', bootstrap=True)

                self.state = self._get_state()
                self._prev_potential = None

        except TypeError:
            # In case someone called reset() with a legacy signature (positional only),
            # attempt to interpret args/kwargs: support reset(), reset(False), reset(True)
            # Best-effort fallback: call no-arg legacy reset code (same as from_init=False)
            self.simulator.resetAllSubnet()
            # replicate minimal legacy re-init
            self.simulator.generateSubnet(self.Max_network_size, 3, 0, 2)
            self.state = self._get_state()
            self._prev_potential = None

        # final safety: ensure state exists
        if not hasattr(self, "state") or self.state is None:
            self.state = self._get_state()

        # make sure perf dict exists (covers weird edge cases)
        self._ensure_perf()

        return self.state
