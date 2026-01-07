import gym
from gym import spaces
import numpy as np
from CDSimulator import CyberDefenseSimulator
import random
import matplotlib.pyplot as plt
import igraph as ig
import pickle
import os
import torch
import logging
import sys
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path

class CyberDefenseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.simulator = CyberDefenseSimulator()
        self.numOfDevice = 3
        self.Max_network_size = 20
        self.Min_network_size = 2
        
        # Simplified action space for demonstration
        self.defender_action_space = spaces.Discrete(4)  # We'll handle the device subset manually
        self.attacker_action_space = spaces.Discrete(1)  # We'll handle the device subset manually
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.numOfDevice * 6,), dtype=np.float32)
        self.state = None
        self.mode = None
        self.starting_compromised = None
        self.step_num = 0
        self.defender_step = 0
        self.attacker_step = 0
        self.checkpoint = None
        self.attack_id = None
        self.work_done = 0
        self.tech = "DQN"
        self.removed_cnt = 0
        self.debug=False
        self.fast_Scan = True
        self.its = None
        self.lambda_events = .7
        self.p_add = .1
        self.p_attacker = 0
        self._exploit_seed = None
        self.MaxExploits = 6
        self.common_exploit_id = None      # will hold the ID of the exploit z "known"
        self.private_exploit_id = None     # will hold the ID of the drawn z' "unknown"
        self.j_private = 0
        self.k_known = 1
        self.preknown = 0
        self.prior_pi = None
        self.zero_day = False   # Set to True to activate Bayesian zero‐day logic.
        self._rng = np.random.RandomState()
        self.max_Dz = 6
        self.checkpoint = None
        self.turbo = True
        self.workload_cap = 0 
        self.time_budget_exceeded = None #early stopping
        self.time_budget_seconds = 2.592e12
        
       
    def set_exploit_seed(self, seed: int):
        """
        Fix the exploit RNG so that all future draws come from a single
        reproducible seed.  Must be called once before stepping for the 
        entire episode.
        """
        self._exploit_seed = seed
        self._rng = np.random.RandomState(seed)

    def sample_exploits(self):
        """
        This is the internal call your environment uses whenever it
        needs a fresh random exploit (e.g. in zero-day mode).  Replace
        usages of `np.random.*` inside the env with `self._rng.*`:
        """
        # Example (pseudocode) – pick K zero‐days among N:
        n_devices = self.Max_network_size
        n_exploits = self.get_num_exploit_indices()
        # say you want to pick one random exploit index per device:
        picks = self._rng.randint(low=0, high=n_exploits, size=(n_devices,))
        return picks

    def get_num_action_types(self):
        if self.mode == 'defender':
            return 10  
        elif self.mode == 'attacker':
            return 3 
        else:
            raise ValueError("Invalid mode: must be either 'defender' or 'attacker'")

    def _get_ordered_devices(self):
        """
        Return a list of devices sorted by their ID, trimmed to exactly
        self.Max_network_size entries.
        """
        ids     = sorted(self.simulator.subnet.net.keys())
        devices = [self.simulator.subnet.net[i] for i in ids]
        return devices[: self.Max_network_size]




    def save_figure_no_overwrite(self,fig, filepath, **savefig_kwargs):
        """
        Save a matplotlib figure without overwriting.
        If `filepath` exists, appends ' (1)', ' (2)', ... before the extension.
        """
        path = Path(filepath)
        stem = path.stem
        suffix = path.suffix

        candidate = path
        counter = 1
        while candidate.exists():
            candidate = path.with_name(f"{stem} ({counter}){suffix}")
            counter += 1

        fig.savefig(candidate, **savefig_kwargs)
        print(f"🔒 Saved network visualization to {candidate}")

    def os_to_float(self, os_obj):
        """
        Encode an OperatingSystem object as a float.
        This is just an example.
        If os_obj has a `name` attribute, use its hash modulo 100,
        otherwise if it has an 'id', return that.
        Otherwise, return 0.0.
        """
        try:
            if hasattr(os_obj, 'name'):
                # Use a simple hash and modulo to keep the number in a reasonable range.
                return float(hash(os_obj.name) % 100)
            elif hasattr(os_obj, 'id'):
                return float(os_obj.id)
            elif isinstance(os_obj, str):
                return float(hash(os_obj) % 100)
            else:
                return 0.0
        except Exception as e:
            return 0.0
        
    def _get_state(self):
        devices = list(self.simulator.subnet.net.values())
        rows = []
        for d in devices:
            # OS → float (your helper)
            os_val = self.os_to_float(d.OS)

            # version might be non‑numeric
            try:
                version_val = float(d.version)
            except Exception:
                version_val = -1.0

            # compromised is always bool/int
            compromised_val = float(d.isCompromised)

            # <--- guard anomaly_score against None
            if d.anomaly_score is None:
                anomaly_val = -1.0
            else:
                try:
                    anomaly_val = float(d.anomaly_score)
                except Exception:
                    anomaly_val = -1.0

            # Known_to_attacker is bool
            known_val = float(d.Known_to_attacker)
            not_added_val = float(d.Not_yet_added)

            rows.append([os_val,
                        version_val,
                        compromised_val,
                        anomaly_val,
                        known_val,
                        not_added_val])

        state_array = np.array(rows, dtype=float)
        # pad/trim to exactly Max_network_size rows
        n_rows = state_array.shape[0]
        if n_rows < self.Max_network_size:
            pad = -np.ones((self.Max_network_size - n_rows, state_array.shape[1]))
            state_array = np.vstack([state_array, pad])
        else:
            state_array = state_array[:self.Max_network_size]

        return state_array.flatten()
    

    def _get_attacker_state(self):
        """
        Returns a vector of length (M*4 + MaxExploits), where:
        - M*4 = filtered device features (OS, version, compromised, Known_to_attacker)
        - MaxExploits = the reserved slot for exploit‐availability bits.

        Any real exploit in simulator.exploits flips its corresponding bit to 1.
        """
        # 1) Grab the “full” M×6 state and reshape
        full_flat = self._get_state()           # length = M*6
        M, C = self.Max_network_size, 6
        mat = full_flat.reshape(M, C)           # shape = (M, 6)

        # 2) Mask out any device the attacker shouldn’t see
        devices = self._get_ordered_devices()
        for i, d in enumerate(devices):
            if (not d.Known_to_attacker) or d.Not_yet_added or (not d.attacker_owned):
                mat[i, :] = -1.0
            else:
                # hide anomaly_score (col 3) and Not_yet_added (col 5)
                mat[i, 3] = -1.0
                mat[i, 5] = -1.0

        # 3) Drop columns 3 and 5, keep only [0,1,2,4] → shape (M, 4)
        kept = np.concatenate([
            mat[:, 0:3],   # cols 0,1,2
            mat[:, 4:5]    # col 4 only
        ], axis=1)         # shape = (M, 4)

        attacker_flat = kept.flatten()   # length = M*4

        # 4) Build a fixed‐length exploit‐availability vector of size MaxExploits
        MaxE = self.MaxExploits
        exploit_bits = np.zeros(MaxE, dtype=np.float32)

        # How many exploits currently exist in the simulator?
        E = self.simulator.getExploitsSize()  # 0 ≤ E ≤ MaxE
        # For each real exploit index i < E, mark exploit_bits[i] = 1.0
        for i in range(min(E, MaxE)):
            exploit_bits[i] = 1.0

        # 5) Concatenate device info (M*4) + exploit bits (MaxE) → length = M*4 + MaxE
        return np.concatenate([attacker_flat, exploit_bits]).astype(np.float32)




    def _get_defender_state(self):
        """
        Mask the full state for the defender’s view:
        – rows for not-yet-added or non-attacker-owned devices --> all –1
        – then always hide isCompromised (col 2) so defender never sees that bit
        """
        flat = self._get_state()
        M, C = self.Max_network_size, 6
        mat = flat.reshape(M, C)

        devices = self._get_ordered_devices()
        for i, d in enumerate(devices):
            if d.Not_yet_added or (not d.attacker_owned):
                mat[i, :] = -1

        mat[:, 2] = -1  # hide isCompromised everywhere
        return mat.flatten()

      

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return [seed]

        
    def checkpoint_variables(self, filename, r):
        """Save current simulator state in memory (and optionally to disk)."""
        variables = {
            'simulator': self.simulator,
            'mode': self.mode,
            'reward': r
        }
        self.checkpoint = variables  # in-memory save
        # If you still want a file backup, uncomment below:
        # with open(filename, 'wb') as f:
        #     pickle.dump(variables, f)

    def step(self, action):
        action_type, exploit_indices, device_indices, app_index = action   # action is a tuple (action_type, exploit, [device_indices],app indicies)
        reward = 0
        

        # Loop over each device index in the subset
        for device_index in device_indices:
            device = self.simulator.subnet.net[device_index]
            if self.mode == 'defender':



                if action_type == 0:
                    if not device.uncleanable:
                        device.isCompromised = False
                        #we can't clean and do work at the same time
                        device.busy= True
                elif action_type == 1:
                    reward -= .5
                    device.mask = True
                elif action_type == 2:
                    reward -= .5
                    device.lie = True
                elif action_type == 3:  # checkpoint
                    reward -= 5
                    self.checkpoint_variables('checkpoint.pkl', reward)
                elif action_type == 4:  # upgrade
                    if app_index < len(device.apps):
                        app = list(device.apps.values())[app_index]
                        max_version = max(a.version for d in self.simulator.subnet.net.values() for a in d.apps.values() if a.id == app.id)
                        app.version = max_version + 1
                        device.busy = True
                elif action_type == 5:  # restore from checkpoint
                    try:
                        with open('checkpoint.pkl', 'rb') as f:
                            variables = pickle.load(f)
                        self.simulator = variables['simulator']
                        
                    except:
                        pass
                    reward -= 5
                elif action_type == 6:  # detect
                    logs = self.simulator.logger.get_logs()
                    for log in logs:
                        predicted_kind = self.detector.predict(log["from_device"], log["to_device"])
                        reward -= .5
                        if predicted_kind == "A":
                            device = self.simulator.subnet.net[log["from_device"]]
                            device.isCompromised = False
                            device.busy = True  # Assuming the device has an attribute 'busy'

                elif action_type == 7:
                    pass


                for _, device_work in self.simulator.subnet.net.items():
                    if not device_work.busy: 
                        if device_work.workload is not None and device_work.workload.processing_time is not None and device_work.workload.processing_time > 0:
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0:
                                device_work.workload = None  
                                #print(f"Workload completed on device {device_work.id}")
                                self.work_done += 1
                            reward += self.work_done

            if self.mode == 'attacker':
                reward = reward + .1  
                if action_type == 0:  # Example of an attack action
                    for exploit_idx in exploit_indices:
                        exploit = next((e for e in self.simulator.exploits if e.id == exploit_idx), None)
                        if exploit is None:
                            continue  # Skip if exploit is not found

                        # Get the compromised device as the starting point
                        compromised_devices = [idx for idx, device in self.simulator.subnet.net.items() if device.isCompromised]
                        for device_id in compromised_devices:
                            try:
                                neighbor_ids = self.simulator.subnet.graph.get(device_id)
                            except:
                                vertex = self.simulator.subnet.graph.vs.find(name=device_id)
                                neighbor_ids = self.simulator.subnet.graph.neighbors(vertex.index, mode="out")
                                neighbor_ids = [self.simulator.subnet.graph.vs[neighbor].attributes()["name"] for neighbor in neighbor_ids]
                            if not neighbor_ids:  # If there are no neighbors, skip to next device
                                continue
                            for neighbor_id in neighbor_ids:
                                neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                                self.simulator.log_communication(device_id, neighbor_id, 'A')
                                if neighbor_device and not neighbor_device.isCompromised:
                                    for app in neighbor_device.apps.values():
                                        for vul in app.vulnerabilities:  # This means there is a match
                                            if any(vul == exploit_vul.getId() for exploit_vul in exploit.target):  # Check if the vulnerability exists in the exploit's targets
                                                neighbor_device.isCompromised = True
                                                break  # If compromised, break out of the loop
                                if neighbor_device.isCompromised:
                                    break  # If compromised, break out of the loop
                    if action_type == 1:  # run exploit
                        reward -= 2

                    if action_type == 2:  # do work on some compromised devices
                        for device in [idx for idx, device in self.simulator.subnet.net.items() if device.isCompromised]:
                            if device.workload is None:
                                workload_id = str(uuid.uuid4())  # Generate a unique workload ID
                                workload = Workload(workload_id, np.ceil(np.random.triangular(0, mode, high, 1)), self.defaultOS, self.defaultversion)
                                workload.adversarial = True
                                self.simulator.assign_workload(workload)
                            else:
                                if device.workload.adversarial and device_workload.processing_time is not None and device_workload.processing_time > 0:
                                    device.workload.processing_time -= 1
                                    if device_workload.processing_time == 0:
                                        device.clear_workload()
                                        reward -= 1
                    if action_type==3: #pass and do nothing action
                        reward = reward - .1 

        self.simulator.generate_workloads(numLoads=50, mode=2, high=3)
        info = {
            'mode': self.mode,
            'step_count': self.step_num,
            'action_taken': action,
            'work_done': self.work_done,
            'Compromised_devices': sum(1 for d in self.simulator.subnet.net.values() if d.isCompromised)
        }

        reward -= sum(1 for d in self.simulator.subnet.net.values() if d.isCompromised)
        self.state = self._get_state()
        self.simulator.system_time=self.simulator.system_time+1
        done = self._check_done()
        return self.state, reward, done, info, self.simulator.logger.get_logs()
    def reset(self):
    
        self.step_num=0
        
        self.simulator.resetAllSubnet()

        targetApps = self.simulator.generateApps(15, True, 2)

        #self.simulator.generateVul(30)
        # generate exploit, pick one exploit, check the compromised device
        minVulperExp=1
        maxVulperExp=3
        if self.attack_id is None:
            self.simulator.generateExploits(2, True, minVulperExp, maxVulperExp)
        else: 
            self.simulator.generateExploits(2, True, minVulperExp, maxVulperExp, mode = self.attack_id)


        for _, device in self.simulator.subnet.net.items():
            device.addApps(targetApps)

        num_exploits = len(self.simulator.exploits)
        self.attacker_action_space = spaces.Discrete(num_exploits+3)#plus 3 for other action flags

        numOfCompromisedDev = []
        
        resetNum = 0 #number of device to be resetted at each time
        
        maxVulperApp = 2
        if self.zero_day:
            addApps = 5
        else:
            addApps = 3
        

        self.simulator.generateSubnet(self.numOfDevice, addApps, 0, maxVulperApp+1)
        g = self.simulator.subnet.initializeRandomGraph(self.numOfDevice, 1)
        # Initialize the uncleanable device as compromised
        num_attacker_owned = 1
        starting_attacker_owned = random.sample(list(self.simulator.subnet.net.keys()), num_attacker_owned)

        self.starting_compromised = starting_attacker_owned

        for device_id in self.starting_compromised:
            self.simulator.subnet.net[device_id].isCompromised = True
            self.simulator.subnet.net[device_id].attacker_owned = True
        #print("Cannot clean:"+str(device_id))
            
            
        
        
        
        self.simulator.generate_workloads(numLoads = 10,mode = 2,high = 5)
    
        self.state =np.zeros((self.numOfDevice , 6))
        

        
        return self.state


    def generate_viz(self, filepath: str = "network_viz.png"):
        """
        Draw the current subnet and save it to `filepath`.
        - DomainControllers: large squares (red if compromised; lightgray if healthy)
        - Other compromised devices: red stars
        - Other healthy devices: lightblue circles
        """
        # 1) grab graph & device mapping
        g          = self.simulator.subnet.graph
        device_map = self.simulator.subnet.net

        # 2) layout
        layout = g.layout("fr")
        coords = {i: layout[i] for i in range(g.vcount())}

        # 3) prep figure
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,6))

        # 4) draw edges
        for e in g.es:
            src, dst = e.tuple
            x0, y0 = coords[src]
            x1, y1 = coords[dst]
            ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5, zorder=1)

        # 5) collect per‐category lists
        dc_x, dc_y = [], []
        dc_color   = []
        comp_x, comp_y = [], []
        healthy_x, healthy_y = [], []

        for idx in range(g.vcount()):
            x, y = coords[idx]
            dev  = device_map[idx]

            # DomainController
            if dev.device_type == "DomainController":
                dc_x.append(x); dc_y.append(y)
                # compromised? red, else lightgray
                dc_color.append("red" if dev.isCompromised else "lightgray")

            # other compromised
            elif dev.isCompromised:
                comp_x.append(x); comp_y.append(y)

            # everyone else healthy
            else:
                healthy_x.append(x); healthy_y.append(y)

        # 6) plot: healthy circles
        ax.scatter(healthy_x, healthy_y,
                   s=100, marker="o", facecolors="lightblue", 
                   edgecolors="k", label="Healthy", zorder=2)

        # 7) compromised stars
        ax.scatter(comp_x, comp_y,
                   s=200, marker="*", facecolors="red", 
                   edgecolors="k", label="Compromised", zorder=3)

        # 8) domain controller squares
        ax.scatter(dc_x, dc_y,
                   s=300, marker="s", facecolors=dc_color, 
                   edgecolors="k", label="DomainController", zorder=4)

        # 9) finalize
        ax.axis("off")
        ax.legend(loc="upper right")

        # 10) save
        #fig.savefig(filepath, bbox_inches="tight")
        self.save_figure_no_overwrite(fig, filepath, bbox_inches="tight")
        plt.close(fig)
        print(f"🔒 Saved network visualization to {filepath}")
        
    def _check_done(self,action):

        if self.step_num > 1000:
            return True
            
        return False


    def sample_action(self):
        # 1) Pick a random action‐type
        if self.mode == 'defender':
            action_type = self.defender_action_space.sample()
        elif self.mode == 'attacker':
            action_type = self.attacker_action_space.sample()
        else:
            raise ValueError("Invalid mode")

        # 2) Random subset of devices
        device_indices = random.sample(
            list(self.simulator.subnet.net.keys()),
            k=random.randint(1, self.numOfDevice)
        )

        # 3) Exactly one exploit‐slot from 0..MaxExploits−1
        exploit_idx = random.randrange(self.MaxExploits)
        exploit_indices = np.array([exploit_idx], dtype=int)

        # 4) Single random app index (or zero)
        num_apps = self.get_num_app_indices()
        app_index = random.randint(0, num_apps - 1) if num_apps > 0 else 0

        return (action_type, exploit_indices, device_indices, app_index)




    def evolve_network(self):
        """
        Optimized network evolution:
        - O(λ) event handling (no O(λ·N) list rebuilds)
        - changed-only vertex attribute writes
        - star reconnection for attacker-owned devices (O(A) vs O(A^2))
        - PA for newly activated non-attackers using one degree snapshot per evolve
        - conditional cache rebuild (only if edges changed)
        """
        # --- params / RNG ---
        lambda_events = getattr(self, "lambda_events", 0.0)
        p_add         = getattr(self, "p_add", 0.5)
        p_attacker    = getattr(self, "p_attacker", 0.0)

        g = self.simulator.subnet.graph

        # ---------- build or validate name->index mapping ----------
        # Create a robust mapping from possible vertex 'name' representations to index.
        # We store name as-is, str(name), and also the integer index, to be resilient.
        if not hasattr(self, "_node_name_to_idx") or self._node_name_to_idx is None:
            m = {}
            try:
                # enumerate vertices once
                for idx, v in enumerate(g.vs):
                    try:
                        vname = v["name"]
                    except Exception:
                        vname = None
                    # primary keys we store: raw value, stringified, and the numeric index
                    if vname is not None:
                        m[vname] = idx
                        try:
                            m[str(vname)] = idx
                        except Exception:
                            pass
                    m[idx] = idx
                    try:
                        m[str(idx)] = idx
                    except Exception:
                        pass
            except Exception:
                m = {}
            self._node_name_to_idx = m

        # helper to get a vertex index by name with caching and a single fallback to igraph.find
        def _get_vid_index(nid):
            # fastest path: direct dict lookup (handles int/string forms)
            if nid in self._node_name_to_idx:
                return self._node_name_to_idx[nid]
            # try stringified key
            s = None
            try:
                s = str(nid)
            except Exception:
                s = None
            if s is not None and s in self._node_name_to_idx:
                return self._node_name_to_idx[s]
            # fallback: try igraph find once and cache result (avoid repeated scans)
            try:
                idx = g.vs.find(name=nid).index
                self._node_name_to_idx[nid] = idx
                try:
                    self._node_name_to_idx[str(nid)] = idx
                except Exception:
                    pass
                return idx
            except Exception:
                # not found
                return None

        # --- initialize incremental active/inactive sets on first run ---
        if not hasattr(self, "_active_ids") or not hasattr(self, "_inactive_ids"):
            act, inact = set(), set()
            for d in self.simulator.subnet.net.values():
                (act if not d.Not_yet_added else inact).add(d.id)
            self._active_ids = act
            self._inactive_ids = inact

        newly_activated = set()
        newly_deactivated = set()

        # We’ll track whether we changed topology (edges) so we know if caches must rebuild.
        edges_changed = False

        # --- Poisson number of events ---
        num_events = np.random.poisson(lam=lambda_events)

        # Fast pick helpers from the incremental sets
        def pick_from(s):
            # random choice from a set without materializing a big list repeatedly
            if not s:
                return None
            return random.choice(tuple(s))

        # --- handle events (activate or deactivate) ---
        for _ in range(num_events):
            if random.random() < p_add:
                # Addition event: activate an inactive node (if any)
                node_id = pick_from(self._inactive_ids)
                if node_id is not None:
                    dev = self.simulator.subnet.net[node_id]
                    dev.Not_yet_added = False
                    self._inactive_ids.discard(node_id)
                    self._active_ids.add(node_id)
                    newly_activated.add(node_id)

                    # With probability p_attacker, mark as attacker owned on activation.
                    if random.random() < p_attacker:
                        dev.isCompromised = True
                        dev.attacker_owned = True
                        dev.Known_to_attacker = True
                        if self.debug:
                            logging.debug(f"Activated node {node_id} as attacker-owned via Poisson event.")
                    else:
                        if self.debug:
                            logging.debug(f"Activated node {node_id} via Poisson event.")
            else:
                # Removal event: deactivate a currently active node if over minimum
                if len(self._active_ids) > max(getattr(self, "numOfDevice", 0),
                                            getattr(self, "Min_network_size", 0)):
                    node_id = pick_from(self._active_ids)
                    if node_id is not None:
                        dev = self.simulator.subnet.net[node_id]
                        dev.Not_yet_added = True
                        dev.workload = None
                        dev.busy_time = 0
                        dev.removed_before = 1
                        self._active_ids.discard(node_id)
                        self._inactive_ids.add(node_id)
                        newly_deactivated.add(node_id)
                        if self.debug:
                            logging.debug(f"Removed node {node_id} via Poisson event.")

        # --- update vertex["active"] ONLY for changed vertices ---
        if newly_activated or newly_deactivated:
            changed = newly_activated | newly_deactivated
            # Build a tiny lookup for speed
            changed_map = {vid: True for vid in newly_activated}
            changed_map.update({vid: False for vid in newly_deactivated})
            # Update attributes on just these vertices using fast index lookup
            for vid in changed:
                idx = _get_vid_index(vid)
                if idx is None:
                    # vertex not found; skip
                    continue
                try:
                    g.vs[idx]["active"] = changed_map[vid]
                except Exception:
                    # tolerate unexpected attribute errors
                    try:
                        v = g.vs[idx]
                        v["active"] = changed_map[vid]
                    except Exception:
                        continue

        # --- Reconnect attacker-owned devices (STAR, not clique) ---
        active_attacker_ids = [d.id for d in self.simulator.subnet.net.values()
                            if d.attacker_owned and (d.id in self._active_ids)]

        if active_attacker_ids:
            hub = active_attacker_ids[0]
            new_edges = []
            hub_idx = _get_vid_index(hub)

            for nid in active_attacker_ids[1:]:
                n_idx = _get_vid_index(nid)
                if n_idx is None:
                    continue
                # Create bidirectional edges if missing.
                if hub_idx is not None:
                    if g.get_eid(hub_idx, n_idx, directed=True, error=False) == -1:
                        new_edges.append((hub_idx, n_idx))
                    if g.get_eid(n_idx, hub_idx, directed=True, error=False) == -1:
                        new_edges.append((n_idx, hub_idx))

            if new_edges:
                g.add_edges(new_edges)
                # Ensure 'blocked' attribute exists and default False for new edges
                if "blocked" not in g.es.attributes():
                    g.es["blocked"] = [False] * g.ecount()
                else:
                    start = g.ecount() - len(new_edges)
                    try:
                        g.es[start:]["blocked"] = [False] * len(new_edges)
                    except Exception:
                        # fallback: set the last edges individually
                        for i in range(start, g.ecount()):
                            try:
                                g.es[i]["blocked"] = False
                            except Exception:
                                pass
                edges_changed = True

        # --- For newly activated non-attackers, ensure connectivity if isolated (PA m=1) ---
        need_pa = False
        pa_candidates = []
        if newly_activated:
            # quick filter: just the newly activated, non-attacker, still active
            for nid in list(newly_activated):
                dev = self.simulator.subnet.net[nid]
                if dev.Not_yet_added or dev.attacker_owned:
                    continue
                idx = _get_vid_index(nid)
                if idx is None:
                    continue
                if g.degree(idx) < 1:
                    need_pa = True
                    break

            if need_pa:
                # Build candidates = list of *active* vertices (excluding the isolated one per-attachment)
                active_vertex_indices = []
                for aid in self._active_ids:
                    idx = _get_vid_index(aid)
                    if idx is not None:
                        active_vertex_indices.append(idx)

                if active_vertex_indices:
                    degs = g.degree(active_vertex_indices, mode="ALL")
                    pa_candidates = list(zip(active_vertex_indices, [d + 1 for d in degs]))

                if pa_candidates:
                    # Precompute cumulative weights for O(log M) sampling
                    import bisect
                    weights = [w for _, w in pa_candidates]
                    cdf = []
                    s = 0
                    for w in weights:
                        s += w
                        cdf.append(s)

                    def sample_pa():
                        if not cdf:
                            return None
                        r = random.uniform(0, cdf[-1])
                        j = bisect.bisect_left(cdf, r)
                        return pa_candidates[j][0]  # return vertex idx

                    for nid in list(newly_activated):
                        dev = self.simulator.subnet.net[nid]
                        if dev.Not_yet_added or dev.attacker_owned:
                            continue
                        idx = _get_vid_index(nid)
                        if idx is None:
                            continue
                        if g.degree(idx) < 1:
                            tgt_idx = sample_pa()
                            if tgt_idx is None:
                                continue
                            # add a single edge v -> tgt (and optionally tgt -> v)
                            if g.get_eid(idx, tgt_idx, directed=True, error=False) == -1:
                                g.add_edges([(idx, tgt_idx)])
                                edges_changed = True
                                # ensure blocked attr for the new edge
                                if "blocked" not in g.es.attributes():
                                    g.es["blocked"] = [False] * g.ecount()
                                else:
                                    try:
                                        g.es[g.ecount()-1]["blocked"] = False
                                    except Exception:
                                        pass

        # --- Rebuild caches only if edges actually changed ---
        if edges_changed:
            # If topology changed but vertex set remains same, our index map is still valid.
            # If you ever add/remove vertices elsewhere, you should rebuild _node_name_to_idx accordingly.
            try:
                self._rebuild_graph_cache()
            finally:
                # If the graph library reindexed vertices in rebuild, try to refresh index map conservatively.
                # Rebuild mapping to be safe (cheap relative to heavy graph ops).
                try:
                    m = {}
                    for idx, v in enumerate(g.vs):
                        try:
                            vname = v["name"]
                        except Exception:
                            vname = None
                        if vname is not None:
                            m[vname] = idx
                            try:
                                m[str(vname)] = idx
                            except Exception:
                                pass
                        m[idx] = idx
                        try:
                            m[str(idx)] = idx
                        except Exception:
                            pass
                    self._node_name_to_idx = m
                except Exception:
                    # if rebuild fails, keep old map (best-effort)
                    pass


