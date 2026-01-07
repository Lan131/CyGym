import copy
import matplotlib.pyplot as plt
import math
import time
import datetime
import random
import numpy as np
import igraph as ig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import uuid
from CDSimulatorComponents import *
import pandas as pd

class CyberDefenseSimulator:
    """
    The simulator, including method to generate subnet, apps, network (which contains a set of subnet), vulnerabilities, exploits.
    Constructor initialize the default OS, App, and Vulnerability.
    """

    def __init__(self):
        # Subnet: set of devices
        self.subnet = Subnet()
        self.apps = set()
        self.logger = Logger()
        self.system_time = 0
        self.detector = Detector()
        self.vulneralbilities = set()
        self.exploits = []
        self.defaultOS = OperatingSystem(0, "OS", 1.0)
        self.defaultversion = 1
        self.defaultApp = App(0, "game", 1.0)
        self.defaulVul = Vulnerability(0, "unknown", self.defaultApp)
        self.df = pd.read_csv('CVE.csv')  # Load CSV into DataFrame

        # Fast mapping from vertex name -> igraph index (keeps igraph lookups O(1))
        # This map is kept up-to-date lazily; call build/update methods when you mutate the graph.
        self._vertex_name_to_index = {}
        self._build_vertex_name_index_map()

    # -------------------------
    # Vertex name <-> index cache helpers
    # -------------------------
    def _build_vertex_name_index_map(self):
        """Build full mapping name -> index for the current graph (safe to call after graph creation/mutation)."""
        try:
            g = getattr(self.subnet, "graph", None)
            if g is None:
                self._vertex_name_to_index = {}
                return
            mapping = {}
            # iterate vertices once
            for idx in range(len(g.vs)):
                try:
                    name = g.vs[idx]["name"] if "name" in g.vs.attributes() else g.vs[idx].index
                except Exception:
                    name = g.vs[idx].index
                mapping[str(name)] = idx
            self._vertex_name_to_index = mapping
        except Exception:
            # fallback to empty
            self._vertex_name_to_index = {}

    def _update_vertex_map_for_vertex(self, vertex_index: int):
        """Add/refresh mapping for a single vertex index."""
        try:
            g = getattr(self.subnet, "graph", None)
            if g is None:
                return
            name = g.vs[vertex_index]["name"] if "name" in g.vs.attributes() else g.vs[vertex_index].index
            self._vertex_name_to_index[str(name)] = vertex_index
        except Exception:
            return

    def _get_neighbor_ids(self, device_id):
        """
        Return neighbor device 'name' ids for the given device_id (string or number).
        Uses the cached name->index mapping; if missing attempts an igraph find once and updates the cache.
        Returns a list (possibly empty) of neighbor ids (strings).
        """
        g = getattr(self.subnet, "graph", None)
        if g is None:
            return []

        key = str(device_id)
        idx = self._vertex_name_to_index.get(key, None)
        if idx is None:
            # lazy fallback: try to find and add to cache
            try:
                v = g.vs.find(name=device_id)
                idx = v.index
                self._vertex_name_to_index[key] = idx
            except Exception:
                return []

        try:
            nbr_indices = g.neighbors(idx, mode="out")
        except Exception:
            # if igraph fails, return empty
            return []

        neighbor_ids = []
        for n in nbr_indices:
            try:
                name = g.vs[n]["name"] if "name" in g.vs.attributes() else g.vs[n].index
                neighbor_ids.append(str(name))
                # keep cache warm
                self._vertex_name_to_index[str(name)] = n
            except Exception:
                neighbor_ids.append(str(n))
                self._vertex_name_to_index[str(n)] = n
        return neighbor_ids

    # -------------------------
    # Workload assignment (fast path)
    # -------------------------

    def log_communication(self, from_device, to_device, kind):
        self.logger.log(self.system_time, from_device, to_device, kind)

    def assign_workload(self,
                        workload,
                        wtype=None,
                        adversarial=False,
                        preferred_device_id=None,
                        allow_neighbor_lookup=True,
                        neighbor_lookup_prob: float = 0.05):
        """
        Fast assignment helper with optional neighbor lookup probability.

        Args:
            workload: Workload object (may have attributes origin_device_id / target_device_id / wtype)
            wtype: optional override for workload wtype
            adversarial: mark assigned workload as adversarial if True
            preferred_device_id: try direct assignment to this device id first (fast O(1))
            allow_neighbor_lookup: if False, never attempt neighbor fallback
            neighbor_lookup_prob: probability in [0,1] to perform neighbor lookup when fallback path used

        Returns:
            True if assigned, False otherwise.
        """
        # Helper to test device suitability
        def _device_matches(dev, w_OS, w_ver, w_wtype):
            if getattr(dev, "Not_yet_added", False):
                return False
            if getattr(dev, "workload", None) is not None:
                return False
            if getattr(dev, "busy_time", 0) and dev.busy_time > 0:
                return False
            if w_wtype is not None:
                # client vs server preferences (server devices should be used for server workloads)
                if w_wtype == 'client' and getattr(dev, "wtype", None) == 'server':
                    return False
                if w_wtype == 'server' and getattr(dev, "wtype", None) != 'server':
                    return False
            if w_OS is not None:
                dev_OS = getattr(dev, "OS", None)
                if dev_OS is None:
                    return False
                # match by object id or OS.id if available
                if dev_OS is not w_OS and getattr(dev_OS, "id", None) != getattr(w_OS, "id", None):
                    return False
            if w_ver is not None:
                if str(getattr(dev, "version", "")).strip() != str(w_ver).strip():
                    return False
            return True

        # 1) Try preferred_device_id (fast)
        if preferred_device_id is not None:
            dev = self.subnet.net.get(preferred_device_id)
            if dev is not None:
                try:
                    w_OS = getattr(workload, "OS", None)
                    w_ver = getattr(workload, "version", None)
                    w_wtype = getattr(workload, "wtype", wtype)
                except Exception:
                    w_OS = None; w_ver = None; w_wtype = wtype
                if _device_matches(dev, w_OS, w_ver, w_wtype):
                    dev.workload = workload
                    workload.assigned = True
                    workload.adversarial = bool(adversarial)
                    return True

        # 2) Try explicit target_device_id stored on workload
        tgid = getattr(workload, "target_device_id", None)
        if tgid is not None:
            dev = self.subnet.net.get(tgid)
            if dev is not None:
                if _device_matches(dev, getattr(workload, "OS", None), getattr(workload, "version", None), getattr(workload, "wtype", wtype)):
                    dev.workload = workload
                    workload.assigned = True
                    workload.adversarial = bool(adversarial)
                    return True

        # 3) Try a single-pass neighbor lookup if origin exists and allowed by probability
        try:
            w_OS = getattr(workload, "OS", None)
            w_ver = getattr(workload, "version", None)
            w_wtype = getattr(workload, "wtype", wtype)
        except Exception:
            w_OS = None; w_ver = None; w_wtype = wtype

        origin = getattr(workload, "origin_device_id", None)

        if origin is not None and allow_neighbor_lookup and random.random() < float(neighbor_lookup_prob):
            try:
                neighbor_ids = self.subnet.graph.get(origin)
            except Exception:
                # single fallback to igraph neighbors (rare)
                try:
                    vertex = self.subnet.graph.vs.find(name=origin)
                    nbr_idx = self.subnet.graph.neighbors(vertex.index, mode="out")
                    neighbor_ids = [self.subnet.graph.vs[n].attributes().get("name", None) for n in nbr_idx]
                    neighbor_ids = [n for n in neighbor_ids if n is not None]
                except Exception:
                    neighbor_ids = []
            for nid in neighbor_ids:
                nd = self.subnet.net.get(nid)
                if nd is None:
                    continue
                if _device_matches(nd, w_OS, w_ver, w_wtype):
                    nd.workload = workload
                    workload.assigned = True
                    workload.adversarial = bool(adversarial)
                    return True

        # 4) Last-ditch: linear scan first-fit (only if workload encodes OS or wtype) — note this is O(V)
        if w_OS is not None or w_wtype is not None:
            for dev_id, dev in self.subnet.net.items():
                if _device_matches(dev, w_OS, w_ver, w_wtype):
                    dev.workload = workload
                    workload.assigned = True
                    workload.adversarial = bool(adversarial)
                    return True

        # 5) no assignment
        return False

    # -------------------------
    # Generate (batch) workloads
    # -------------------------
    def generate_workloads(self,
                           numLoads,
                           mode,
                           high,
                           wtype='client',
                           adversarial=False,
                           lazy_generate: bool | None = None,
                           lazy_local_prob: float = 0.9,
                           neighbor_lookup_prob: float = 0.05):
        """
        Efficient batch workload generator with an optional 'lazy_generate' mode.

        Args:
            numLoads: requested number of workloads
            mode, high: used for processing_time generation (kept for compatibility)
            wtype: 'client' or 'server'
            adversarial: mark produced workloads as adversarial if True
            lazy_generate: if True, create workloads *at* sampled device but with probability (1 - lazy_local_prob)
                           attempt to assign them elsewhere (neighbor lookup controlled by neighbor_lookup_prob).
                           If None (default) we auto-enable lazy mode when subnet size > 500.
            lazy_local_prob: when lazy_generate True, probability to assign locally (no graph lookup).
            neighbor_lookup_prob: when trying to place a non-local workload, probability to perform the neighbor lookup.

        Returns:
            list of Workload objects created (some may be assigned, some may be left unassigned if no candidate)
        """
        if numLoads <= 0:
            return []

        # Auto-enable lazy mode on large graphs if caller didn't explicitly set it
        if lazy_generate is None:
            lazy_generate = (len(self.subnet.net) > 500)

        # Build free candidate list once (O(V) single pass)
        free_candidates = []
        for dev_id, dev in self.subnet.net.items():
            if getattr(dev, "Not_yet_added", False):
                continue
            if getattr(dev, "workload", None) is not None:
                continue
            if getattr(dev, "busy_time", 0) and dev.busy_time > 0:
                continue
            dev_wtype = getattr(dev, "wtype", None)
            if wtype == 'client' and dev_wtype == 'server':
                continue
            if wtype == 'server' and dev_wtype != 'server':
                continue
            free_candidates.append(dev_id)

        if not free_candidates:
            return []

        k = min(int(numLoads), len(free_candidates))
        try:
            sampled_ids = random.sample(free_candidates, k)
        except ValueError:
            sampled_ids = free_candidates[:k]

        created_workloads = []
        for did in sampled_ids:
            dev = self.subnet.net.get(did)
            if dev is None:
                continue
            wid = str(uuid.uuid4())
            processing_time = int(np.ceil(np.random.triangular(0, mode, high, 1)))
            dev_ver = getattr(dev, "version", None)
            w_ver = str(dev_ver) if dev_ver is not None else str(self.defaultversion)
            # create workload targeted at this device by default (helps compatibility)
            w = Workload(wid, processing_time, dev.OS, w_ver)
            setattr(w, "origin_device_id", did)
            setattr(w, "target_device_id", did)
            setattr(w, "wtype", wtype)

            if not lazy_generate:
                # fast direct assign
                dev.workload = w
                w.assigned = True
                w.adversarial = bool(adversarial)
                created_workloads.append(w)
                continue

            # lazy_generate path:
            # With probability lazy_local_prob assign directly to origin (no lookup).
            # Otherwise, try to assign elsewhere (neighbor lookup with probability neighbor_lookup_prob).
            if random.random() < float(lazy_local_prob):
                dev.workload = w
                w.assigned = True
                w.adversarial = bool(adversarial)
                created_workloads.append(w)
                continue

            # non-local attempt: use assign_workload which encapsulates neighbor lookup + last-ditch linear scan.
            assigned = self.assign_workload(
                w,
                wtype=wtype,
                adversarial=adversarial,
                preferred_device_id=None,
                allow_neighbor_lookup=True,
                neighbor_lookup_prob=float(neighbor_lookup_prob)
            )
            # assign_workload may have assigned via neighbor or via linear scan.
            w.assigned = bool(assigned)
            created_workloads.append(w)

        return created_workloads

    def process_subnet_workloads(self, debug=False):
        """
        Tick processing_time of any assigned workloads by 1.
        When processing_time reaches 0, clear the workload from the device.
        Returns the count of completed workloads in this tick.
        """
        completed_workloads = 0
        for _, device in self.subnet.net.items():
            wl = getattr(device, "workload", None)
            if wl is not None and getattr(wl, "processing_time", None) is not None:
                if wl.processing_time > 0:
                    wl.processing_time = wl.processing_time - 1
                    if wl.processing_time == 0:
                        # clear the workload
                        device.workload = None
                        completed_workloads += 1
        return completed_workloads

    # -------------------------
    # Reset / generation helpers
    # -------------------------
    def resetAllSubnet(self):
        """
        reset everything in the subnet, meaning DELETING all devices & graph
        """
        self.subnet.net.clear()
        self.subnet.graph = ig.Graph()
        self.subnet.resetAllCompromisedSubnet()
        # clear vertex map
        self._vertex_name_to_index = {}

        # If devices existed previously, those sets remain but we cleared net; leave apps/vulns unchanged

    def resetByNumSubnet(self, resetNum):
        # randomDevices are a LIST of randomly selected device's ID
        try:
            randomDevices = self.randomSampleGenerator(list(self.subnet.net.keys()), resetNum)
            self.subnet.resetSomeCompromisedSubnet(randomDevices)
        except Exception:
            pass

    def getSubnetSize(self):
        """Returns the size of the subnet (number of devices)"""
        return len(self.subnet.net)

    def getNetworkSize(self):
        """Returns size of the network (number of subnets)"""
        return len(self.network)

    def getVulneralbilitiesSize(self):
        """Returns number of vulnerabilities"""
        return len(self.vulneralbilities)

    def getExploitsSize(self):
        """Returns number of exploits"""
        return len(self.exploits)

    def generateApps(self, numOfApps, addVul=False, numOfVul=1, vul_to=None):
        app_list = []
        if isinstance(numOfApps, int):
            for count in range(numOfApps):
                random_app = App(count, self.AppTypeGenerator(),
                                 self.randomNumberGenerator(1.0, 3.0))
                if addVul:
                    if vul_to is None:
                        appVul = self.generateVul(numOfVul, random_app)
                    else:
                        appVul = self.generateVul(numOfVul, random_app, mode="target", vulID=vul_to)

                    if len(self.vulneralbilities) >= numOfVul:
                        for i in range(numOfVul):
                            random_app.addVulnerability(self.randomSampleGenerator(self.vulneralbilities))
                    else:
                        random_app.addVulnerability(self.defaulVul)
                app_list.append(random_app)
                self.apps.add(random_app)
        return app_list

    def generateDevice(self, numOfApps=1, minVulperApp=0, maxVulperApp=0):
        if isinstance(numOfApps, int):
            AppsList = []
            for i in range(minVulperApp, maxVulperApp):
                AppsList.append(self.randomSampleGenerator(self.apps))
            currSize = self.getSubnetSize()
            newDevice = Device(currSize, self.defaultOS, self.defaultversion, 0)
            newDevice.addApps(AppsList)
            return newDevice
        else:
            print("not a valid input for generate Device")

    def generateSubnet(self, numOfDevice, addApps=None, minVulperApp=0, maxVulperApp=0):
        """
        add device to our subnet with the specified number of app(optional),
        number range of vulnerabilities to be added to the device.
        """
        if isinstance(numOfDevice, int):
            for count in range(numOfDevice):
                if addApps is None:
                    newDevice = self.generateDevice()
                else:
                    newDevice = self.generateDevice(addApps, minVulperApp, maxVulperApp)
                self.subnet.addDevices(newDevice)

            # The graph / subnet likely mutated; rebuild the vertex index map for fast lookups.
            try:
                self._build_vertex_name_index_map()
            except Exception:
                pass
        else:
            print("not a valid input for generate subnet")

    def redeploy_apps_with_unique_vulns(self,
                                       num_apps=20,
                                       vul_per_app=2,
                                       min_apps_per_device=1,
                                       max_apps_per_device=3):
        """
        Clears all existing apps/vulns from simulator and devices,
        then regenerates `num_apps` apps (each with `vul_per_app` vulns)
        and assigns each device between min_apps_per_device and max_apps_per_device of them.
        """
        # 1) Clear everything
        self.apps.clear()
        self.vulneralbilities.clear()
        for device in self.subnet.net.values():
            device.apps.clear()

        # 2) Generate a pool of apps (with vul_per_app vulnerabilities each)
        app_pool = self.generateApps(
            numOfApps=num_apps,
            addVul=True,
            numOfVul=vul_per_app
        )

        # 3) Shuffle & assign to devices
        for device in self.subnet.net.values():
            k = random.randint(min_apps_per_device, max_apps_per_device)
            chosen = random.sample(app_pool, k)
            device.addApps(chosen)

    def generateVul(self, numOfVul, targetApp=None, targetOS=None, mode="random", vulID=None):
        assert mode in ('random', 'target')
        vul_probabilities = []

        if mode == "random":
            sampled_data = self.df.sample(n=numOfVul)
            for index, row in sampled_data.iterrows():
                vulType = 'unknown'
                target = targetApp if targetApp else (targetOS if targetOS else self.defaultApp)
                newVul = Vulnerability(row['matchCriteriaId'], vulType, target)
                self.vulneralbilities.add(newVul)
                newVul.exploitability_score = row.get('exploitabilityScore', 0.0)
                newVul.impact_score = row.get('impactScore', 0.0)
                probability = newVul.exploitability_score / 10.0
                vul_probabilities.append((newVul, probability))
                if targetApp or targetOS:
                    target.addVulnerability(newVul)
        elif mode == "target" and vulID is not None:
            row = self.df[self.df['matchCriteriaId'] == vulID]
            if not row.empty:
                row = row.iloc[0]
                vulType = 'unknown'
                target = targetApp if targetApp else (targetOS if targetOS else self.defaultApp)
                newVul = Vulnerability(row['matchCriteriaId'], vulType, target)
                newVul.exploitability_score = row.get('exploitabilityScore', 0.0)
                newVul.impact_score = row.get('impactScore', 0.0)
                self.vulneralbilities.add(newVul)
                probability = newVul.exploitability_score / 10.0
                vul_probabilities.append((newVul, probability))
                if targetApp or targetOS:
                    target.addVulnerability(newVul)
        else:
            print("Invalid mode or missing vulID for target mode")

        return vul_probabilities

    def changeVulTarget(self):
        if len(self.apps) == 0:
            print("not enough apps")
        if len(self.apps) < len(self.vulneralbilities):
            print("apps less than vul")
        for vul in self.vulneralbilities:
            try:
                vul.setTarget(self.randomSampleGenerator(self.apps))
            except Exception:
                pass

    def generateExploits(
        self,
        numOfExploits,
        addVul=False,
        minVulperExp=0,
        maxVulperExp=0,
        mode="random",
        expID=None,
        discovered=False
    ):
        assert mode in ("random", "target"), "mode must be 'random' or 'target'"

        def _attach_extra(exploit):
            if not addVul or not self.vulneralbilities:
                return
            k = int(self.randomNumberGenerator(minVulperExp, maxVulperExp))
            for _ in range(k):
                extra_v = random.choice(list(self.vulneralbilities))
                exploit.setTargetVul(extra_v)

        if mode == "random":
            sampled = self.df.sample(n=numOfExploits, replace=False)
            for _, row in sampled.iterrows():
                new_vul = Vulnerability(
                    row["matchCriteriaId"],
                    "unknown",
                    self.defaultApp
                )
                new_vul.exploitability_score = row.get("exploitabilityScore", 0.0)
                new_vul.impact_score = row.get("impactScore", 0.0)
                self.vulneralbilities.add(new_vul)

                exp = Exploit(new_vul.id, "unknown", self.defaultApp)
                exp.setTargetVul(new_vul)
                _attach_extra(exp)
                self.exploits.append(exp)

        else:  # mode == "target"
            if expID is None:
                print("Invalid mode or missing expID for target mode")
                return
            subset = self.df[self.df["matchCriteriaId"] == expID]
            if subset.empty:
                print(f"No CVE entry found for ID={expID}")
                return
            row = subset.iloc[0]
            for _ in range(numOfExploits):
                new_vul = Vulnerability(
                    row["matchCriteriaId"],
                    "unknown",
                    self.defaultApp
                )
                new_vul.exploitability_score = row.get("exploitabilityScore", 0.0)
                new_vul.impact_score = row.get("impactScore", 0.0)
                self.vulneralbilities.add(new_vul)

                exp = Exploit(new_vul.id, "unknown", self.defaultApp)
                exp.discovered = discovered
                exp.setTargetVul(new_vul)
                _attach_extra(exp)
                self.exploits.append(exp)

    def attackSubnet(self, exploit):
        """attack the subnet with a SINGLE Exploit that has valid vulnerabilities"""
        print(f'expected target is vulneralbility with id:{exploit.target.keys()}')
        self.subnet.attack(exploit, self.subnet.net)

    def randomNumberGenerator(self, a, b):
        if (a == b):
            return a
        randomNum = random.randint(int(a*10), int(b*10))/10
        return randomNum

    def randomRangeGenerator(self, a=1.0, b=1.0):
        num1 = self.randomNumberGenerator(a, b)
        num2 = self.randomNumberGenerator(a, b)
        maxR = max(num1, num2)
        minR = min(num1, num2)
        return minR, maxR

    def AppTypeGenerator(self):
        types = ["game", "lifestype", "social",
                 "entertainment", "productivity"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def VulTypeGenerator(self):
        types = ["unknown", "misconfigurations", "outdated software",
                 "unauthorized access", "weak user credentials", "Unsecured APIs"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def ExpTypeGenerator(self):
        types = ["unknown", "known"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def randomSampleGenerator(self, sampleSet, numOfSample=1):
        listSet = list(sampleSet)
        if numOfSample == 1 or len(sampleSet) < numOfSample:
            return random.choice(listSet)
        else:
            multiSample = set()
            i = 0
            while len(multiSample) != numOfSample:
                app = random.choice(listSet)
                multiSample.add(app)
                i = i + 1
                if i > 1000:
                    break
            return list(multiSample)

    def getinfo(self):
        print("subnet : ")
        for devId, dev in self.subnet.net.items():
            print("\t device id: " + str(dev.getId()))
        print("vulnerabilities : ")
        for vul in self.vulneralbilities:
            print("\t vulnerability id: " + str(vul.getId()))
        print("exploits : ")
        for exp in self.exploits:
            print("\t exploits id: " + str(exp.getId()))

# -------------------------
# Logger and Detector (unchanged logic, small robustness tweaks)
# -------------------------
class Logger:
    def __init__(self):
        self.logs = []

    def log(self, time_step, from_device, to_device, kind):
        self.logs.append({
            "time_step": time_step,
            "from_device": from_device,
            "to_device": to_device,
            "kind": kind
        })

    def get_logs(self):
        return self.logs

    def clear_logs(self):
        self.logs = []

class Detector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(n_estimators=2, max_samples=256, n_jobs=1)
        self.trained = False
        self.random_detection = False

    def train(self, logs=None):
        if logs is None or len(logs) == 0:
            # Initialize random detection logic
            self.random_detection = True
        else:
            X = [[log["from_device"], log["to_device"]] for log in logs]
            self.model.fit(X)
            self.trained = True
            self.random_detection = False

    def predict(self, from_device, to_device, return_score=False):
        if self.random_detection:
            result = random.choice(["A", "D"])
            return (result, None) if return_score else result

        if not self.trained:
            return ("D", None) if return_score else "D"

        point = np.array([from_device, to_device]).reshape(1, -1)
        prediction = self.model.predict(point)
        result = "A" if prediction == -1 else "D"

        if return_score:
            score = self.model.decision_function(point)
            return result, float(score[0])
        return result

    def batch_predict(self, log_points):
        if self.random_detection:
            return [random.choice(["A", "D"]) for _ in log_points]

        if not self.trained:
            return ["D" for _ in log_points]

        points = np.array(log_points)
        predictions = self.model.predict(points)
        return ["A" if pred == -1 else "D" for pred in predictions]

    def evaluate(self, test_logs):
        if not test_logs:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        X_test = [[log["from_device"], log["to_device"]] for log in test_logs]
        y_true = [1 if log["kind"] == "A" else 0 for log in test_logs]
        y_pred = [1 if self.model.predict([x])[0] == -1 else 0 for x in X_test]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
