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
# class CyberDefenseSimulator:

# App: Id, type, vulneralbility, version
# OS: Id, type, vulnerabilities, version
# Device: Id, OS, {app}, address, isCompromised, Workload
# Subnet: set of devices
# network: set of subnet
# exploit: vulnerability, OS, app
# workflow: source, test, size(# of steps)
# Vulnerability

# class CyberDefenseSimulator


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
        self.detector= Detector()
        self.vulneralbilities = set()
        #self.exploits = set()
        self.exploits = []
        self.defaultOS = OperatingSystem(0, "OS", 1.0)
        self.defaultversion = 1
        self.defaultApp = App(0, "game", 1.0)
        self.defaulVul = Vulnerability(0, "unknown", self.defaultApp)
        self.df = pd.read_csv('CVE.csv')  # Load CSV into DataFrame

    def log_communication(self, from_device, to_device, kind):
        self.logger.log(self.system_time, from_device, to_device, kind)
    
    def assign_workload(self, workload, wtype=None, adversarial=False):
        for device_id, device in self.subnet.net.items():
            # First try to assign the workload to the current device.
            if (device.OS.id == workload.OS.id and 
                device.OS.type == workload.OS.type and 
                device.wtype == workload.wtype and
                str(device.OS.version) == str(workload.version)):
                if not device.workload:
                    device.workload = workload
                    workload.assigned = True
                    #print(f"Workload {workload.id} assigned to device {device_id}")
                    return True
            else:
                # If not assigned, try to assign to a neighbor.
                try:
                    neighbor_ids = self.subnet.graph.get(device_id)  # returns a list of neighbor IDs
                except Exception:
                    vertex = self.subnet.graph.vs.find(name=device_id)
                    neighbor_indices = self.subnet.graph.neighbors(vertex.index, mode="out")
                    neighbor_ids = [self.subnet.graph.vs[neighbor].attributes()["name"] for neighbor in neighbor_indices]

                if neighbor_ids is None:
                    continue

                for neighbor_id in neighbor_ids:
                    # Check if the edge from device_id to neighbor_id is blocked.
                    try:
                        source_vertex = self.subnet.graph.vs.find(name=device_id)
                        target_vertex = self.subnet.graph.vs.find(name=neighbor_id)
                        edge_candidates = self.subnet.graph.es.select(_source=source_vertex.index, 
                                                                        _target=target_vertex.index)
                        # If no edge exists or the edge is blocked, skip this neighbor.
                        if len(edge_candidates) == 0 or edge_candidates[0]["blocked"]:
                            continue
                    except Exception as e:
                        # In case of any errors during edge checking, skip this neighbor.
                        continue

                    neighbor_device = self.subnet.net.get(neighbor_id)
                    self.log_communication(device_id, neighbor_id, 'D')
                    if neighbor_device:
                        # Check if neighbor's OS and version match the workload.
                        if neighbor_device.OS == workload.OS and str(neighbor_device.version) == str(workload.version):
                            if not neighbor_device.workload:
                                neighbor_device.workload = workload
                                workload.assigned = True
                                workload.adversarial = adversarial
                                #print(f"Workload {workload.id} assigned to neighbor device {neighbor_id} (passed from {device_id})")
                                return True

        #print(f"Workload {workload.id} could not be assigned to any device")
        return False


    def generate_workloads(self, numLoads, mode, high , wtype = 'client', adversarial = False):


        assigned_devices = set()  # Set to keep track of assigned devices

        for _ in range(numLoads):
            device_found = False  # Flag to check if a device is found
            for device_id, device in self.subnet.net.items():
                if device_id not in assigned_devices and not device.workload:
                    workload_id = str(uuid.uuid4())  # Generate a unique workload ID
                    selected_os, selected_version = device.OS, device.version

                    # Create a Workload object with the generated workload ID and selected OS/version
                    workload = Workload(workload_id, np.ceil(np.random.triangular(0, mode, high, 1)), selected_os, str(selected_version))

                    # Try to assign the workload to a device
                    assigned = self.assign_workload(workload, adversarial = adversarial , wtype = wtype)
                    if assigned:
                        assigned_devices.add(device_id)  # Mark the device as assigned
                        device_found = True
                        break  # Exit the loop once a device is found and assigned
                    
            if not device_found:
                pass



        
    def process_subnet_workloads(self,debug = False):
        # Iterate over all devices in the subnet
        completed_workloads = 0
        for _, device in self.subnet.net.items():
            if device.workload is not None and device.workload.processing_time is not None and device.workload.processing_time > 0:
                device.workload.processing_time = device.workload.processing_time - 1
                if device.workload.processing_time == 0:
                    device.workload == None
                    completed_workloads = completed_workloads + 1
        return completed_workloads

                

    def resetAllSubnet(self):
        """
        reset everything in the subnet, meaning DELETING all
        """
        self.subnet.net.clear()
        self.subnet.graph = ig.Graph()
        self.subnet.resetAllCompromisedSubnet()
        for _, device in self.subnet.net.items():
            device.set_random_success_prob()

        
    def resetByNumSubnet(self, resetNum):
        #randomDevices are a LIST of randomly selected device's ID
        randomDevices = self.randomSampleGenerator(self.subnet.net.keys(), resetNum)
        self.subnet.resetSomeCompromisedSubnet(randomDevices)

            

    def getSubnetSize(self):
        """
        Returns:
            int: returns the size of the subnet, equivalent to the number of devices
        """
        return len(self.subnet.net)

    def getNetworkSize(self):
        """
        Returns:
            int: returns the size of the network, equivalent to the number of subnet in the network
        """
        return len(self.network)

    def getVulneralbilitiesSize(self):
        """
        Returns:
            int: returns the size of the vulerability set
        """
        return len(self.vulneralbilities)

    def getExploitsSize(self):
        """
        Returns:
            int: returns the size of the Exploit set
        """
        return len(self.exploits)

    def generateApps(self, numOfApps, addVul=False, numOfVul=1, vul_to = None):
        """_summary_
        generates a list of apps with the specified number of apps to be added to the device.
        hence returns a list of apps.
        Args:
            numOfApps (int): number of apps to be generated
        """
        app_list = []
        if type(numOfApps) == int:
            for count in range(numOfApps):
                random_app = App(count, self.AppTypeGenerator(),
                                 self.randomNumberGenerator(1.0, 3.0))
                if addVul:
                    if vul_to is None:
                        appVul = self.generateVul(numOfVul, random_app)
                    else:
                        appVul = self.generateVul(numOfVul, random_app, mode = "target", vulID = vul_to)
                    
                    if(len(self.vulneralbilities)>=numOfVul):
                        for i in range(numOfVul):
                            random_app.addVulnerability(self.randomSampleGenerator(self.vulneralbilities))
                    else:
                        random_app.addVulnerability(self.defaulVul)
                app_list.append(random_app)
                self.apps.add(random_app)
        return app_list

    def generateDevice(self, numOfApps=1, minVulperApp=0, maxVulperApp=0):
        """_summary_
        generates a list of Devices with the specified number of app,
        number range of vulnerabilities to be added to the device.
        hence returns a list of Devices.
        Args:
            numOfApps (int, optional): number of apps per device. Defaults to 1.
            minVulperApp (int, optional): min vulnerability per app. Defaults to 0.
            maxVulperApp (int, optional): max vulnerability per app, can be none. Defaults to 0.
        Returns:
            List: return list of devices as specified from the input
        """
        if type(numOfApps) == int:
            AppsList = []
            # self.generateApps(numOfApps, True, int(self.randomNumberGenerator(minVulperApp,maxVulperApp)))
            for i in range(minVulperApp, maxVulperApp):
                AppsList.append(self.randomSampleGenerator(self.apps))
            currSize = self.getSubnetSize()
            newDevice = Device(currSize, self.defaultOS,self.defaultversion,  0)
            newDevice.addApps(AppsList)
            
            return newDevice
        else:
            print("not a valid input for generate Device")

    # add devices to subnet
    def generateSubnet(self, numOfDevice, addApps=None, minVulperApp=0, maxVulperApp=0):
        """_summary_
        add device to our subnet with the specified number of app(optional),
        number range of vulnerabilities to be added to the device.
        no returns

        Args:
            numOfDevice (int): specifies number of app to be added to the subnet
            addApps (List, optional): allows specified apps to be added. Defaults to None.
            minVulperApp (int, optional): min vulnerability per app. Defaults to 0.
            maxVulperApp (int, optional): max vulnerability per app, can be none. Defaults to 0.

        """

        if type(numOfDevice) == int:
            currSize = self.getSubnetSize()
            
            for count in range(numOfDevice):
               
                if addApps is None:
                    newDevice = self.generateDevice()
                else:
                    newDevice = self.generateDevice(
                        addApps, minVulperApp, maxVulperApp)
                self.subnet.addDevices(newDevice)
            # print(f'{numOfDevice} of devices added to subnet')
        else:
            print("not a valid input for generate subnet")

    #  def generateNetwork



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
            numOfApps = num_apps,
            addVul     = True,
            numOfVul   = vul_per_app
        )

        # 3) Shuffle & assign to devices
        for device in self.subnet.net.values():
            k = random.randint(min_apps_per_device, max_apps_per_device)
            chosen = random.sample(app_pool, k)
            device.addApps(chosen)


    def generateVul(self, numOfVul, targetApp=None, targetOS=None, mode="random", vulID=None):
        """Generate Vulnerability, either target App is given or target OS is given, cannot be both
            numOfVul specifies the number of vul given to the target OS or App
        Args:
            numOfVul (int): specifies number of vul generated to be added to the Vulnerability set
            targetApp (App, optional): target app of the vulnerability. Defaults to None.
            targetOS (OS, optional): target OS of the vulnerability. Defaults to None.
            mode (str, optional): mode of operation, 'random' or 'target'. Defaults to "random".
            vulID (int, optional): vulnerability ID from CSV when mode is 'target'. Defaults to None.
        """
        assert mode in ('random', 'target')
        vul_probabilities = []

        if mode == "random":
            sampled_data = self.df.sample(n=numOfVul)
            for index, row in sampled_data.iterrows():
                vulType = 'unknown'  # Or some logic to determine type
                target = targetApp if targetApp else (targetOS if targetOS else self.defaultApp)
                newVul = Vulnerability(row['matchCriteriaId'], vulType, target)
                self.vulneralbilities.add(newVul)
                newVul.exploitability_score = row['exploitabilityScore']
                newVul.impact_score = row['impactScore']
                probability = newVul.exploitability_score / 10.0
                vul_probabilities.append((newVul, probability))
                if targetApp or targetOS:
                    target.addVulnerability(newVul)
        elif mode == "target" and vulID is not None:
            row = self.df[self.df['matchCriteriaId'] == vulID]
            if not row.empty:
                row = row.iloc[0]
                vulType = 'unknown'  # Or some logic to determine type
                target = targetApp if targetApp else (targetOS if targetOS else self.defaultApp)
                newVul = Vulnerability(row['matchCriteriaId'], vulType, target)
                newVul.exploitability_score = row['exploitabilityScore']
                newVul.impact_score = row['impactScore']
                self.vulneralbilities.add(newVul)
                probability = newVul.exploitability_score / 10.0
                vul_probabilities.append((newVul, probability))
                if targetApp or targetOS:
                    target.addVulnerability(newVul)
        else:
            print("Invalid mode or missing vulID for target mode")

        #print("Vulernability prob is: "+str(vul_probabilities))
        return vul_probabilities


    def changeVulTarget(self):
        """
        after vul generated and app generated, this method can be used to reset the vul target from dummy target to a randomized app
        """
        if len(self.apps) == 0:
            print("not enough apps")
        if len(self.apps) < len(self.vulneralbilities):
            print("apps less than vul")
        for vul in self.vulneralbilities:
            vul.setTarget(self.randomSampleGenerator(self.appss))
    '''
    def generateExploits(self, numOfExploits, addVul=False, minVulperExp=0, maxVulperExp=0, mode="random", expID=None):
            """Generate specified input number of exploits that is added to the simulator's exploit subnet.
            Args:
                numOfExploits (int): Number of exploits to generate.
                addVul (bool): Whether to add vulnerabilities to the exploit.
                minVulperExp (int): Minimum number of vulnerabilities per exploit.
                maxVulperExp (int): Maximum number of vulnerabilities per exploit.
                mode (str): Mode of operation, 'random' or 'target'.
                expID (str): Exploit ID from CSV when mode is 'target'.
            """
            currSize = len(self.exploits)
            assert mode in ('random', 'target')
            if mode == "random":
                sampled_data = self.df.sample(n=numOfExploits)
                for index, row in sampled_data.iterrows():
                    ExpType = 'unknown'  # Or some logic to determine type
                    newExploit = Exploit(row['matchCriteriaId'], ExpType, self.defaultApp)
                    self.exploits.append(newExploit)  # Add to list
                    if addVul:
                        for i in range(int(self.randomNumberGenerator(minVulperExp, maxVulperExp))):
                            vul = self.randomSampleGenerator(self.vulneralbilities)
                            newExploit.setTargetVul(vul)
            elif mode == "target" and expID is not None:
                row = self.df[self.df['matchCriteriaId'] == expID]
                if not row.empty:
                    row = row.iloc[0]
                    ExpType = 'unknown'  # Or some logic to determine type
                    newExploit = Exploit(row['matchCriteriaId'], ExpType, self.defaultApp)
                    self.exploits.append(newExploit)  # Add to list
                    if addVul:
                        for i in range(int(self.randomNumberGenerator(minVulperExp, maxVulperExp))):
                            vul = self.randomSampleGenerator(self.vulneralbilities)
                            newExploit.setTargetVul(vul)
            else:
                print("Invalid mode or missing expID for target mode")

            print(f'{numOfExploits} of Exploits added to exploits')
    '''
    def generateExploits(
        self,
        numOfExploits,
        addVul=False,
        minVulperExp=0,
        maxVulperExp=0,
        mode="random",
        expID=None,
        discovered = False
    ):
        """
        Generate `numOfExploits` exploits, each targeting exactly one fresh vulnerability.
        Signature is unchanged so existing calls still work.

        Args:
            numOfExploits (int): how many exploits to generate
            addVul (bool): if True, also attach additional vulnerabilities from the pool
            minVulperExp, maxVulperExp: range for number of extra vulnerabilities to attach
            mode (str): "random" or "target"
            expID (str): if mode=="target", the specific CVE ID to target
        """
        assert mode in ("random", "target"), "mode must be 'random' or 'target'"

        # helper to attach extra existing vulnerabilities
        def _attach_extra(exploit):
            if not addVul or not self.vulneralbilities:
                return
            k = int(self.randomNumberGenerator(minVulperExp, maxVulperExp))
            for _ in range(k):
                extra_v = random.choice(list(self.vulneralbilities))
                exploit.setTargetVul(extra_v)

        if mode == "random":
            # 1) sample that many distinct CVE rows
            sampled = self.df.sample(n=numOfExploits, replace=False)
            for _, row in sampled.iterrows():
                # 2) build a fresh Vulnerability object
                new_vul = Vulnerability(
                    row["matchCriteriaId"],
                    "unknown",
                    self.defaultApp
                )
                new_vul.exploitability_score = row["exploitabilityScore"]
                new_vul.impact_score        = row["impactScore"]
                self.vulneralbilities.add(new_vul)

                # 3) create an Exploit that points *only* at that new vulnerability
                exp = Exploit(new_vul.id, "unknown", self.defaultApp)
                exp.setTargetVul(new_vul)

                # 4) optionally attach extra existing vulns
                _attach_extra(exp)

                self.exploits.append(exp)

        else:  # mode == "target"
            if expID is None:
                print("Invalid mode or missing expID for target mode")
                return
            # find that CVE row
            subset = self.df[self.df["matchCriteriaId"] == expID]
            if subset.empty:
                print(f"No CVE entry found for ID={expID}")
                return
            row = subset.iloc[0]
            for _ in range(numOfExploits):
                # 1) make a fresh Vulnerability with that exact ID
                new_vul = Vulnerability(
                    row["matchCriteriaId"],
                    "unknown",
                    self.defaultApp
                )
                new_vul.exploitability_score = row["exploitabilityScore"]
                new_vul.impact_score        = row["impactScore"]
                self.vulneralbilities.add(new_vul)

                # 2) and an Exploit targeting it
                exp = Exploit(new_vul.id, "unknown", self.defaultApp)
                exp.discovered = discovered
                exp.setTargetVul(new_vul)

                # 3) optionally attach extra existing vulns
                _attach_extra(exp)

                self.exploits.append(exp)

        print(f"{numOfExploits} exploits added (mode={mode}).")


    def attackSubnet(self, exploit):
        """_summary_
            attack the subnet with a SINGLE Exploit that has valid vulnerabilities
        Args:
            Exploit (Exploit): one Exploit
        """
        print(
            f'expected target is vulneralbility with id:{exploit.target.keys()}')
        self.subnet.attack(exploit, self.subnet.net)

    def randomNumberGenerator(self, a, b):
        """
        generates and return 1 decimal point random number in range a to b
        Args:
            a (int): lower bound
            b (int): higher bound
        """
        if (a == b):
            return a
        randomNum = random.randint(int(a*10), int(b*10))/10
        return randomNum

    def randomRangeGenerator(self, a=1.0, b=1.0):
        """randomly generates a lower bound and a higher bound from the input range
        Args:
            a (float, optional): input lowest boundary. Defaults to 1.0.
            b (float, optional): input upper boundary. Defaults to 1.0.

        Returns:
            range: returns 2 parameter, first the lower boundary, then the higher boundary
        """
        num1 = self.randomNumberGenerator(a, b)
        num2 = self.randomNumberGenerator(a, b)
        maxR = max(num1, num2)
        minR = min(num1, num2)
        return minR, maxR

    def AppTypeGenerator(self):
        """generate app type from the 5 most common types shown below. SUBJECT TO CHANGE
        Returns:
            AppType: returns one of the the App type
        """
        types = ["game", "lifestype", "social",
                 "entertainment", "productivity"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def VulTypeGenerator(self):
        """generate vulnerability type from the common types shown below. SUBJECT TO CHANGE
        Returns:
            VulType: returns one of the the Vul type
        """
        types = ["unknown", "misconfigurations", "outdated software",
                 "unauthorized access", "weak user credentials", "Unsecured APIs"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def ExpTypeGenerator(self):
        """generate exploit type from the 2 types shown below. SUBJECT TO CHANGE
        Returns:
            Exploit Type: returns one of the the exploit type
        """
        types = ["unknown", "known"]
        randomNum = random.randrange(0, len(types)-1)
        return types[randomNum]

    def randomSampleGenerator(self, sampleSet, numOfSample=1):
        """_summary_
        generates 1 sample basd on chosen set given by the argument
        Args:
            sampleSet (set): specifies which set to chose (from the private var in constructor)
            numOfSample (int): specifies the number of Sample to be returned. default to 1
        Returns:
            type of the set: 1 sample or a list depending on the parameter numOfSample
        """
        
        listSet = list(sampleSet)
        if numOfSample==1 or len(sampleSet)<numOfSample:
            return random.choice(listSet)
        else:
            multiSample = set()
            i=0
            while(len(multiSample)!=numOfSample):
                app = random.choice(listSet)
                multiSample.add(app)
                i=i+1
                if i>1000:
                    break
            return list(multiSample)

   

    def getinfo(self):
        """diplay info about the subnet, mainly for debug purpose
        """
        print("subnet : ")
        for devId, dev in self.subnet.net.items():
            print("\t device id: " + str(dev.getId()))
        print("vulnerabilities : ")
        for vul in self.vulneralbilities:
            print("\t vulnerability id: " + str(vul.getId()))
        print("exploits : ")
        for exp in self.exploits:
            print("\t exploits id: " + str(exp.getId()))


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
            #print("Using random detection due to insufficient training data.")
        else:
            X = [[log["from_device"], log["to_device"]] for log in logs]
            self.model.fit(X)
            self.trained = True
            self.random_detection = False

    def predict(self, from_device, to_device, return_score=False):
        # Random detection mode if there is no training data.
        if self.random_detection:
            result = random.choice(["A", "D"])
            return (result, None) if return_score else result

        # If not trained, default to non-anomalous ("D")
        if not self.trained:
            return ("D", None) if return_score else "D"

        point = np.array([from_device, to_device]).reshape(1, -1)
        prediction = self.model.predict(point)
        result = "A" if prediction == -1 else "D"

        if return_score:
            # Get anomaly score: lower scores indicate higher anomaly likelihood.
            score = self.model.decision_function(point)
            # score_samples returns an array; extract the first element.
            return result, score[0]
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
        X_test = [[log["from_device"], log["to_device"]] for log in test_logs]
        y_true = [1 if log["kind"] == "A" else 0 for log in test_logs]  # Assume "A" is the anomaly class
        y_pred = [1 if self.model.predict([x])[0] == -1 else 0 for x in X_test]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
