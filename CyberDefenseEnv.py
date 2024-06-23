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


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class CyberDefenseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.simulator = CyberDefenseSimulator()
        self.numOfDevice = 3
        
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
        

    def get_num_action_types(self):
        if self.mode == 'defender':
            return 8  # Defender has 8 action types: 0-7
        elif self.mode == 'attacker':
            return 3  # Attacker has  action types: 0-2
        else:
            raise ValueError("Invalid mode: must be either 'defender' or 'attacker'")


    def _get_attacker_state(self):
        defender_state = self._get_state()
        state_matrix = defender_state.reshape((self.numOfDevice, -1))

        for i, device in enumerate(self.simulator.subnet.net.values()):
            if not device.Known_to_attacker:
                state_matrix[i, :] = -1 #corresponds to unknown

        attacker_state_flat = state_matrix.flatten()
        return attacker_state_flat
      
    def _get_defender_state(self):
        # Get the state of all devices
        state = self._get_state()
    
        # Reshape the state into a matrix
        state_matrix = state.reshape((self.numOfDevice, -1))
    
        # Set the compromised flag to -1 for all devices (corresponds to unknown)
        state_matrix[:, 2] = -1
    
        # Flatten the defender state matrix
        defender_state_flat = state_matrix.flatten()
    
        # Return the flattened defender state
        return defender_state_flat

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return [seed]

        
    def checkpoint_variables(self, filename,r):
        # Define a dictionary of variables to save
        variables = {
            'simulator': self.simulator,
            'mode': self.mode,
            'reward': r
        }
        with open(filename, 'wb') as f:
            pickle.dump(variables, f)

    def step(self, action):
        action_type, exploit_indices, device_indices, app_index = action   # action is a tuple (action_type, exploit, [device_indices],app indicies)
        reward = 0
        done = False

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
                    reward -= .5
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
                        reward = variables['reward']
                    except:
                        pass
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

        self.simulator.generate_workloads(numLoads=100, mode=5, high=7)
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
            
            
        
        
        
        self.simulator.generate_workloads(numLoads = 2,mode = 2,high = 5)
    
        self.state =np.zeros((self.numOfDevice , 6))
        

        
        return self.state

    def generate_viz(self):
        fig, ax = plt.subplots()
        ig.plot(self.simulator.subnet.net, target=ax)
        return

    def _get_state(self):
        # State representation
        return np.array([
            [device.OS, device.version, float(device.isCompromised)]
            for _, device in self.simulator.subnet.net.items()]).flatten()

    def _check_done(self):
        if self.step_num >= 500:
            return True
        return False

    def sample_action(self):
        if self.mode == 'defender':
            action_type = self.defender_action_space.sample()
        elif self.mode == 'attacker':
            action_type = self.attacker_action_space.sample()

        device_indices = random.sample(list(self.simulator.subnet.net.keys()), k=random.randint(1, self.numOfDevice))  # Random subset
        exploit_indices = random.sample(list(self.simulator.exploits), k=random.randint(1, len(self.simulator.exploits)))
        app_indices = []
        


        return (action_type, exploit_indices, device_indices, app_indices)

'''
def calculate_max_compromise_proportion(simulator):
    """
    Calculate the maximum possible proportion of devices that can be compromised starting from any compromised or attacker-owned device.
    
    Args:
        simulator: The simulator object containing the network graph and exploits.
    
    Returns:
        float: The proportion of devices that can be compromised.
    """
    simulator=simulator.simulator
    compromised_devices = set(device_id for device_id, device in simulator.subnet.net.items() if device.isCompromised or device.attacker_owned)
    attacker_owned_cnt = len([device for device in simulator.subnet.net.values() if device.attacker_owned])
    total_devices = len([device for device in simulator.subnet.net.values() if not device.attacker_owned])
    exploits = simulator.exploits

    newly_compromised = True
    while newly_compromised:
        newly_compromised = False
        current_compromised = list(compromised_devices)

        for device_id in current_compromised:
            device = simulator.subnet.net[device_id]
            try:
                neighbor_ids = simulator.subnet.graph.get(device_id)
            except:
                vertex = simulator.subnet.graph.vs.find(name=device_id)
                neighbor_ids = simulator.subnet.graph.neighbors(vertex.index, mode="out")
                neighbor_ids = [simulator.subnet.graph.vs[neighbor].attributes()["name"] for neighbor in neighbor_ids]

            if not neighbor_ids:
                continue

            for neighbor_id in neighbor_ids:
                neighbor_device = simulator.subnet.net.get(neighbor_id)
                if neighbor_device and not neighbor_device.isCompromised and not neighbor_device.attacker_owned:
                    if device.device_type == "DomainController":
                        neighbor_device.isCompromised = True
                        compromised_devices.add(neighbor_id)
                        newly_compromised = True
                        break
                    if neighbor_device.reachable_by_attacker:
                        neighbor_device.isCompromised = True
                        compromised_devices.add(neighbor_id)
                        newly_compromised = True
                        break
                    if neighbor_device and not neighbor_device.isCompromised:
                        for exploit in exploits:
                            for app in neighbor_device.apps.values():
                                for vul in app.vulnerabilities.values():
                                    if vul.id in exploit.target:
                                        neighbor_device.isCompromised = True
                                        compromised_devices.add(neighbor_id)
                                        newly_compromised = True
                                        break
                        if neighbor_device.isCompromised:
                            break
    # Return the ratio of compromised devices excluding attacker-owned devices
    return len([device for device in compromised_devices if not simulator.subnet.net[device].attacker_owned]) / (total_devices)
'''

def calculate_max_compromise_proportion(simulator):
  return 1
