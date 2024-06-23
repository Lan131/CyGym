
import gym
import matplotlib.pyplot as plt
import os
import imageio
import igraph as ig
from collections import deque
import numpy as np
from gym import spaces
import random
from CyberDefenseEnv import CyberDefenseEnv
from CDSimulatorComponents import App, Device, OperatingSystem, Workload, os_encoder
import logging
import uuid
import time
import pickle
VOLT_CVE_ID = "ED3A999C-9184-4D27-A62E-3D8A3F0D4F27"
VOLT_DC_CVE_ID = "0A5713AE-B7C5-4599-8E4F-9C235E73E5F6"

# Configure logging
#logging.basicConfig(level=logging.DEBUG, filename='cyberdefense_debug.log', filemode='w',
#                    format='%(asctime)s - %(levelname)s - %(message)s')

class Volt_Typhoon_CyberDefenseEnv(CyberDefenseEnv):

    def __init__(self):
        super().__init__()
        self.app_id_mapping = self._create_app_id_mapping()
        self.defaultOS = OperatingSystem(id=0, type="default", version="1.0")
        self.defaultversion = "1.0"
        self.default_mode = 1
        self.default_high = 2
        self.work_scale = .000001
        self.num_attacker_owned = 1
        self.base_line = "Nash"
        self.tech = "DQN"
        

    def get_num_action_types(self,mode = None ):
        if mode == 'defender':
            return 5  # Defender has 8 action types: 0-6
        elif mode == 'attacker':
            return 2  # Attacker has 4 action types: 0-1
        else:
            raise ValueError("Invalid mode: must be either 'defender' or 'attacker'")

    def get_device_indices(self):
        return [device.id for device in self.simulator.subnet.net.values()]

    def get_num_exploit_indices(self):
        # Assuming the number of exploit indices is based on the number of unique exploits
        return len(self.simulator.exploits)  # Adjust based on actual exploit structure

    def get_num_app_indices(self):
        # Assuming the number of app indices is based on the number of unique apps
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

    def _get_state(self):
        max_apps = max(len(device.apps) for device in self.simulator.subnet.net.values())
        state = []

        for device in self.simulator.subnet.net.values():
            os_info = [float(device.OS.id), float(device.OS.version)]
            apps_info = [float(self._encode_app_id(app.id)) for app in device.apps.values()] + [float(app.version) for app in device.apps.values()]
            apps_info.extend([0.0] * (2 * max_apps - len(apps_info)))
            busy_time = [float(device.busy_time)]
            workload_info = [float(device.workload.processing_time) if device.workload else -1.0]
            #graph_index = [float(device.id)]
            #pass adjaceny matrix maybe?

            device_state = os_info + apps_info + workload_info + busy_time  # Exclude is_compromised
            state.append(device_state)

        state_flat = np.array(state, dtype=np.float32).flatten()
        return state_flat

    def step(self, action, agent_cnt = None):
        action_type, exploit_indices, device_indices, app_index = action   # action is a tuple (action_type, exploit, [device_indices], app indices)
        

        # Log the status of every device before the action
        for device_id, device in self.simulator.subnet.net.items():
            #logging.debug(f"Before action: Device {device_id} - isCompromised: {device.isCompromised}, attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, mode: {self.mode}, always reachable: {device.reachable_by_attacker}")
            pass
        reward = 0
        done = False
        def_work = 0
        att_work = 0
        
        for _,device in self.simulator.subnet.net.items():
            if device.busy_time > 0:
                device.busy_time = device.busy_time - 1

                    
    
        if self.mode == 'defender':
            if self.base_line != "Nash":
                action_type = 4  # no defence baseline
            #logging.debug(f"Defender action taken: {action_type} on device {device_indices} at step {self.step_num }")
        else:
            #logging.debug(f"Attacker action taken: {action_type} on device {device_indices} at step {self.step_num }")
            pass

        # Loop over each device index in the subset
        for device_index in device_indices:
            device = self.simulator.subnet.net[device_index]
            
            if self.mode == 'defender':
                
                
                if action_type == 0:
                    if not device.attacker_owned:
                        device.isCompromised = False
                        device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))
                        if device.workload:
                              device.workload = None


                elif action_type == 1:  # checkpoint
                    reward -= 0.05
                    
                    if self.tech == "DO":
                        self.checkpoint_variables('checkpoint_DO.pkl', reward) #TO DO: take system down for some amount of time
                    else:
                        self.checkpoint_variables('checkpoint.pkl', reward) #TO DO: take system down for some amount of time
                elif action_type == 2:  # TO DO: figure out an availible set of upgrades
                    if app_index != []:
                        if app_index < len(device.apps) :
                            app = list(device.apps.values())[app_index]
                            max_version = max(float(a.version) for d in self.simulator.subnet.net.values() for a in d.apps.values() if a.id == app.id)
                            app.version = str(max_version + 1)
                            device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))

                elif action_type == 2:  # restore from checkpoint #TO DO: take system down for some amount of time
                    try:
                        if self.tech == "DO":
                            
                            with open('checkpoint_DO.pkl', 'rb') as f:
                                variables = pickle.load(f)
                            self.simulator = variables['simulator']
                            reward = variables['reward']
                        else:
                            with open('checkpoint.pkl', 'rb') as f:
                                variables = pickle.load(f)
                            self.simulator = variables['simulator']
                            reward = variables['reward']                            
                    except:
                        pass
                elif action_type == 3:  # detect
                    logs = self.simulator.logger.get_logs()
                    for log in logs:
                        predicted_kind = self.simulator.detector.predict(log["from_device"], log["to_device"])
                        reward -= 0.05
                        if predicted_kind == "A":
                            device = self.simulator.subnet.net[log["from_device"]]
                            device.isCompromised = False
                            device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))

                elif action_type == 4:
                    pass
                

            if self.mode == 'attacker':

                if action_type == 0:  # Example of an attack action
                    #if device.busy_time > 0:
                    #    pass
                    #else:
                      #device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))
                      reward -=.01
                      for exploit_idx in exploit_indices:
                          exploit = next((e for e in self.simulator.exploits if e.id == exploit_idx), None)


                          # Get the compromised device as the starting point
                          compromised_devices = [idx for idx, device in self.simulator.subnet.net.items() if device.isCompromised or device.attacker_owned]
                          for device_id in compromised_devices:
                              
                              device = self.simulator.subnet.net[device_id]
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
                                  reward -= 0.01
                                  if device.device_type == "DomainController":
                                      neighbor_device.isCompromised = True
                                      #logging.debug(f"Device {neighbor_id} compromised by Domain Controller {device_id}.")
                                      break

                                  if neighbor_device.reachable_by_attacker:
                                      neighbor_device.isCompromised = True
                                      break
                                  if neighbor_device and not neighbor_device.isCompromised and neighbor_device.Known_to_attacker:
                                      for app in neighbor_device.apps.values():
                                          for vul in app.vulnerabilities.values():  # Iterate over the values
                                              if exploit is not None:
                                                if vul.id in exploit.target:  # Check if the vulnerability exists in the exploit's targets
                                                    #logging.debug(f"Compromising device {neighbor_id} with vulnerability match.")
                                                    neighbor_device.isCompromised = True
                                                    break  # If compromised, break out of the loop
                                      if neighbor_device.isCompromised:
                                          break  # If compromised, break out of the loop
                              if neighbor_device.isCompromised:
                                  break  # If compromised, break out of the loop

                
                elif action_type == 1:  # Probing action
                    #if device.busy_time > 0:
                    #    pass
                    #else:
                    #device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))
                    # Choose a compromised or attacker-owned device to probe from
                    compromised_devices = [idx for idx, device in self.simulator.subnet.net.items() if device.isCompromised or device.attacker_owned]
                    if compromised_devices:
                        probe_from_device_id = random.choice(compromised_devices)
                        probe_from_device = self.simulator.subnet.net[probe_from_device_id]
                        
                        try:
                            neighbor_ids = self.simulator.subnet.graph.get(probe_from_device_id)
                        except:
                            vertex = self.simulator.subnet.graph.vs.find(name=probe_from_device_id)
                            neighbor_ids = self.simulator.subnet.graph.neighbors(vertex.index, mode="out")
                            neighbor_ids = [self.simulator.subnet.graph.vs[neighbor].attributes()["name"] for neighbor in neighbor_ids]
    
                        for neighbor_id in neighbor_ids:
                            neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                            if neighbor_device and not neighbor_device.Known_to_attacker:
                                neighbor_device.Known_to_attacker = True
                                reward -= 0.01
                                break
    


                elif action_type == 3:  # pass and do nothing action
                    pass
          
        
        if self.mode == 'defender':
            if self.base_line == "Preset":
                # Preset policy baseline
                if self.defender_step % 30 == 0:
                    #logging.debug(f"Preset defense clearing compromised devices at step {self.defender_step}.")
                    for _, device in self.simulator.subnet.net.items():
                        if not device.attacker_owned:
                            device.isCompromised = False
                            device.workload = None
                            device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))
                            #logging.debug(f"Device {device.id} marked as uncompromised.")
    
                if self.defender_step % 7 == 0:
                    #logging.debug(f"Preset defense scanning for compromised devices at step {self.defender_step}.")
                    logs = self.simulator.logger.get_logs()
                    if logs:
                        log_points = [(log["from_device"], log["to_device"]) for log in logs]
                        if log_points:
                            predictions = self.simulator.detector.batch_predict(log_points)
                            for log, predicted_kind in zip(logs, predictions):
                                reward -= 0.05
                                if predicted_kind == "A":
                                    device = self.simulator.subnet.net[log["from_device"]]
                                    device.busy_time = np.ceil(np.random.triangular(0, self.default_mode, self.default_high, 1))
                                    if not device.attacker_owned:
                                        device.isCompromised = False
                                        device.workload = None
                                        #logging.debug(f"Device {log['from_device']} marked as uncompromised.")
    
                                    
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt: #only execute this for the last device 
         
            for _, device_work in self.simulator.subnet.net.items():
                    if device.busy_time == 0:
                        if device_work.workload is not None and device_work.workload.processing_time is not None and device_work.workload.processing_time > 0 and not device_work.workload.adversarial :
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0:
                                device.workload = None
                                self.work_done += 1
                                
                        if device_work.workload is not None and device_work.workload.processing_time is not None and device_work.workload.processing_time > 0 and device_work.workload.adversarial :
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0:
                                device_work.workload = None
                                att_work += 1        
                                
                    def_work = self.work_done * self.work_scale
                
            self.simulator.generate_workloads(numLoads=1, mode=5, high=7)
            self.simulator.generate_workloads(numLoads=1, mode=5, high=7 , adversarial = True)  

        
        info = {
            'mode': self.mode,
            'step_count': self.step_num,
            'action_taken': action,
            'work_done': self.work_done,
            'Compromised_devices': len([device for device in self.simulator.subnet.net.values() if device.isCompromised and not device.attacker_owned])
        }

        if self.mode == 'defender':
            #print("Defender action Immediate reward:"+str(reward))
            reward -=  len([device for device in self.simulator.subnet.net.values() if device.isCompromised and not device.attacker_owned])
            reward = reward + def_work
            #print("Defender Immediate reward:"+str(reward))
        else:
            #print("Attacker action Immediate reward:"+str(reward))
            reward += att_work +  len([device for device in self.simulator.subnet.net.values() if device.isCompromised and not device.attacker_owned])
            #print("Attacker total Immediate reward:"+str(reward))
       
        self.state = self._get_state()
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt: #only execute this for the last device 
          self.step_num += 1
          if self.mode == "attacker":
              self.attacker_step += 1
          else:
              self.defender_step += 1
        
        done = self._check_done()

                # Log the status of every device before the action
        for device_id, device in self.simulator.subnet.net.items():
            #logging.debug(f"After action: Device {device_id} - isCompromised: {device.isCompromised}, attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, done: {done}, reward for step: {reward}")
            pass

        return self.state, reward, done, info, self.simulator.logger.get_logs()

    
    
    def initialize_environment(self):


        self.simulator.resetAllSubnet()

        targetApps = self.simulator.generateApps(3, True, 1, vul_to=VOLT_CVE_ID)

        minVulperExp = 1
        maxVulperExp = 1

        self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_CVE_ID)
        self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_DC_CVE_ID)

        for _, device in self.simulator.subnet.net.items():
            device.addApps(targetApps)

        num_exploits = 1
        self.attacker_action_space = spaces.Discrete(num_exploits + 3)

        maxVulperApp = 1
        addApps = 3

        self.simulator.generateSubnet(self.numOfDevice, addApps, 0, maxVulperApp + 1)
        g = self.simulator.subnet.initializeVoltTyGraph(self.numOfDevice)
        app_types = ['VPN', 'RDP', 'ActiveDirectory', 'AdminPasswordService', 'FortiOS']
        app_versions = ['1.0', '2.0', '3.0']
        fortios_version = '3.1'
        fortios_count = 0

        most_connected_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: len(g.neighbors(d.id)), reverse=True)[:3]
        for dc_device in most_connected_devices:
            dc_device.addApps([App(id=f"ActiveDirectory_{dc_device.id}", type="ActiveDirectory", version="1.0"),
                            App(id=f"Windows_Server_2019_{dc_device.id}", type="Windows_Server_2019", version="2019")])
            dc_device.device_type = "DomainController"

            for app in dc_device.getApps().values():
                if app.type == 'Windows_Server_2019' and app.version == "2019":
                    vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_DC_CVE_ID)
                    for vul, prob in vulnerabilities:
                        if random.random() < prob:
                            app.addVulnerability(vul)

        for device in self.simulator.subnet.net.values():
            if device not in most_connected_devices:
                apps = []
                for app_type in app_types:
                    if app_type == 'ActiveDirectory':
                        continue
                    app_version = random.choice(app_versions)
                    if app_type == 'FortiOS' and fortios_count < 5:
                        app_version = fortios_version
                        fortios_count += 1
                    apps.append(app)
                device.addApps(apps)

        for device in self.simulator.subnet.net.values():
            for app in device.getApps().values():
                if app.type == 'FortiOS' and app.version == fortios_version:
                    vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_CVE_ID)
                    for vul, prob in vulnerabilities:
                        if random.random() < prob:
                            app.addVulnerability(vul)

        self.num_attacker_owned = 1
        starting_attacker_owned = random.sample(list(self.simulator.subnet.net.keys()), self.num_attacker_owned)

        self.starting_compromised = starting_attacker_owned

        for device_id in self.starting_compromised:
            self.simulator.subnet.net[device_id].isCompromised = True
            self.simulator.subnet.net[device_id].attacker_owned = True
            self.simulator.subnet.net[device_id].Known_to_attacker = True

        attacker_owned_devices = [device_id for device_id in self.starting_compromised]
        self.simulator.subnet.connectAttackerOwnedDevices(g, attacker_owned_devices)

        self.step_num = 0
        self.simulator.generate_workloads(numLoads=1, mode=2, high=5)

        self.state = self._get_state()
        if self.tech == "DQN":
            with open("initial_net.pkl", 'wb') as f:
                pickle.dump({
                    'simulator': self.simulator,
                    'state': self.state
                }, f)
        else:
            with open("initial_net_DO.pkl", 'wb') as f:
                pickle.dump({
                    'simulator': self.simulator,
                    'state': self.state
                }, f) 
        return self.state
    
    
    def reset(self,from_init = True):

        if from_init:
            if self.tech == "DQN":
                with open(f"initial_net.pkl", 'rb') as f:
                    saved_data = pickle.load(f)
            else:
                with open(f"initial_net_DO.pkl", 'rb') as f:
                    saved_data = pickle.load(f)
            
            self.simulator = saved_data['simulator']
            self.state = saved_data['state']
        else:
            self.simulator.resetAllSubnet()
        
            targetApps = self.simulator.generateApps(3, True, 1, vul_to=VOLT_CVE_ID)
        
            minVulperExp = 1
            maxVulperExp = 1
        
            self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_CVE_ID)
            self.simulator.generateExploits(1, True, minVulperExp, maxVulperExp, mode="target", expID=VOLT_DC_CVE_ID)
        
            for _, device in self.simulator.subnet.net.items():
                device.addApps(targetApps)
        
            num_exploits = 1
            self.attacker_action_space = spaces.Discrete(num_exploits + 3)
        
            maxVulperApp = 1
            addApps = 3
        
            self.simulator.generateSubnet(self.numOfDevice, addApps, 0, maxVulperApp + 1)
            g = self.simulator.subnet.initializeVoltTyGraph(self.numOfDevice)
        
            app_types = ['VPN', 'RDP', 'ActiveDirectory', 'AdminPasswordService', 'FortiOS']
            app_versions = ['1.0', '2.0', '3.0']
            fortios_version = '3.1'
            fortios_count = 1
        
            most_connected_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: len(g.neighbors(d.id)), reverse=True)[:3]
            for dc_device in most_connected_devices:
                dc_device.addApps([App(id=f"ActiveDirectory_{dc_device.id}", type="ActiveDirectory", version="1.0"),
                                App(id=f"Windows_Server_2019_{dc_device.id}", type="Windows_Server_2019", version="2019")])
                dc_device.device_type = "DomainController"
        
                for app in dc_device.getApps().values():
                    if app.type == 'Windows_Server_2019' and app.version == "2019":
                        vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_DC_CVE_ID)
                        for vul, prob in vulnerabilities:
                            if random.random() < prob:
                                app.addVulnerability(vul)
        
            for device in self.simulator.subnet.net.values():
                if device not in most_connected_devices:
                    apps = []
                    for app_type in app_types:
                        if app_type == 'ActiveDirectory':
                            continue
                        app_version = random.choice(app_versions)
                        if app_type == 'FortiOS' and fortios_count < 2:
                            app_version = fortios_version
                            fortios_count += 1
                        apps.append(app)
                    device.addApps(apps)
        
            for device in self.simulator.subnet.net.values():
                for app in device.getApps().values():
                    if app.type == 'FortiOS' and app.version == fortios_version:
                        vulnerabilities = self.simulator.generateVul(1, targetApp=app, mode="target", vulID=VOLT_CVE_ID)
                        for vul, prob in vulnerabilities:
                            if random.random() < prob:
                                app.addVulnerability(vul)
        
            self.num_attacker_owned = 10
            starting_attacker_owned = random.sample(list(self.simulator.subnet.net.keys()), self.num_attacker_owned)
            self.starting_compromised = starting_attacker_owned
        
            for device_id in self.starting_compromised:
                self.simulator.subnet.net[device_id].isCompromised = True
                self.simulator.subnet.net[device_id].attacker_owned = True
                self.simulator.subnet.net[device_id].Known_to_attacker = True
        
            attacker_owned_devices = [device_id for device_id in self.starting_compromised]
            self.simulator.subnet.connectAttackerOwnedDevices(g, attacker_owned_devices)
        
            # Ensure at least one neighbor is reachable by attacker
            for device_id in self.starting_compromised:
                try:
                    neighbor_ids = self.simulator.subnet.graph.get(device_id)
                except:
                    vertex = self.simulator.subnet.graph.vs.find(name=device_id)
                    neighbor_ids = self.simulator.subnet.graph.neighbors(vertex.index, mode="out")
                    neighbor_ids = [self.simulator.subnet.graph.vs[neighbor].attributes()["name"] for neighbor in neighbor_ids]
                
                if neighbor_ids:
                    neighbor_device_id = random.choice(neighbor_ids)
                    self.simulator.subnet.net[neighbor_device_id].reachable_by_attacker = True
        
            self.step_num = 0
            self.simulator.generate_workloads(numLoads=1, mode=2, high=5)
        
            self.state = self._get_state()
        return self.state
  
