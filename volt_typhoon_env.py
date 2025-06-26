
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
        self.intial_ratio_compromise = .4 #This across all devices in the network
        self.fast_scan = True
        self.γ = 0.99
        self._prev_potential = None
        self._prev_att_potential = None
        self.def_scale = 1
        self.scan_cnt = 0
        self.checkpoint_count=0
        self.defensive_cost=0
        self.clearning_cost=0
        self.revert_count=0
        self.compromised_devices_cnt=0
        self.edges_blocked=0
        self.edges_added=0
        self.snapshot_path = None #This holds a snapshot env generated via init_experiments. It allows you to control for topology.

        
        
        

    def get_num_action_types(self,mode = None ):
        if mode == 'defender':
            return 10 # Defender has 9 action types: 0-9
        elif mode == 'attacker':
            return 3  # Attacker has 2 action types: 0-1
        else:
            raise ValueError("Invalid mode: must be either 'defender' or 'attacker'")

    def get_device_indices(self):
        return [device.id for device in self.simulator.subnet.net.values()]

    def get_num_exploit_indices(self):
        # Assuming the number of exploit indices is based on the number of unique exploits
        return len(self.MaxExploits)  # Adjust based on actual exploit structure
        #return len(self.simulator.exploits)

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


    def step(self, action, agent_cnt = None):
        # ─── handle the “baseline” override ────────────────────────────────────
        if action is None:
            # we’re in one of the fixed‐policy modes, so build the proper tuple
            if self.mode == 'defender':
                # No‐Defense = do “pass” (action_type 7) on all defender‐eligible devices
                if self.base_line == "No Defense":
                    
                    action_type = 8
                    exploit_indices = [0]
                    device_indices = [
                        d.id
                        for d in self.simulator.subnet.net.values()
                        if not d.attacker_owned and not d.Not_yet_added
                    ]
                    app_index = 0
                # Preset is handled further down in your existing code,
                # so here we just do a no‐op as well
                elif self.base_line == "Preset":
                    action_type, exploit_indices, device_indices, app_index = 7, [0], [], 0
                else:
                    # fallback to pure “pass”
                    action_type, exploit_indices, device_indices, app_index = 7, [0], [], 0

            elif self.mode == 'attacker':
                # No‐Attack = do “pass” (action_type 3) on all attacker‐known devices
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

        # ─── now safe to unpack ────────────────────────────────────────────────
        action_type, exploit_indices, device_indices, app_index = action
        

        

        # Log the status of every device before the action
        for device_id, device in self.simulator.subnet.net.items():
            if self.debug:
                logging.debug(f"Before action: Device {device_id} - isCompromised: {device.isCompromised}, attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, mode: {self.mode}, always reachable: {device.reachable_by_attacker},To be Added: {device.Not_yet_added}")
            pass
        reward = 0
        done = False
        def_work = 0
        att_work = 0
        shaping_bonus = 0
        
        for _,device in self.simulator.subnet.net.items():
            if device.busy_time > 0:
                device.busy_time = device.busy_time - 1

                    
    
        if self.mode == 'defender':
            if self.base_line != "Nash" :
                action_type = 8  # no defence baseline, do the pass action (see next code block)- this overrides the policy provided action
            if self.debug:
                logging.debug(f"Defender action taken: {action_type} on device {device_indices} at step {self.step_num }")
        else:
            if self.debug:
                logging.debug(f"Attacker action taken: {action_type} on device {device_indices} at step {self.step_num }")
            pass

        if self.mode == 'defender':
            ckpt_path = 'checkpoint_DO' +str(self.its)+'.pkl'
            if action_type == 2:
                if self.tech == "DO":
                        self.checkpoint_count += 1
                        self.checkpoint_variables(ckpt_path, reward) #TO DO: take system down for some amount of time
                else:
                        self.checkpoint_count += 1
                        self.checkpoint_variables(ckpt_path, reward) #TO DO: take system down for some amount of time
                reward -= 0.5*len(device_indices)*self.def_scale
                self.defensive_cost += 0.5*len(device_indices)*self.def_scale
                for _,device in self.simulator.subnet.net.items():
                    if device.busy_time is not None and device.busy_time > 0:
                        device.busy_time = device.busy_time + 1 #stalls for a single step
            elif action_type == 3:  
                try:
                    self.revert_count += 1
                    with open(ckpt_path, 'rb') as f:
                        variables = pickle.load(f)
                    self.simulator = variables['simulator']
                    for _,device in self.simulator.subnet.net.items():
                        device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))
                        device.workload = None
                    reward -= len(device_indices)*self.def_scale
                except:
                    pass #need this block in case the checkpoint isn't there

                                                   


            elif action_type == 10: #train scanner
                device.busy_time = device.busy_time + 1 #stall for a time
                reward -= 1*self.def_scale
                data = self.simulator.logger.get_logs()
                self.simulator.detector.train(data)

   
        # Loop over each device index in the subset
        for device_index in device_indices:
            device = self.simulator.subnet.net[device_index]
            if device.Not_yet_added:
                pass
            else:
            
                if self.mode == 'defender':
                    
                    
                    if action_type == 1: #clean
                       
                        
                        if not device.attacker_owned:
                            reward += ( +0.3 if device.isCompromised else -0.01 )*self.def_scale
                            self.clearning_cost += ( +0.3 if device.isCompromised else 0.01 )*self.def_scale
                            self.defensive_cost += ( +0.3 if device.isCompromised else 0.01 )*self.def_scale


                            for eid in device.compromised_by:
                                # find that Exploit object
                                for exp in self.simulator.exploits:
                                    if exp.id == eid:
                                        exp.discovered = True
                                        break
                            device.compromised_by.clear()
                            device.isCompromised = False                            
                            device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))
                            if device.workload:
                                device.workload = None
                                



                    elif action_type == 4:  #upgrade
                        reward -= 1
                        if app_index != []:
                            if app_index < len(device.apps) :
                                app = list(device.apps.values())[app_index]
                                max_version = max(float(a.version) for d in self.simulator.subnet.net.values() for a in d.apps.values() if a.id == app.id)
                                app.version = str(max_version + 1)
                                device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))


                    elif action_type == 5:  # detect
                        self.scan_cnt += 1
                        logs = self.simulator.logger.get_logs()


                        
                        if not self.fast_scan:
                            for log in logs:
                                predicted_kind , device.anomaly_score = self.simulator.detector.predict(from_device=log["from_device"], to_device=log["to_device"], return_score=True)
                                reward -= 0.5*self.def_scale
                                self.defensive_cost -= 0.5*self.def_scale
                                
                                
                                if predicted_kind == "A":
                                    device = self.simulator.subnet.net[log["from_device"]]
                                    for eid in device.compromised_by:
                                        # find that Exploit object
                                        for exp in self.simulator.exploits:
                                            if exp.id == eid:
                                                exp.discovered = True
                                                break
                                    device.compromised_by.clear()
                                    device.isCompromised = False
                     
                                    device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))
                        else:
                            # ——————— DETECT ACTION ———————
                            logs = self.simulator.logger.get_logs()

                            # only look at the last 30 interactions
                            window = logs[-30:]

                            if window:
                                # batch‐predict them in one go (much faster than looping, and keeps your
                                # IsolationForest calls bounded at 30 points per detect)
                                log_points   = [(l["from_device"], l["to_device"]) for l in window]
                                predictions  = self.simulator.detector.batch_predict(log_points)  # ["A","D",...]

                                # count how many are anomalous
                                num_anoms    = sum(1 for p in predictions if p == "A")
                                majority     = len(predictions) // 2 + 1

                                
                                reward      -= 0.5*self.def_scale
                                self.defensive_cost  -= 0.5*self.def_scale

                                # if more than half are flagged
                                if num_anoms >= majority:
                                    # find which devices sent anomalous traffic
                                    flagged_senders = {
                                        l["from_device"]
                                        for l,p in zip(window, predictions)
                                        if p == "A"
                                    }
                                    for dev_id in flagged_senders:
                                        dev = self.simulator.subnet.net[dev_id]
                                        # “clean” them
                                        dev.isCompromised = False
                                        dev.busy_time     = np.ceil(
                                            np.random.triangular(1, self.default_mode, self.default_high, 1)
                                        )
                   
                    elif action_type == 6:
                        # Edge blocking: mark one random unblocked edge as blocked for this device.
                        reward -= 0.5*self.def_scale
                        self.defensive_cost -= 0.5*self.def_scale
                        g = self.simulator.subnet.graph
                        try:
                            # Find the vertex corresponding to the current device.
                            vertex = g.vs.find(name=device.id)
                            # Get all incident edges (both in and out).
                            incident_edges = g.incident(vertex.index, mode="all")
                            # Filter only those edges that are not already blocked.
                            unblocked_edges = [e for e in incident_edges if not g.es[e]["blocked"]]
                            if unblocked_edges:
                                # Randomly choose one edge to block.
                                edge_to_block = random.choice(unblocked_edges)
                                g.es[edge_to_block]["blocked"] = True
                                self.edges_blocked += 1
                                if self.debug:
                                    logging.debug(f"Defender blocked edge {edge_to_block} for device {device.id}.")
                            else:
                                if self.debug:
                                    logging.debug(f"Device {device.id} has no unblocked edges to block.")
                        except Exception as e:
                            if self.debug:
                                logging.error(f"Error blocking edge for device {device.id}: {e}")

                        
                    elif action_type == 7:
                        #node removal 
                        reward -= 0.5*self.def_scale
                        device.Not_yet_added = True
                        device.isCompromised = False
                        device.compromised_by.clear()
                        if device.workload:
                                device.workload = None

                    elif action_type == 8:
                        pass

                    elif action_type == 9:
                        reward -= 0.5*self.def_scale
                        # Edge unblocking: mark one random blocked edge as unblocked for this device.
                        g = self.simulator.subnet.graph
                        try:
                            # Find the vertex corresponding to the current device.
                            vertex = g.vs.find(name=device.id)
                            # Get all incident edges.
                            incident_edges = g.incident(vertex.index, mode="all")
                            # Filter only the blocked edges.
                            blocked_edges = [e for e in incident_edges if g.es[e]["blocked"]]
                            if blocked_edges:
                                # Randomly choose one edge to unblock.
                                edge_to_unblock = random.choice(blocked_edges)
                                g.es[edge_to_unblock]["blocked"] = False
                                self.edges_added += 1
                                if self.debug:
                                    logging.debug(f"Defender unblocked edge {edge_to_unblock} for device {device.id}.")
                            else:
                                if self.debug:
                                    logging.debug(f"Device {device.id} has no blocked edges to unblock.")
                        except Exception as e:
                            if self.debug:
                                logging.error(f"Error unblocking edge for device {device.id}: {e}")


                if self.mode == 'attacker':
                    
                    compromised_devices = [idx for idx, device in self.simulator.subnet.net.items() if device.isCompromised or device.attacker_owned]
                    
                    if self.debug:
                        logging.debug(f"Attacker action taken: {action_type} on device {device_indices} at step {self.step_num }")

                    if action_type == 1 and self.base_line != "No Attack":  
                        
                        if self.zero_day:
                            owned_indices = list(self.common_exploit_indices | self.private_exploit_indices)


                     
                        for raw in exploit_indices:
                           

                            if self.zero_day and raw not in owned_indices:
                                # penalize the agent for trying to use an exploit it doesn’t own
                                raw = random.choice(owned_indices) #give it one anyways
                                #shaping_bonus -=  .5 # if you’re adding to shaped_reward
                                
                                

                            # now raw is guaranteed to be an int
                            if 0 <= raw < len(self.simulator.exploits):
                                exploit = self.simulator.exploits[raw]
                            else:
                                # fallback only for weird cases
                                exploit = next((e for e in self.simulator.exploits if e.id == raw), None)

                            if exploit is None:
                                continue

                            if self.zero_day and raw not in owned_indices:
                                continue

                            # Get the compromised device as the starting point
                       
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
                                source_vertex = self.simulator.subnet.graph.vs.find(name=device_id)
                                for neighbor_id in neighbor_ids:
                                   
                                    # Check if the edge from device_id to neighbor_id is blocked
                                    neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                                   
                                    try:
                                        target_vertex = self.simulator.subnet.graph.vs.find(name=neighbor_id)
                                        edge_candidates = self.simulator.subnet.graph.es.select(_source=source_vertex.index, _target=target_vertex.index)

                                        # If no edge is found or the edge is blocked, skip this neighbor.
                                        if len(edge_candidates) == 0 or edge_candidates[0]["blocked"]:
                                            if self.debug:
                                                logging.debug(f"Edge from {device_id} to {neighbor_id} is blocked. Skipping communication.")
                                            continue
                                    except Exception as e:
                                        if self.debug:
                                            logging.error(f"Error checking edge between {device_id} and {neighbor_id}: {e}")
                                        continue
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
                                            for vul in app.vulnerabilities.values():  # Iterate over the values

                                                if exploit is not None:
                                                    if vul.id in exploit.target:  # Check if the vulnerability exists in the exploit's targets
                                                        #if self.zero_day:
                                                            #reward += getattr(vul, 'impact_score', None)
                                                        if self.debug:
                                                                            logging.debug(
                                                                                            f"Compromising device {neighbor_id} via vuln {vul.id} "
                                                                                            f"using exploit {exploit.id}"
                                                                                        )
                                                                            #print(f"Compromising device {neighbor_id} via vuln {vul.id} "
                                                                            #                f"using exploit {exploit.id}")
                                                        neighbor_device.isCompromised = True
                                                        #reward += vul.impact_score
                                                        break  # If compromised, break out of the loop
                                        if neighbor_device.isCompromised:
                                            break  # If compromised, break out of the loop

                    
                    if action_type == 2 and self.base_line != "No Attack":  # Probing action
                        # Choose a compromised or attacker-owned device to probe from

                        if compromised_devices:
                            probe_from_device_id = random.choice(compromised_devices)
                            probe_from_device = self.simulator.subnet.net[probe_from_device_id]
                            source_vertex = self.simulator.subnet.graph.vs.find(name=probe_from_device_id)
                            try:
                                neighbor_indices = self.simulator.subnet.graph.neighbors(source_vertex.index, mode="out")
                                neighbor_ids = [self.simulator.subnet.graph.vs[n]["name"] for n in neighbor_indices]
                            except Exception as e:
                                if self.debug:
                                    logging.error(f"Error retrieving neighbors for device {probe_from_device_id}: {e}")
                                neighbor_ids = []
                            
                            for neighbor_id in neighbor_ids:
                                source_vertex = self.simulator.subnet.graph.vs.find(name=probe_from_device_id)
                                try:
                                    target_vertex = self.simulator.subnet.graph.vs.find(name=neighbor_id)
                                    # Get the edge from the probing device to the neighbor device
                                    edge_candidates = self.simulator.subnet.graph.es.select(_source=source_vertex.index, 
                                                                                            _target=target_vertex.index)
                                  
                                    # If there is no edge or the edge is blocked, skip this neighbor.
                                    if len(edge_candidates) == 0 or edge_candidates[0]["blocked"]:
                                        if self.debug:
                                            logging.debug(f"Edge from {probe_from_device_id} to {neighbor_id} is blocked. Skipping probe.")
                                        continue
                                except Exception as e:
                                    if self.debug:
                                        logging.error(f"Error checking edge from {probe_from_device_id} to {neighbor_id}: {e}")
                                    continue

                                neighbor_device = self.simulator.subnet.net.get(neighbor_id)
                                if neighbor_device and not neighbor_device.Known_to_attacker:
                                    neighbor_device.Known_to_attacker = True
                                    reward += 0.1
                                    if self.debug:
                                        logging.debug(f"Probed device {neighbor_id} from {probe_from_device_id}.")
                                    break

        


                    if action_type == 3 or self.base_line == "No Attack":  # pass and do nothing action
                        pass
            
        
        if self.mode == 'defender':
            if self.base_line == "Preset":
                # Preset policy baseline
                if self.defender_step % 30 == 0:
                    if self.debug:
                        logging.debug(f"Preset defense clearing compromised devices at step {self.defender_step}.")
                    for _, device in self.simulator.subnet.net.items():
                        if not device.attacker_owned:
                            reward -= 1*self.def_scale
                            self.defensive_cost -= 1*self.def_scale
                            self.clearning_cost -= 1*self.def_scale
                            
                            device.isCompromised = False
                            device.workload = None
                            device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))
                            if self.debug:
                                logging.debug(f"Device {device.id} marked as uncompromised.")
    
                if self.defender_step % 7 == 0:
                    if self.debug:
                        logging.debug(f"Preset defense scanning for compromised devices at step {self.defender_step}.")
                    logs = self.simulator.logger.get_logs()
                    if logs:
                        log_points = [(log["from_device"], log["to_device"]) for log in logs]
                        if log_points:
                            predictions = self.simulator.detector.batch_predict(log_points)
                            for log, predicted_kind in zip(logs, predictions):
                                reward -= 0.5*self.def_scale
                                self.defensive_cost -= 0.5*self.def_scale
                                self.scan_cnt += 1
                                if predicted_kind == "A":
                                    device = self.simulator.subnet.net[log["from_device"]]
                                    device.busy_time = np.ceil(np.random.triangular(1, self.default_mode, self.default_high, 1))
                                    if not device.attacker_owned:
                                        device.isCompromised = False
                                        device.workload = None
                                        if self.debug:
                                            logging.debug(f"Device {log['from_device']} marked as uncompromised.")
    
                                    
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt: #only execute this for the last device 
         
            for _, device_work in self.simulator.subnet.net.items():
                    if device.busy_time == 0 and not device.Not_yet_added:
                        if device_work.workload is not None and device_work.workload.processing_time is not None and device_work.workload.processing_time > 0 and not device_work.workload.adversarial :
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0 :
                                device.workload = None
                                self.work_done += 1
                                
                        if device_work.workload is not None and device_work.workload.processing_time is not None and device_work.workload.processing_time > 0 and device_work.workload.adversarial :
                            device_work.workload.processing_time -= 1
                            if device_work.workload.processing_time == 0 :
                                device_work.workload = None
                                att_work += 1        
                                
                    def_work = self.work_done + def_work
            def_work =  self.work_scale * def_work    
                
            self.simulator.generate_workloads(numLoads=100, mode=1, high=3, wtype = 'client')
            self.simulator.generate_workloads(numLoads=10, mode=1, high=3, wtype = 'server')
            #self.simulator.generate_workloads(numLoads=1, mode=5, high=7 , adversarial = True)  
        self.compromised_devices_cnt += len([d for d in self.simulator.subnet.net.values() if d.isCompromised and not d.attacker_owned and not d.Not_yet_added])
      
        info = {
            'mode': self.mode,
            'step_count': self.step_num,
            'revert_count':self.revert_count,
            'checkpoint_count':self.checkpoint_count,
            'defensive_cost':self.defensive_cost,
            'clearning_cost':self.clearning_cost,
            'Scan_count':self.scan_cnt,
            'action_taken': action,
            'work_done': self.work_done,
            'Compromised_devices': self.compromised_devices_cnt,
            'Edges Blocked': self.edges_blocked,
            'Edges Added': self.edges_added,
            }

        
        if self.mode == 'defender':
            
            n_comp   = sum(
                1
                for d in self.simulator.subnet.net.values()
                if d.isCompromised and not d.attacker_owned and not d.Not_yet_added
            )

            # 6) raw vs. shaped
            raw_reward    = reward + def_work - n_comp*self.comp_scale 
            shaped_reward = raw_reward 

        else:
            # ——— simplified attacker reward: only count compromises ———
            reward = reward + self.comp_scale * (
                len([
                    d for d in self.simulator.subnet.net.values()
                    if d.isCompromised and not d.attacker_owned and not d.Not_yet_added
                ]) +
                10 * len([
                    d for d in self.simulator.subnet.net.values()
                    if d.isCompromised and d.device_type == "DomainController"
                ])
            )

            # ── ATTACKER POTENTIAL‐BASED SHAPING ────────────────────
            n_comp     = sum(
                1
                for d in self.simulator.subnet.net.values()
                if d.isCompromised and not d.attacker_owned and not d.Not_yet_added
            )
            max_nodes  = len(self.simulator.subnet.net)
            φ_a_new    = n_comp / max_nodes

            if not hasattr(self, '_prev_att_potential') or self._prev_att_potential is None:
                self._prev_att_potential = φ_a_new

            α_att       = 0.1
            shaped_inc  = self.γ * φ_a_new - self._prev_att_potential
            shaping_bonus = α_att * shaped_inc + shaping_bonus

            self._prev_att_potential = self.γ * φ_a_new

            raw_reward    = reward
            shaped_reward = raw_reward + shaping_bonus

                    
                                
            
            #print("Attacker total Immediate reward:"+str(reward)+"for baseline: "+str(self.base_line))

        self.state = self._get_state()
        if agent_cnt is None or len([device for device in self.simulator.subnet.net.values()]) == agent_cnt: #only execute this for the last device 
          self.step_num += 1
          if self.mode == "attacker":
              self.attacker_step += 1
          else:
              self.defender_step += 1
        
        done = self._check_done(action)

                # Log the status of every device before the action
        for device_id, device in self.simulator.subnet.net.items():
            if self.debug:
                logging.debug(f"After action: Device {device_id} - isCompromised: {device.isCompromised}, attacker_owned: {device.attacker_owned}, busy_time: {device.busy_time}, done: {done}, reward for step: {reward},To be Added: {device.Not_yet_added}")
            pass

        self.evolve_network()
       
        return self.state, raw_reward, shaped_reward, done, info, self.simulator.logger.get_logs()


    
    
    def initialize_environment(self):
        """
        Fully initialize the environment (called once at startup).  Modified to support
        “zero-day” mode: when self.zero_day=True, we generate k “common” exploits plus
        j “private” zero-day exploits.  Otherwise, we fall back to the existing behavior
        of generating two target exploits (VOLT_CVE_ID and VOLT_DC_CVE_ID).
        """

        # ─── 1) Reset simulator’s subnet and related state ─────────────────────
        self.simulator.resetAllSubnet()

        # ─── 2) Generate a small pool of “target” apps & vulnerabilities ───────
        #    (As in original code: 3 apps with 1 vuln each, targeted at VOLT_CVE_ID.)
        targetApps = self.simulator.generateApps(3, True, 1, vul_to=VOLT_CVE_ID)

        # ─── 3) ZERO‐DAY EXPLOIT SETUP ────────────────────────────────────────
        # Prepare lists to record which exploit‐IDs are “common” vs. “unknown‐pool” vs. “private”
        self.common_exploit_ids  = []
        self.unknown_pool_ids    = []
        self.private_exploit_ids = []
        

        if not isinstance(self.k_known, int) or self.k_known < 0:
            self.k_known = 1
        if not isinstance(self.j_private, int) or self.j_private < 0:
            self.j_private = 0
        
        
        minVulperExp = 1
        maxVulperExp = 1

        if self.zero_day:
            # We assume that `self.k_known` and `self.j_private` were set elsewhere.
            # Generate a total of (k_known) + (k_known + j_private) random exploits.
            total_to_generate =(self.k_known + self.j_private)
            self.simulator.generateExploits(
                total_to_generate,
                addVul=True,
                minVulperExp=minVulperExp,
                maxVulperExp=maxVulperExp,
                mode="random"
            )




            # Label the first k_known exploits as “common”
            for idx in range(self.k_known):
                exp_obj = self.simulator.exploits[idx]
                self.common_exploit_ids.append(exp_obj.id)
            self.common_exploit_indices = set(range(self.k_known))


            self.private_exploit_indices = {
                idx
                for idx, exp in enumerate(self.simulator.exploits)
                if exp.id in self.private_exploit_ids
                }

            # The next (k_known + j_private) exploits form the “unknown pool”
            start_idx = self.k_known
            end_idx   =  (self.k_known + self.j_private)
            for idx in range(start_idx, end_idx):
                exp_obj = self.simulator.exploits[idx]
                self.unknown_pool_ids.append(exp_obj.id)
            self.prior_pi = { z: 1/(self.j_private +self.k_known) for z in self.unknown_pool_ids}    

            # Now draw exactly j_private exploits (without replacement) from unknown_pool_ids
            # don’t force dtype=int — keep the UUID strings
            pool  = self.unknown_pool_ids            # e.g. ['3831AB03-…', 'ED3A999C-…', …]
            # build raw probs from your prior
            probs = [ self.prior_pi.get(z, 0.0) for z in pool ]
            total = sum(probs)

            if total <= 0.0:
                # fallback to uniform
                probs = [1.0/len(pool)] * len(pool)
            else:
                # normalize so they sum to 1
                probs = [p/total for p in probs]

            # now this will work
            chosen = self._rng.choice(pool,
                                    size=self.j_private,
                                    replace=False,
                                    p=probs)
            self.private_exploit_ids = list(chosen)
            self.private_exploit_indices = {
            idx
            for idx, exp in enumerate(self.simulator.exploits)
            if exp.id in self.private_exploit_ids
            }

        else:
            # ─── fallback (original behavior): generate exactly 2 “targeted” exploits ─────────
            self.simulator.generateExploits(
                1,
                addVul=True,
                minVulperExp=minVulperExp,
                maxVulperExp=maxVulperExp,
                mode="target",
                expID=VOLT_CVE_ID
            )
            self.simulator.generateExploits(
                1,
                addVul=True,
                minVulperExp=minVulperExp,
                maxVulperExp=maxVulperExp,
                mode="target",
                expID=VOLT_DC_CVE_ID
            )
            # Mark both as “common” (since no private zero‐days in this branch)
            for exp_obj in self.simulator.exploits:
                self.common_exploit_ids.append(exp_obj.id)
            self.unknown_pool_ids    = []
            self.private_exploit_ids = []

        # ─── 4) Attach those targetApps (from step 2) to all existing devices (none exist yet) ───
        # In the original code, apps were added after subnet generation; here we attach to each device below.

        # ─── 5) Compute attacker_action_space size (num_exploits + 3) ─────────────────────────
        # Note: we’ll overwrite this again after subnet/device‐creation, but initialize now:
        num_exploits = len(self.simulator.exploits)
        self.attacker_action_space = spaces.Discrete(num_exploits + 3)

        # ─── 6) Build the active subnet of exactly self.numOfDevice nodes ──────────────────────
        maxVulperApp = 1
        addApps      = 3

        # 6a) Generate every device (with random apps/vulns) up to Max_network_size
        self.simulator.generateSubnet(self.Max_network_size, addApps, 0, maxVulperApp + 1)

        # 6b) Build the specialized Volt‐Typhoon graph
        g = self.simulator.subnet.initializeVoltTyGraph(self.Max_network_size)

        # 6c) Now assign “targetApps” (from step 2) to each device
        for device in self.simulator.subnet.net.values():
            device.addApps(targetApps)

        # ─── 7) Enforce a forced active set (highest‐degree devices + any starting compromised) ─
        all_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: d.id)

        # Determine the top‐3 most connected devices in the graph
        most_connected_devices = sorted(
            all_devices,
            key=lambda d: len(g.neighbors(d.id)),
            reverse=True
        )[:3]
        forced_active_ids = {dev.id for dev in most_connected_devices}

        # Include any pre‐specified starting_compromised devices
        starting_compromised = self.starting_compromised or []
        forced_active_ids = forced_active_ids.union(set(starting_compromised))

        # Default initial active set: first self.numOfDevice devices by ID
        initial_active_ids = {dev.id for dev in all_devices[: self.numOfDevice]}

        # Union of those two sets is the final active set
        active_set_ids = initial_active_ids.union(forced_active_ids)
        for device in all_devices:
            if device.id in active_set_ids:
                device.Not_yet_added = False
            else:
                device.Not_yet_added = True

        # ─── 8) DomainController assignment and app/vuln injection ────────────────────────
        app_types     = ['VPN', 'RDP', 'ActiveDirectory', 'AdminPasswordService', 'FortiOS']
        app_versions  = ['1.0', '2.0', '3.0']
        fortios_version = '3.1'
        fortios_count   = 0

        # Mark the 3 most‐connected as DomainControllers with specific apps
        for dc_device in most_connected_devices:
            dc_device.addApps([
                App(id=f"ActiveDirectory_{dc_device.id}", type="ActiveDirectory", version="1.0"),
                App(id=f"Windows_Server_2019_{dc_device.id}", type="Windows_Server_2019", version="2019")
            ])
            dc_device.device_type = "DomainController"

            # Attach a vulnerability (VOLT_DC_CVE_ID) to that Windows_Server_2019 app
            for app in dc_device.getApps().values():
                if app.type == 'Windows_Server_2019' and app.version == "2019":
                    vulnerabilities = self.simulator.generateVul(
                        1, targetApp=app, mode="target", vulID=VOLT_DC_CVE_ID
                    )
                    for vul, prob in vulnerabilities:
                        if random.random() < prob:
                            app.addVulnerability(vul)

        # For all other devices, add a few random apps (including possible FortiOS) and
        # attach VOLT_CVE_ID vulnerabilities to FortiOS where applicable
        for device in self.simulator.subnet.net.values():
            if device not in most_connected_devices:
                apps = []
                for app_type in app_types:
                    if app_type == 'VPN':
                        device.wtype = 'server'
                    if app_type == 'ActiveDirectory':
                        device.wtype = 'server'
                        continue
                    app_version = random.choice(app_versions)
                    if app_type == 'FortiOS' and fortios_count < 5:
                        app_version = fortios_version
                        fortios_count += 1
                    apps.append(App(id=f"{app_type}_{device.id}", type=app_type, version=app_version))
                device.addApps(apps)

        # Now attach VOLT_CVE_ID to any FortiOS instance at the designated version
        for device in self.simulator.subnet.net.values():
            for app in device.getApps().values():
                if app.type == 'FortiOS' and app.version == fortios_version:
                    vulnerabilities = self.simulator.generateVul(
                        1, targetApp=app, mode="target", vulID=VOLT_CVE_ID
                    )
                    for vul, prob in vulnerabilities:
                        if random.random() < prob:
                            app.addVulnerability(vul)

        # ─── 9) Mark a set of devices as attacker‐owned/initially compromised ─────────────
        self.num_attacker_owned = 5
        starting_attacker_owned = random.sample(
            list(self.simulator.subnet.net.keys()),
            self.num_attacker_owned
        )
        self.starting_compromised = starting_attacker_owned

        for device_id in self.starting_compromised:
            dev = self.simulator.subnet.net[device_id]
            dev.isCompromised     = True
            dev.attacker_owned    = True
            dev.Known_to_attacker = True
            dev.Not_yet_added     = False

        # Connect attacker‐owned devices in the graph (so they’re “linked” for further attacks)
        attacker_owned_devices = [device_id for device_id in self.starting_compromised]
        self.simulator.subnet.connectAttackerOwnedDevices(g, attacker_owned_devices)

        # ─── 10) Generate initial workloads (clients + servers) ───────────────────────────
        self.step_num = 0
        self.simulator.generate_workloads(numLoads=100, mode=2, high=5, wtype='client')
        self.simulator.generate_workloads(numLoads=10,  mode=2, high=5, wtype='server')

        # Also randomly compromise some “known-to-attacker” nodes at reset, per original logic
        for device in self.simulator.subnet.net.values():
            if not device.Not_yet_added:
                if random.random() < self.intial_ratio_compromise:
                    device.isCompromised = True
                    device.Known_to_attacker = True

        # ─── 11) Finalize attacker_action_space now that exploits are fully generated ─────
        num_exploits = len(self.simulator.exploits)
        self.attacker_action_space = spaces.Discrete(num_exploits + 3)

        # Defender’s action space remains as before (Discrete(4))
        self.defender_action_space = spaces.Discrete(11)

        # ─── 12) Record initial shared state and reset potentials ────────────────────────
        self.state = self._get_state()
        self._prev_potential     = None
        self._prev_att_potential = None

        # Save the initial snapshot to disk (only once)
        if self.tech == "DQN":
            fname = f"initial_net_its{self.its}.pkl"
        else:
            fname = f"initial_net_DO_its{self.its}.pkl"
        if not os.path.exists(fname):
            with open(fname, 'wb') as f:
                pickle.dump({
                    'simulator': self.simulator,
                    'state':     self.state
                }, f)

        return self.state


    def randomize_compromise_and_ownership(self):
        """
        Randomly reassign which active, non-DC devices are 'attacker_owned' and which
        are 'compromised', preserving the original counts of each among those devices.
        DomainControllers are left untouched.
        """
        # 1) collect all active devices, split into DC vs non-DC
        active       = [d for d in self.simulator.subnet.net.values() if not d.Not_yet_added]
        non_dcs      = [d for d in active if d.device_type != "DomainController"]
        domain_ctrs  = [d for d in active if d.device_type == "DomainController"]

        # 2) record how many non-DCs are owned / compromised
        K_owned = sum(1 for d in non_dcs if d.attacker_owned)
        K_comp  = sum(1 for d in non_dcs if d.isCompromised)

        # 3) clear ONLY the non-DCs
        for d in non_dcs:
            d.attacker_owned     = False
            d.isCompromised      = False
            d.Known_to_attacker  = False

        # (leave domain controllers' flags alone)

        # 4) sample new attacker_owned among non-DCs
        new_owned = set(random.sample(non_dcs, K_owned))
        for d in new_owned:
            d.attacker_owned     = True
            d.isCompromised      = True
            d.Known_to_attacker  = True

        # 5) of the remaining non-DCs, sample the extra compromised
        remaining = [d for d in non_dcs if d not in new_owned]
        extra_comp = K_comp - K_owned
        if extra_comp > 0:
            for d in random.sample(remaining, extra_comp):
                d.isCompromised      = True
                d.Known_to_attacker  = True

        # domain controllers remain exactly as they were

        
    def reset(self,from_init = True):

        if from_init:
            # if you loaded env via pickle in your DO‐runner, set:
            #    env.snapshot_path = "/path/to/initial_env_DO_seedX.pkl"
            #
            # now we load that exact file
            if not hasattr(self, 'snapshot_path'):
                raise RuntimeError("you must set env.snapshot_path before calling reset(from_init=True)")
            with open(self.snapshot_path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, Volt_Typhoon_CyberDefenseEnv):
                # your init script pickled the whole env
                self.simulator = loaded.simulator
                self.state     = loaded.state
            else:
                # fallback to old dict format if you ever used that
                self.simulator = loaded['simulator']
                self.state     = loaded['state']
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
        
            self.simulator.generateSubnet(self.Max_network_size, addApps, 0, maxVulperApp + 1) 
            g = self.simulator.subnet.initializeVoltTyGraph(self.Max_network_size)

            #set some devices to not generate
            all_devices = sorted(self.simulator.subnet.net.values(), key=lambda d: d.id)

            # Determine the forced active sets.
            most_connected_devices = sorted(all_devices, key=lambda d: len(g.neighbors(d.id)), reverse=True)[:3]
            forced_active_ids = {device.id for device in most_connected_devices}

            # Use an empty list if self.starting_compromised is None
            starting_compromised = self.starting_compromised if self.starting_compromised is not None else []
            forced_active_ids = forced_active_ids.union(set(starting_compromised))

            # Select the initial active set (by your default criteria) from the sorted list.
            initial_active_ids = {device.id for device in all_devices[:self.numOfDevice]}

            # The final active set is the union of the default active devices with those we want to force active.
            active_set_ids = initial_active_ids.union(forced_active_ids)
            # Now update each device's Not_yet_added flag accordingly.
            for device in all_devices:
                if device.id in active_set_ids:
                    device.Not_yet_added = False
                else:
                    device.Not_yet_added = True
        
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
                        if app_type == 'VPN':
                            device.wtype = 'server'
                        if app_type == 'ActiveDirectory':
                            device.wtype = 'server'
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
        
            self.num_attacker_owned = 5 
            starting_attacker_owned = random.sample(list(self.simulator.subnet.net.keys()), self.num_attacker_owned)
            self.starting_compromised = starting_attacker_owned
        
            for device_id in self.starting_compromised:
                self.simulator.subnet.net[device_id].isCompromised = True
                self.simulator.subnet.net[device_id].attacker_owned = True
                self.simulator.subnet.net[device_id].Known_to_attacker = True
                self.simulator.subnet.net[device_id].Not_yet_added = False
                    
        
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
            self.simulator.generate_workloads(numLoads=100, mode=2, high=5, wtype = 'client')
            self.simulator.generate_workloads(numLoads=10, mode=2, high=5, wtype = 'server')
        
            self.state = self._get_state()
            self._prev_potential = None
        return self.state  
