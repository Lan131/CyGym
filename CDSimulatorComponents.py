import copy
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import igraph as ig
import igraph as ig
import numpy as np


def os_encoder(os, os_types):
    encoded_os = [0] * len(os_types)
    if os.getType() in os_types:
        encoded_os[os_types.index(os.getType())] = 1
    return encoded_os + [float(os.getVersion())]

class Workload:
    def __init__(self, id, processing_time, OS, version):
        self.id = id
        self.processing_time = processing_time
        self.OS = OS  # operatingSystem
        self.version = version  # version
        self.assigned = False
        self.adversarial = False
        self.wtype = None  # 'client' or 'server'
        

class OperatingSystem:
    """Operating System (OS) contains ID, type, version, vulnerabilities"""
    def __init__(self, id, type, version):
        self.id = id
        self.type = type  # "known", "unknown"
        self.version = version
        self.vulnerabilities = {}


    def clone(self):
        """Create a clone of this operating system."""
        return OperatingSystem(
            id=self.id,
            type=self.type,
            version=self.version
        )

    def getId(self):
        """return OS ID
        Returns:
            int: id of the OS
        """
        return self.id

    def getType(self):
        """return OS type
        Returns:
            str: type of the OS
        """
        return self.type

    def getVersion(self):
        """return version of the OS
        Returns:
            str: version of the OS
        """
        return self.version
    
    def __eq__(self, other):
        if isinstance(other, OperatingSystem):
            return self.id == other.id and self.type == other.type and self.version == other.version
        return False

    def __repr__(self):
        return f"OperatingSystem(id={self.id}, type={self.type}, version={self.version})"

    def __hash__(self):
        return hash((self.id, self.type, self.version))

    def addVulnerability(self, vul):
        """add vulnerability to the OS, if OS already contains the vulnerbility, no vul will be added
        else, the vulnerability will be added to the OS
        Args:
            vul (Vulnerability): specifies vulnerabiltiy to be added
        """
        if isinstance(vul, Vulnerability):
            if vul in self.vulnerabilities:
                print("already contain the Vulnerability")
            else:
                self.vulnerabilities.update({vul.getId(): vul})
                # print("Vulnerability "+str(vul.getId())+" added successfully")
        else:
            print("not a Vulnerability")

    def removeVulnerability(self, vul):
        """remove vulnerability from the OS, if OS doesn’t contains the vulnerbility, no vul will be remove
        else, the vulnerability will be removed from the OS
        Args:
            vul (Vulnerability): specifies vulnerabiltiy to be removed
        """
        if isinstance(vul, Vulnerability):
            if vul.getId() in self.vulnerabilities.keys():
                self.vulnerabilities.pop(vul.getId())
                print("Vulnerability "+str(vul.getId())+" removed successfully")
            else:
                print("doesn’t contain the Vulnerability")
        else:
            print("not a Vulnerability")
    
    def getinfo(self):
        """print info about the OS object, mainly for DEBUGGING"""
        stringId = str(self.getId())
        print("OS id: " + stringId)
        print("OS type: " + self.getType())
        print("OS version: " + self.getVersion())
        print("OS Vulnerability: ")
        for vulId, vul in self.vulnerabilities.items():
            print("\tvul id: " + str(vul.getId()))



class App:
    """App containing Id, type, vulneralbility, version
    """
    def __init__(self, id, type, version):
        self.id = id
        self.type = type
        self.version = version
        self.vulnerabilities = {}

    def clone(self):
        """Create a clone of this app."""
        cloned_app = App(
            id=self.id,
            type=self.type,
            version=self.version
        )
        cloned_app.vulnerabilities = self.vulnerabilities.copy()
        return cloned_app

    def getId(self):
        """return App's unique ID
        Returns:
            int: id of the App
        """
        return self.id

    def getType(self):
        """return App type
        Returns:
            str: type of the App
        """
        return self.type



    def getVersion(self):
        """return version of the App

        Returns:
            str: version of the App
        """
        return self.version
    

    def addVulnerability(self, vul):
        """add vulnerability to the App, if App already contains the vulnerbility, no vul will be added
        else, the vulnerability will be added to the App
        Args:
            vul (Vulnerability): specifies vulnerabiltiy to be added
        """
        if isinstance(vul, Vulnerability):
            if vul in self.vulnerabilities:
                print("already contain the Vulnerability")
            else:
                self.vulnerabilities.update({vul.getId(): vul})
                # print("Vulnerability "+str(vul.getId())+" added successfully")
        else:
            print("not a Vulnerability")

    def removeVulnerability(self, vul):
        """remove vulnerability from the OS, if OS doesn't contains the vulnerbility, no vul will be remove
        else, the vulnerability will be removed from the OS

        Args:
            vul (Vulnerability): specifies vulnerabiltiy to be removed
        """
        if isinstance(vul, Vulnerability):
            if vul.getId() in self.vulnerabilities.keys():
                self.vulnerabilities.pop(vul.getId())
                print("Vulnerability "+str(vul.getId())+" removed successfully")
            else:
                print("doesn't contain the Vulnerability")
        else:
            print("not a Vulnerability")

    def getVulnerabilities(self):
        """get existing vulnerbility(s) from the App
        Returns:
            List: List of vulnerabilities returned
        """
        print("Vulternability of app id {" + self.id + "} includes:")
        for vul in self.vulnerabilities:
            print(vul)
        return self.vulnerabilities

    def getinfo(self):
        """print info about the App object, mainly for DEBUGGING
        """
        stringId = str(self.getId())
        print("\napp id: " + stringId)
        print("app type: " + self.getType())
        print("app version: " + str(self.getVersion()))
        print("app vulneralbiblity: ")
        for vulId, vul in self.vulnerabilities.items():
            print("\tvul id: " + str(vul.getId()))


class Device:
    """Device class with ID, OS, {app}, address, isCompromised"""
    def __init__(self, id, OS, address, version, device_type=None):
        self.id = id
        self.OS = OS  # operatingSystem
        self.version = version  # operatingSystem
        self.apps = {}
        self.address = address
        self.isCompromised = False
        self._private_compromise = False
        self.workload = None
        self.mask = False
        self.lie = False
        self.success_prob = 1
        self.attacker_owned = False  # some devices cannot be cleaned
        self.graph_index = None  # Store the graph index for the device
        self.device_type = device_type  # Type of device (e.g., router, server, workstation)
        self.Known_to_attacker = False
        self.reachable_by_attacker = False
        self.busy_time = 0
        self.agent = None #only for maddpg when network size is large (ie >20 devices or so)
        self.Not_yet_added = False
        self.anomaly_score = 0
        self.removed_before = 0 #this is a flag for use in the scanner as evolved network produce new network configurations
        self.wtype = 'client'
        self.compromised_by = set()


    def clone(self):
        """Create a clone of this device."""
        cloned_os = self.OS.clone()

        cloned_device = Device(
            id=self.id,
            OS=cloned_os,
            address=self.address,
            version=self.version,
            device_type=self.device_type
        )
        cloned_device.apps = {app_id: app.clone() for app_id, app in self.apps.items()}
        cloned_device.isCompromised = False
        cloned_device.workload = None
        cloned_device.mask = self.mask
        cloned_device.lie = self.lie
        cloned_device.success_prob = 1
        cloned_device.attacker_owned = False
        cloned_device.graph_index = None
        cloned_device.busy = False
        cloned_device.Known_to_attacker = True
        return cloned_device



    def set_random_success_prob(self, min_prob=0.0, max_prob=1.0):
        """
        Sets a random success probability for the device between min_prob and max_prob.
        """
        self.success_prob = random.uniform(min_prob, max_prob)
    def getId(self):
        """return Device's unique ID
        Returns:
            int: id of the Device
        """
        return self.id
    def set_workload(self, workload):
        """Set the workload for the device

        Args:
            workload (Workload): Workload object to set
        """
        if isinstance(workload, Workload):
            self.workload = workload
        else:
            print("Invalid workload")

    def getState(self, os_types):
        apps_info = [(app.type, app.version) for app in self.apps.values()]
        os_info = os_encoder(self.OS, os_types)
        workload_time = self.workload.get_processing_time() if hasattr(self, 'workload') else -1
        return {
            'id': self.id,
            'os_info': os_info,
            'apps_info': apps_info,
            'isCompromised': self.isCompromised,
            'workload_time': workload_time,
            'graph_index': self.id  # Assuming the ID is used as the graph index
        }

    def get_workload(self):
        """return workload of the Device

        Returns:
            Workload: workload of the Device
        """
        return self.workload

    def process_workload(self):
        """Process the current workload assigned to the device"""
        if self.workload:
            processing_time = self.workload.process_workload()
            print(f"Device {self.id} processed workload {self.workload.id} in {processing_time} units")
            self.workload = None
        else:
            print(f"Device {self.id} has no workload to process")

    def getAddress(self):
        """return Device's unique address
        Returns:
            int: addr of the Device
        """
        return self.address

        
    def getWorkload(self):
        """return workload of the App

        Returns:
            int: workload of the App
        """
        return self.workload


    def getApps(self):
        """return Device's apps
        Returns:
            dict: dictionary of Apps in the Device
        """
        return self.apps

    def addSingleApp(self, appName):
        """add single App to the device

        Args:
            appName (App): App to be added
        """
        if isinstance(appName, App):
            self.apps[appName.getId()] = appName
            # print("app "+str(appName.getId())+" added successfully")
        else:
            print("not an app")

    # Apps is a name
    def addApps(self, Apps):
        if isinstance(Apps, list):
            for app in Apps:
                self.addSingleApp(app)
        else:
            print("not a valid app list")

    def removeApp(self, appName):
        if isinstance(appName, App):
            if appName.getId() in self.apps.keys():
                self.apps.pop(appName.getId())
                print("App "+str(appName.getId())+" removed successfully")
            else:
                print("doesn't contain the App")
        else:
            print("not a App")

    def getIsCompromised(self):
        return self.isCompromised

    def attackSingleDevice(self, exploitTargetVul):
        """_summary_
        attack a single device, helper method for attack Device method (see below)
        Args:
            exploitTargetVul (dictionary): passes the diction of the exploit's target vulnerability

        Returns:
            boolean: true of false depends on whether if the attack was successful
        """
        for vulId, vul in exploitTargetVul.items():
            for appId, app in self.apps.items():
                if (vul in app.vulnerabilities.values()):
                    self.isCompromised = True
                    return True

                
                if vul in self.OS.vulnerabilities.values():
                    self.isCompromised = True
                    print("OS attacked!!!!!!!!!!!!!")
                    print(f'Device {self.getId()} attacked successful')
                    return True

        # print(f'Device {self.getId()} not attacked ')
        return False

    def attackDevice(self, exploit):
        if isinstance(exploit, Exploit) != True:
            print("not a valid exploit")
            return False
        # check if device vulnerable to exploit
        if self.getIsCompromised():
            # if not vulnerable, return false
            return False
            # if vulnerable:
        else:
            return self.attackSingleDevice(exploit.target)

    def resetIsCompromise(self):
        #return true if compromise state reseted to false, return false is ALREADY NOT COMPROMISED
        if self.getIsCompromised():
            self.isCompromised = False
            return True
        else:
            # if not compromised and not reseted, return false
            return False

    def getinfo(self):
        """print info about the Device object, mainly for DEBUGGING
        """
        stringId = str(self.getId())
        if self.device_type is None:
            self.device_type = "Not assigned"
        print("device id: " + stringId)
        print("device address: " + str(self.getAddress()))
        print("device OS type: " + self.OS.type)
        print("device OS version: " + str(self.OS.version))
        print("device type: " + str(self.device_type))
        print("device apps: ")
        for appID, app in self.apps.items():
            print("\t app id: " + str(app.getId()))

    def getState(self, os_types, app_types):
        apps_info = [0] * len(app_types)
        for app in self.apps.values():
            if app.type in app_types:
                apps_info[app_types.index(app.type)] = 1
        os_info = os_encoder(self.OS, os_types)
        workload_time = self.workload.get_processing_time() if hasattr(self, 'workload') else -1
        return os_info + apps_info + [float(self.isCompromised), workload_time, self.id]  # Assuming the ID is used as the graph index


class Vulnerability:
    def __init__(self, id, vulType, target, minR=None, maxR=None):
        self.id = id
        # vul <-> one app, or one os (target 1 thing), but a seq of version
        self.Vultarget = None
        self.setTarget(target)
        self.type = vulType  # known, unknwon
        self.versionMin = 1.0
        self.versionMax = 1.0
        self.setRange(minR, maxR)
        self.exploitability_score = None
        self.impact_score = None

    def getId(self):
        return self.id

    def getMax(self):
        return self.versionMax

    def getMin(self):
        return self.versionMin

    def setRange(self, minRange=None, maxRange=None):
        if minRange is not None:
            self.versionMin = minRange
        if maxRange is not None:
            self.versionMax = maxRange

    def setTarget(self, target):
        if self.Vultarget != None:
            print("already assigned vulnerability")
        if isinstance(target, OperatingSystem) or isinstance(target, App):
            self.Vultarget = target
        else:
            print("not a valid vul target")

    def getInfo(self):
        print(f'type is {str(self.type)}')
        print(f'range is min: {self.versionMin} and max: {self.versionMax}')


class Exploit:
    def __init__(self, id, expType, minR=None, maxR=None):
        self.id = id
        self.target = {}  # dict of target vulnerability
        self.type = expType  # known and unknown
        self.versionMin = 1.0
        self.versionMax = 1.0
        self.setRange(minR, maxR)
        self.discovered = True

    def getId(self):
        return self.id

    def getMax(self):
        return self.versionMax

    def getMin(self):
        return self.versionMin

    def setRange(self, minRange=None, maxRange=None):
        if minRange is not None:
            self.versionMin = minRange
        if maxRange is not None:
            self.versionMax = maxRange

    # assume targets is a single vul, list/set of vulnerabilities
    def setTargetVul(self, targetVul):
        if (type(targetVul) != list):
            if (isinstance(targetVul, Vulnerability)):
                targetVul = [targetVul]
            else:
                targetVul = list(targetVul)
        for vul in targetVul:
            if isinstance(vul, Vulnerability):
                if vul.getId() in self.target.keys():
                    continue
                else:
                    self.target.update({vul.getId(): vul})
                    # print("target vul "+str(vul.getId()) +" added successfully")
            else:
                print("not a valid vul target")

    def getInfo(self):
        print(f'type is {self.type}')
        print(f'range is min: {self.versionMin} and max: {self.versionMax}')
        print("targeted apps: ")
        for targetID, targetApp in self.target.items():
            print("\t target id: " + str(targetApp.getId())+" vul")


# Subnet: set of devices
class Subnet:
    def __init__(self):
        self.net = {}
        self.graph = None  # This will store the igraph graph object
        self.graph_dict = {}  # This is your dictionary representation
        self.numOfCompromised = 0

    def convert(self, lst):
        res_dct = {lst[i].getId(): lst[i] for i in range(0, len(lst))}
        # print(res_dct)
        return res_dct

    def initializeRandomGraph(self, num_devices, connection_probability=1):
        """
        Initialize a random directed graph for the subnet.

        Args:
        - num_devices: Number of devices in the network.
        - connection_probability: Probability of having a connection between any two devices.
        """
        g = ig.Graph(directed=True)
        device_ids = list(self.net.keys())  # Assuming self.net is already populated
        g.add_vertices(device_ids)

        for i in range(num_devices):
            
            for j in range(num_devices):

                if i != j and random.random() < connection_probability:
                    
                    g.add_edge(device_ids[i], device_ids[j])

        # Store the igraph object
        self.graph = g

        # Convert igraph Graph object to a dictionary representation
        self.graph_dict = {device_id: g.neighbors(device_id, mode="out") for device_id in device_ids}

        return g


    def initializeVoltTyGraph(self, num_devices, m=2):
        """
        Initialize a scale-free directed graph for the subnet using the Barabási-Albert model.

        Args:
        - num_devices: Number of devices in the network.
        - m: Number of edges to attach from a new node to existing nodes. Default is 2.
        """
        # Generate a directed Barabási-Albert graph
        g = ig.Graph.Barabasi(n=num_devices, m=m, directed=True)

        # Assign device types and OS types
        device_types = ['router', 'switch', 'server', 'workstation', 'firewall', 'VPN_gateway']
        os_types = {
            'router': ['Embedded Linux', 'Cisco IOS', 'Juniper Junos'],
            'switch': ['Embedded Linux', 'Cisco IOS', 'Juniper Junos'],
            'server': ['Windows Server', 'Linux (Ubuntu)', 'Linux (CentOS)', 'UNIX'],
            'workstation': ['Windows 10', 'Windows 11', 'macOS', 'Linux (Ubuntu)'],
            'firewall': ['Embedded Linux', 'Cisco IOS', 'Juniper Junos'],
            'VPN_gateway': ['Embedded Linux', 'Cisco IOS', 'Juniper Junos']
        }

        for i in range(num_devices):
            device_type = random.choice(device_types)
            os_choice = random.choice(os_types[device_type])
            address = f"192.168.0.{i}"
            version = str(random.choice([1.0, 2.0, 3.0])) if 'Linux' in os_choice else '1.0'
            os_object = OperatingSystem(id=i, type=os_choice, version=version)

            device = Device(
                id=i,
                OS=os_object,
                address=address,
                version=version
            )
            device.device_type = device_type
            self.net[i] = device

        device_ids = list(self.net.keys())
        g.vs["name"] = device_ids
        for edge in g.es:
            edge["blocked"] = False

        # Store the igraph object
        self.graph = g

        # Convert igraph Graph object to a dictionary representation
        self.graph_dict = {device_id: g.neighbors(g.vs.find(name=device_id).index, mode="out") for device_id in device_ids}


        return g


    def get_neighbors(self, device_id):
        if self.graph is not None:
            try:
                vertex = self.graph.vs.find(name=device_id)
                neighbors = self.graph.neighbors(vertex.index, mode="out")
                return [self.graph.vs[neighbor]["name"] for neighbor in neighbors]
            except ValueError:
                print(f"Device ID {device_id} not found in graph")
                return []
        else:
            return []



    def connectAttackerOwnedDevices(self, g, attacker_owned_devices, attackable_devices_per_attacker=1):
        """
        Ensure attacker-owned devices are connected to all other devices in the graph and 
        at least one connected device is attackable.

        Args:
        - g: The igraph Graph object.
        - attacker_owned_devices: List of attacker-owned device IDs.
        - attackable_devices_per_attacker: Number of attackable devices per attacker-owned device.
        """
        device_ids = g.vs["name"]
        for attacker_device_id in attacker_owned_devices:
            if attacker_device_id in device_ids:
                attacker_neighbors = set(device_ids) - {attacker_device_id}
                for neighbor in attacker_neighbors:
                    g.add_edges([(attacker_device_id, neighbor)])

                # Ensure each attacker-owned device has at least one attackable neighbor
                attackable_neighbors = random.sample(sorted(attacker_neighbors), min(attackable_devices_per_attacker, len(attacker_neighbors)))
                for neighbor_id in attackable_neighbors:
                    neighbor_device = self.net[neighbor_id]
                    neighbor_device.reachable_by_attacker = True
                    #print(f"Device {neighbor_id} marked as reachable by attacker.")

                    # Ensure the neighbor device is connected in the graph
                    g.delete_edges(g.incident(neighbor_id, mode="out"))
                    g.add_edges([(attacker_device_id, neighbor_id)])

    def connectNonAttackerDevice(self, g, new_device_id, m=2):
        """
        Connect a newly active non-attacker-owned device to m existing active devices
        using a Barabási preferential attachment mechanism.
        
        Args:
            g: The igraph Graph object.
            new_device_id: The device id of the new active device.
            m: Number of connections to form.
        """
        # Gather candidate nodes: active nodes that are not the new device.
        candidate_nodes = [v["name"] for v in g.vs if v.get("active", False) and v["name"] != new_device_id]
        if not candidate_nodes:
            return
        # Calculate degrees for candidate nodes.
        degrees = np.array([g.degree(v_name) for v_name in candidate_nodes])
        total_degree = degrees.sum()
        if total_degree == 0:
            # If none have any connections yet, choose uniformly.
            probabilities = np.ones(len(candidate_nodes)) / len(candidate_nodes)
        else:
            probabilities = degrees / total_degree

        # Ensure we don't try to connect to more nodes than available.
        m = min(m, len(candidate_nodes))
        selected_nodes = np.random.choice(candidate_nodes, size=m, replace=False, p=probabilities)
        for target in selected_nodes:
            g.add_edges([(new_device_id, target)])


    def addDevices(self, device):
        if type(device) == list:
            for dev in device:
                self.net.update({dev.getId(): dev})
                # print("device "+str(dev.getId())+" added successfully")
        elif isinstance(device, Device):
            self.net.update({device.getId(): device})
            # print("device "+str(device.getId())+" added successfully")
        else:
            print("not a device")

    # targetDevices is a dict
    def attack(self, exploit, targetDevices):

        if type(targetDevices) != dict:
            print("target Device is not a dict")
        if isinstance(exploit, Exploit):
            # exploit.target
            print("attacking: ")
            for devId, device in targetDevices.items():
                # exploit.target.items():
                if (devId in self.net.keys()):
                    success = self.net.get(devId).attackDevice(exploit)
                    if success:
                        self.numOfCompromised += 1

        else:
            print("not an exploit, invalid parameter")

    def getDeviceNumber(self):
        return self.len(self.net)

    def getCompromisedNum(self):
        return self.numOfCompromised

        
    def resetAllCompromisedSubnet(self):
        for deviceId, device in self.net.items():
            device.resetIsCompromise()
        self.numOfCompromised = 0
    
    def resetSomeCompromisedSubnet(self, DevIDList):
        for devID in DevIDList:
            state = self.net.get(devID).resetIsCompromise()
            if state:
                self.numOfCompromised-=1
            

    def getinfo(self):
        print("num of compromised: " + str(self.getCompromisedNum()))
        print("subnet devices:")
        for deviceid, device in self.net.items():
            print("\t device id: " + str(deviceid))


