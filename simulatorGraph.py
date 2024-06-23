import unittest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from CDSimulator import * 
import igraph as ig

class CDSimulator():
    
    def __init__(self):
        """
        setUp initialize of different classes
        """
        simulator = CyberDefenseSimulator()
        

if __name__ == "__main__":
    simulator = CyberDefenseSimulator()
    
    targetApps = simulator.generateApps(30, True, 2)
    print(simulator.getVulneralbilitiesSize())
    
    # simulator.generateVul(20)
    # generate exploit, pick one exploit, check the compromised device
    minVulperExp=1
    maxVulperExp=3
    simulator.generateExploits(20, True, minVulperExp, maxVulperExp)
    
    print(f'exploit size is {simulator.getExploitsSize()}')
    ranExploit = simulator.randomSampleGenerator(simulator.exploits)
    print(ranExploit.getInfo())
    numOfCompromised1 = []
    
    # test how num of compromised change with time
    # different parameters to be set
    numOfCompromisedDev = []
    numOfIteration = int(input("num of iterations (suggested 50): "))
    resetNum = 300 #number of device to be resetted at each time
    resetStep = 5 #number of step before resetting some devices
    maxVulperApp = 4
    addApps = 6
    numOfDevice = 500

    simulator.generateSubnet(numOfDevice, addApps, 0, maxVulperApp+1)
    g = simulator.subnet.initializeRandomGraph(resetNum, 0.05)
    
    numLoads = int(input("Enter the number of workloads: "))
    
    simulator.generate_workloads(numLoads=100, mode=10, high=100)
    simulator.generateSubnet(numOfDevice, addApps, 0, maxVulperApp+1)
    
    # Initialize variables
    work_complete = 0
    work_completed_over_time = []
    
    # Process workloads over 100 time steps
    for _ in range(50):
        work_done_right_now = simulator.process_subnet_workloads()
        work_complete += work_done_right_now
        work_completed_over_time.append(work_complete)
    
    simulator.generate_workloads(numLoads=100, mode=10, high=100)
    # Process workloads over 100 time steps
    for _ in range(50):
        work_done_right_now = simulator.process_subnet_workloads()
        work_complete += work_done_right_now
        work_completed_over_time.append(work_complete)

    # Plot the graph
    fig, axs = plt.subplots(2)
    fig.suptitle('CyGym Diagnostics')
    
    # Plot for work completed over time
    axs[1].plot(range(100), work_completed_over_time)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Total Work Completed')
    axs[1].set_title('Total Work Completed Over Time')
    
    # Plot for num of compromised devices over time
    for timeStep in range(0, numOfIteration):
        ranExploit = simulator.randomSampleGenerator(simulator.exploits)
        simulator.attackSubnet(ranExploit)
        numOfCompromisedDev.append(simulator.subnet.getCompromisedNum())
        
        if timeStep % resetStep == 0:
            simulator.resetByNumSubnet(resetNum)
            print("num of compromised is now: " + str(simulator.subnet.getCompromisedNum()) +
                  "\nnum of device is now: " + str(simulator.getSubnetSize()))
    
    axs[0].set_title("Compromised Devices Over Time")
    axs[0].set_ylabel("Compromised Device Count")
    axs[0].set_xlabel("Iteration")
    axs[0].plot(range(numOfIteration), numOfCompromisedDev)
    axs[0].set_xticks(np.arange(min(range(numOfIteration)), max(range(numOfIteration))+1, 2.0))
    axs[0].set_xlim(0, numOfIteration)
    
    # Commented out the removed subplots
    # # Plot for max vulnerabilities per app and num of compromised devices
    # maxVulperApp = 10
    # addApps = 20
    # ranExploit = simulator.randomSampleGenerator(simulator.exploits)
    # simulator.subnet.resetAllCompromisedSubnet()
  
    # for i in range(1, maxVulperApp+1):
    #     simulator.generateSubnet(numOfDevice, addApps, 0, i)
        
    #     simulator.attackSubnet(ranExploit)
    #     numOfCompromised1.append(simulator.subnet.getCompromisedNum())
    #     simulator.resetAllSubnet()
    
    # axs[1].set_title("Max Vulnerabilities per App and Number of Compromised Device")
    # axs[1].set_ylabel("# of Compromised Device")
    # axs[1].set_xlabel("# Max Vul per App")
    # axs[1].plot(range(maxVulperApp), numOfCompromised1)
    # axs[1].set_xticks(np.arange(min(range(maxVulperApp)), max(range(maxVulperApp))+1, 1.0))
    # axs[1].set_xlim(1, maxVulperApp+1)
    
    # # Plot for max num apps per device and num of compromised devices
    # maxVulperApp = 5
    # addApps = 20
    # ranExploit = simulator.randomSampleGenerator(simulator.exploits)
    # numOfCompromised2 = []
    # for i in range(1, addApps+1):
    #     simulator.generateSubnet(numOfDevice, i, maxVulperApp-1, maxVulperApp)
    #     simulator.attackSubnet(ranExploit)
    #     currentCompromised = simulator.subnet.getCompromisedNum()
    #     numOfCompromised2.append(currentCompromised)
    #     simulator.resetAllSubnet()

    # axs[2].set_title("Max num App per device and Number of Compromised Device")
    # axs[2].set_ylabel("# of Compromised Device")
    # axs[2].set_xlabel("# Max Num App per Device")
    # axs[2].plot(range(1, addApps+1), numOfCompromised2)
    # axs[2].set_xticks(np.arange(min(range(addApps)), max(range(addApps))+1, 1.0))
    # axs[2].set_xlim(0, addApps)
    
    plt.tight_layout()
    plt.show()
