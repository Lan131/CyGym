This repo is for CyGym: A Cybersecurity Simulation Environment for Game-Theoretic Analysis a gamesec2025 submission



files description, in order of abstraction 
CDSimulatorComponents.py - Contains logic for various simulatorobjects such as devices, workloads, exploits, etc
CDSimulator.py - This file contains the simulator class it builds from objects inside CDSimulatorComponents.py
CyberDefenseEnv.py - This contains the basic CyGym env object.
volt_typhoon_env.py - Overloads CyberDefenseEnv.py to create specific logic for the volt typhoon attack.
simulatorGraph.py and simulatorTest.py are diagnostic tools to verify the CDSimulator components. May be useful is developing that script.
utils.py - contains various utilities for do_agent.py
do_agent.py - contains logic for DOAR and agent behavior
volt_typhoon_do.py - plays and solves the game


Additionally CVE data is hosted here https://www.kaggle.com/datasets/mlanier/cygym-cve-csv the scripts source this as CVE.csv.




