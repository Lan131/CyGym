
import os
import pickle
import argparse

from volt_typhoon_env import Volt_Typhoon_CyberDefenseEnv
from utils import set_seed

parser = argparse.ArgumentParser(
    description="Initialize Volt‑Typhoon DO environments for multiple instances"
)

parser.add_argument(
    '--num_of_device', type=int, default=10,
    help="Number of devices"
)
parser.add_argument(
    '--out_dir', type=str, default='.',
    help="Directory in which to save snapshots"
)
parser.add_argument(
    '--seed', type=int, default=1,
    help="Random seed"
)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


# List of instance IDs to generate
env_itrs = [1]
#env_itrs=[0]
set_seed(args.seed)
print("Starting file creation")

for its in env_itrs:
    # 1) create and configure a fresh env for this 'its'
    env = Volt_Typhoon_CyberDefenseEnv()
    env.its               = its
    #env.numOfDevice       = args.num_of_device
    env.numOfDevice       = 10
    env.Max_network_size  = 20
    env.base_line         = "Nash"
    env.tech              = "DO"
    env.mode              = "defender"
    env.j_private = 1
    env.k_known = 1
    env.zero_day = False

    # 2) initialize (build the network, pick initial compromised, etc.)
    env.initialize_environment()

    # 3) write out the full env object
    snap_path = os.path.join(
        args.out_dir,
        f"initial_net_DO_its{its}.pkl"
    )

    # Overwrite any existing snapshot to ensure it's the proper env
    with open(snap_path, 'wb') as f:
        pickle.dump(env, f)
    print(f"✅ Saved snapshot for its={its} to {snap_path}")
