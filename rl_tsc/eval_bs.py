import torch
import os
import sys
import numpy as np
import time
from collections import deque
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Original TSC utilities
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS
# Dixon-log Alpha-Fairness allocator
from utils.allocator import TelcoAllocator

# TraCI Fix for SUMO connectivity
try:
    import traci
    sys.modules["libsumo"] = traci
except ImportError:
    pass

path_convert = get_abs_path(__file__)

def render_terminal_full_junction_dashboard(step, utility, total_vehs, demand_dict, allocation_dict, c_max):
    """
    Generates a real-time visual dashboard for all 12 lanes of the junction.
    This provides a complete view of the 5G cell resource distribution.
    """
    print(f"\n{'='*95}")
    print(f" STEP: {step:03d} | TOTAL CAPACITY: {c_max} Mbps | NETWORK UTILITY (U): {utility:.4f}")
    print(f" TOTAL VEHICLES: {int(total_vehs)} | MONITORING: Full 12-Lane Junction")
    print(f"{'-'*95}")
    print(f" {'LANE ID (SUMO)':<25} {'Demand':>8} {'Allocation (Mbps)':>22} {'Visual Bar':<20}")
    print(f"{'-'*95}")

    bar_length = 20
    # Sort lanes by ID to group them by Edge (Street)
    sorted_lanes = sorted(demand_dict.keys())

    for i, lane in enumerate(sorted_lanes):
        d = demand_dict.get(lane, 0)
        c = allocation_dict.get(lane, 0)
        
        filled = int((c / c_max) * bar_length) if c_max > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Add a visual separator every 3 lanes (since each street in your XML has 3 lanes)
        if i > 0 and i % 3 == 0:
            print(f"{'·'*95}")
            
        print(f" {lane:<25} {int(d):>8} {c:>14.2f} Mbps   [{bar}]")
    
    print(f"{'='*95}")

# --- SETUP CONFIGURATION ---
scenario_key = "Hongkong_YMT"
config = SCENARIO_CONFIGS.get(scenario_key)
SCENARIO_NAME = config["SCENARIO_NAME"]
TLS_ID = config["JUNCTION_NAME"]

if __name__ == '__main__':
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{config['NETFILE']}.sumocfg")
    
    params = {
        'tls_id': TLS_ID,
        'num_seconds': 500,
        'number_phases': config["PHASE_NUMBER"],
        'sumo_cfg': sumo_cfg,
        'use_gui': True, 
        'log_file': path_convert(f"./eval_full_junction.log"),
    }

    # 1. Initialize Normalized TSC Environment
    env = DummyVecEnv([make_env(env_index='0', **params)])
    env = VecNormalize.load(
        load_path=path_convert(f'./results/{SCENARIO_NAME}/models/last_vec_normalize.pkl'),
        venv=env
    )
    env.training = False
    env.norm_reward = False

    # 2. Load Pre-trained TSC Agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPO.load(path_convert(f'./results/{SCENARIO_NAME}/models/last_rl_model.zip'), env=env, device=device)

    # 3. Initialize Allocator Settings
    telco_bs = TelcoAllocator(c_max=50.0, alpha=1.0)
    
    obs = env.reset()
    step_counter = 0

    print("\n[START] Full 12-Lane Junction Slicing Monitor...")
    
    try:
        while True:
            # A. TSC Brain Prediction
            action, _ = model.predict(obs, deterministic=True)

            # B. NSC Decision: Extract all 12 unique lanes
            all_controlled = traci.trafficlight.getControlledLanes(TLS_ID)
            # We remove duplicates while preserving order, taking all available lanes (should be 12)
            junction_lanes = list(dict.fromkeys(all_controlled))

            demand_snapshot = {}
            for lane_id in junction_lanes:
                demand_snapshot[lane_id] = float(traci.lane.getLastStepVehicleNumber(lane_id))

            # C. Multi-Lane Proportional Fairness Allocation
            total_d = sum(demand_snapshot.values())
            allocation_dist = {}
            current_utility = 0.0

            for lane_id, d in demand_snapshot.items():
                if total_d > 0:
                    c_assigned = (d / total_d) * telco_bs.c_max
                    allocation_dist[lane_id] = c_assigned
                    # Manual Utility Calculation to verify PF performance
                    if d >= 1.0:
                        current_utility += d * np.log1p(c_assigned / d)
                else:
                    # If intersection is empty, distribute capacity equally
                    allocation_dist[lane_id] = telco_bs.c_max / len(junction_lanes)

            total_vehs = sum(demand_snapshot.values())

            # D. Simulation Step
            obs, rewards, dones, infos = env.step(action)

            # E. Update Dashboard
            if step_counter % 5 == 0:
                render_terminal_full_junction_dashboard(
                    step_counter, current_utility, total_vehs, 
                    demand_snapshot, allocation_dist, 
                    telco_bs.c_max
                )
                time.sleep(0.01)

            step_counter += 1
            if dones[0] or step_counter >= 500:
                break

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    finally:
        env.close()