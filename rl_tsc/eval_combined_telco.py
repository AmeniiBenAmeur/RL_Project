import torch
import os
import sys
import numpy as np
from loguru import logger
from collections import deque
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Original TSC utilities from authors
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS
# Dixon-log Alpha-Fairness allocator
from utils.allocator import TelcoAllocator

# Fix for TraCI on Windows environments
try:
    import traci
    sys.modules["libsumo"] = traci
except ImportError:
    pass

path_convert = get_abs_path(__file__)
logger.remove()

# Configuration based on the chosen scenario
scenario_key = "Hongkong_YMT"
config = SCENARIO_CONFIGS.get(scenario_key)
SCENARIO_NAME  = config["SCENARIO_NAME"]
TLS_ID         = config["JUNCTION_NAME"]   # Used by TraCI to read lane vehicle counts

if __name__ == '__main__':
    # Define simulation paths
    sumo_cfg = path_convert(f"../sim_envs/{SCENARIO_NAME}/{config['NETFILE']}.sumocfg")
    tls_add = [
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/e2.add.xml'),
        path_convert(f'../sim_envs/{SCENARIO_NAME}/add/tls_programs.add.xml'),
    ]

    # Simulation parameters
    params = {
        'tls_id':        TLS_ID,
        'num_seconds':   500,
        'number_phases': config["PHASE_NUMBER"],
        'sumo_cfg':      sumo_cfg,
        'tls_state_add': tls_add,
        'use_gui':       True,
        'log_file':      path_convert(f"./eval_combined.log"),
    }

    # Initialize Environment
    env = DummyVecEnv([make_env(env_index='0', **params)])

    # Load normalization statistics
    env = VecNormalize.load(
        load_path=path_convert(f'./results/{SCENARIO_NAME}/models/last_vec_normalize.pkl'),
        venv=env
    )
    env.training    = False
    env.norm_reward = False

    # Load the Pre-trained PPO Agent (Traffic Light)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPO.load(
        path_convert(f'./results/{SCENARIO_NAME}/models/last_rl_model.zip'),
        env=env,
        device=device
    )

    # Initialize BS Allocator
    # Utility: U = sum_i [ d_i * log(1 + c_i / d_i) ]
    # KKT solution: c_i = d_i * C_max / sum(d_j)
    telco_bs = TelcoAllocator(c_max=50.0, alpha=1.0)

    # --- MOVING AVERAGE BUFFER (5 frames) ---
    # Used as fallback demand source if TraCI read fails.
    obs_buffer = deque(maxlen=5)

    obs = env.reset()
    step_counter = 0

    print("\n" + "="*60)
    print(f"BS Configured   : Capacity={telco_bs.c_max} Mbps, Alpha={telco_bs.alpha}")
    print("UTILITY FUNCTION: U = sum_i [ d_i * log(1 + c_i / d_i) ]")
    print("ALLOCATION RULE : c_i = d_i * C_max / sum(d_j)  [KKT optimal]")
    print("DEMAND SOURCE   : TraCI raw vehicle counts (integer per lane)")
    print("FALLBACK        : 5-frame moving average on normalised obs")
    print("="*60)

    try:
        while True:
            # 1. TSC Decision (Traffic Light via PPO)
            action, _state = model.predict(obs, deterministic=True)

            # 2. NSC Decision (Base Station)
            # --- Primary path: raw vehicle counts from TraCI ---
            traci_available = False
            try:
                demand_snapshot = TelcoAllocator.get_traci_demand(TLS_ID)
                power_dist      = telco_bs._allocate_from_dict(demand_snapshot)
                traci_available = True
            except Exception:
                # --- Fallback path: normalised observation with moving average ---
                obs_buffer.append(obs[0])
                avg_obs         = np.mean(np.array(obs_buffer), axis=0)
                demand_snapshot = telco_bs.get_demand_snapshot(avg_obs)
                power_dist      = telco_bs.get_lane_allocation(avg_obs)

            # --- Per-user capacity: c_i / d_i ---
            per_user_cap = telco_bs.get_per_user_capacity(power_dist, demand_snapshot)

            # --- Network utility: U = sum_i [ d_i * log(1 + c_i / d_i) ] ---
            utility = telco_bs.compute_utility(power_dist, demand_snapshot)

            # 3. Environment Step
            obs, rewards, dones, infos = env.step(action)

            # Also update obs_buffer every step for fallback continuity
            if traci_available:
                obs_buffer.append(obs[0])

            # 4. Monitoring
            if step_counter % 5 == 0:
                source = "TraCI" if traci_available else (
                    "STABILIZING" if len(obs_buffer) < 5 else "SMOOTHED"
                )
                total_vehs = sum(demand_snapshot.values())
                print(f"\n[Step {step_counter}] ({source}) | U={utility:.4f} | Total vehicles={total_vehs}")
                print(f"  {'Lane':<8} {'Demand (veh)':>14} {'Alloc (Mbps)':>14} {'c/d (Mbps/veh)':>16}")
                print(f"  {'-'*56}")
                for lane in telco_bs.lane_ids:
                    d   = demand_snapshot[lane]
                    c   = power_dist[lane]
                    c_d = per_user_cap[lane]
                    # Show demand as integer when using TraCI (raw counts), float otherwise
                    d_fmt = f"{int(d):>14}" if traci_available else f"{d:>14.3f}"
                    print(f"  {lane:<8} {d_fmt} {c:>14.2f} {c_d:>16.4f}")

            step_counter += 1

            if dones[0] or step_counter >= 500:
                print("\nSimulation successfully completed.")
                break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")