"""
Final Strategy Comparison: Uniform vs KKT vs PPO
Architecture: Spatio-Temporal Transformer (5x12) -> 4-Sector Allocation
Scenario: Hongkong_YMT
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from gymnasium import spaces

# --- WINDOWS DLL PATCH ---
try:
    import traci
    sys.modules["libsumo"] = traci
    os.environ['LIBSUMO_AS_TRACI'] = '1'
except ImportError:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tshub.utils.get_abs_path import get_abs_path
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS

path_convert = get_abs_path(__file__)

# --- CONFIGURATION ---
SCENARIO  = "Hongkong_YMT"
config    = SCENARIO_CONFIGS.get(SCENARIO)
C_MAX     = 50.0
STEPS     = 400
JUNCTION  = config["JUNCTION_NAME"] 
DIRECTIONS = ["North", "South", "East", "West"]

# ---------------------------------------------------------------------------
# 1. UTILS & GEOMETRY
# ---------------------------------------------------------------------------
def lane_to_direction(lane_id: str) -> str:
    try:
        shape = traci.lane.getShape(lane_id)
        dx, dy = shape[-1][0] - shape[0][0], shape[-1][1] - shape[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        if -45 <= angle < 45: return "West"
        elif 45 <= angle < 135: return "South"
        elif angle >= 135 or angle < -135: return "East"
        else: return "North"
    except: return "Unknown"

def compute_utility(allocation, demand, epsilon=1e-6):
    total = 0.0
    for c_i, d_i in zip(allocation, demand):
        if d_i > 0:
            total += d_i * np.log1p(c_i / d_i)
    return float(total)

# ---------------------------------------------------------------------------
# 2. DUMMY ENV FOR LOADING
# ---------------------------------------------------------------------------
class _DummyObs(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 12), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    def reset(self, **kwargs): return np.zeros((5, 12), np.float32), {}
    def step(self, a): return np.zeros((5, 12), np.float32), 0.0, False, False, {}

# ---------------------------------------------------------------------------
# 3. MAIN EVALUATION
# ---------------------------------------------------------------------------
def run_comparison():
    print(f"\n{'='*65}\n STRATEGY COMPARISON: Uniform | KKT | PPO Hybrid\n{'='*65}")

    # Environment Setup
    log_dir = path_convert("./logs_comparison/")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    sumo_cfg = path_convert(f"../sim_envs/{config['SCENARIO_NAME']}/{config['NETFILE']}.sumocfg")

    # Load TSC (Traffic Light)
    tsc_env_fn = make_env(env_index='compare', tls_id=JUNCTION, num_seconds=STEPS+200, 
                          number_phases=config["PHASE_NUMBER"], sumo_cfg=sumo_cfg, use_gui=False, log_file=log_dir)
    tsc_env = DummyVecEnv([tsc_env_fn])
    tsc_norm_path = path_convert(f"./results/{SCENARIO}/models/last_vec_normalize.pkl")
    tsc_env = VecNormalize.load(tsc_norm_path, tsc_env)
    tsc_env.training = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsc_model = PPO.load(path_convert(f"./results/{SCENARIO}/models/last_rl_model.zip"), env=tsc_env, device=device)

    # Load NSC (Network Slice Controller)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    ppo_model_path = os.path.join(root_dir, "ppo_bs_hybrid_final") # SB3 aggiunge .zip da solo
    ppo_norm_path = os.path.join(root_dir, "vec_bs_hybrid_final.pkl")

    print(f"Loading NSC Model from: {ppo_model_path}")
    ppo_model = PPO.load(ppo_model_path, device=device)

    print(f"Loading NSC Normalizer from: {ppo_norm_path}")
    ppo_norm = VecNormalize.load(ppo_norm_path, DummyVecEnv([_DummyObs]))
    ppo_norm.training = False
    ppo_norm.norm_reward = False
    
    # Simulation Variables
    obs = tsc_env.reset()
    obs_buffer = [] 
    history = []
    
    try:
        for step in range(STEPS):
            # A. TSC Move
            tsc_action, _ = tsc_model.predict(obs, deterministic=True)
            obs, _, done, _ = tsc_env.step(tsc_action)

            # B. Demand Extraction (TraCI)
            raw_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(JUNCTION)))
            dir_demand = {d: 0.0 for d in DIRECTIONS}
            lane_counts_12 = np.zeros(12, dtype=np.float32)
            
            for i, lane_id in enumerate(raw_lanes[:12]):
                count = float(traci.lane.getLastStepVehicleNumber(lane_id))
                lane_counts_12[i] = count
                d_name = lane_to_direction(lane_id)
                if d_name in dir_demand: dir_demand[d_name] += count
            
            demand_vec = np.array([dir_demand[d] for d in DIRECTIONS])

            # C. PPO Observation (Rolling Window 5x12)
            obs_buffer.append(lane_counts_12)
            if len(obs_buffer) > 5: obs_buffer.pop(0)
            while len(obs_buffer) < 5: obs_buffer.insert(0, np.zeros(12))
            
            ppo_obs_raw = np.array(obs_buffer, dtype=np.float32)[np.newaxis, ...]
            ppo_obs_norm = ppo_norm.normalize_obs(ppo_obs_raw)

            # D. ALLOCATIONS
            # 1. Uniform
            c_uni = np.ones(4) * (C_MAX / 4.0)
            # 2. KKT
            sum_d = np.sum(demand_vec)
            c_kkt = (demand_vec / (sum_d + 1e-9)) * C_MAX if sum_d > 0 else c_uni
            # 3. PPO
            ppo_act, _ = ppo_model.predict(ppo_obs_norm, deterministic=True)
            w_ppo = np.clip(ppo_act.flatten(), 1e-6, None)
            c_ppo = (w_ppo / np.sum(w_ppo)) * C_MAX

            # E. UTILITIES
            u_uni = compute_utility(c_uni, demand_vec)
            u_kkt = compute_utility(c_kkt, demand_vec)
            u_ppo = compute_utility(c_ppo, demand_vec)

            history.append({"step": step, "Uniform": u_uni, "KKT": u_kkt, "PPO": u_ppo, 
                            "dN": dir_demand["North"], "dS": dir_demand["South"], 
                            "dE": dir_demand["East"], "dW": dir_demand["West"]})

            if step % 50 == 0:
                print(f"Step {step:03d} | Total Veh: {int(sum_d):02d} | PPO Utility: {u_ppo:.3f}")

            if done[0]: break

    finally:
        tsc_env.close()

    # --- REPORTING ---
    df = pd.DataFrame(history)
    avgs = df[["Uniform", "KKT", "PPO"]].mean()
    print(f"\n{'='*65}\n{'STRATEGY':<12} | {'AVG UTILITY':>12} | {'GAIN vs UNIFORM':>18}\n{'-'*65}")
    for name in ["Uniform", "KKT", "PPO"]:
        gain = ((avgs[name] - avgs["Uniform"]) / (avgs["Uniform"] + 1e-9)) * 100
        print(f"{name:<12} | {avgs[name]:>12.4f} | {gain:>17.2f}%")
    print(f"{'='*65}\n")

    # --- PLOTTING ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(df["step"], df["KKT"], '--', color='blue', label='KKT (Optimum)')
    ax[0].plot(df["step"], df["PPO"], color='green', label='PPO Transformer')
    ax[0].plot(df["step"], df["Uniform"], ':', color='red', label='Uniform Baseline')
    ax[0].set_ylabel("PF Utility"); ax[0].legend(); ax[0].grid(alpha=0.3)
    
    for d, c in zip(["dN", "dS", "dE", "dW"], ["blue", "orange", "green", "red"]):
        ax[1].fill_between(df["step"], df[d], alpha=0.3, label=d[1:], color=c)
    ax[1].set_ylabel("Vehicles"); ax[1].set_xlabel("Steps"); ax[1].legend(ncol=4); ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path_convert("./comparison_final.png"), dpi=200)
    plt.show()

if __name__ == "__main__":
    run_comparison()