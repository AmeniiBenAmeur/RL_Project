import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# --- WINDOWS DLL PATCH ---
try:
    import traci
    sys.modules["libsumo"] = traci
    os.environ['LIBSUMO_AS_TRACI'] = '1'
except ImportError:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

warnings.filterwarnings("ignore", category=UserWarning)

from utils.make_tsc_env import make_env
from bs_env_2 import BSEnv  
from _config import SCENARIO_CONFIGS

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))

SCENARIO = "Hongkong_YMT"
config = SCENARIO_CONFIGS.get(SCENARIO)

def get_util(c_list, d_list, epsilon=1e-6):
    c = np.array(c_list)
    d = np.array(d_list)
    return np.sum(d * np.log1p(c / (d + epsilon)))

def run_multi_capacity_comparison():
    CAPACITIES = [50.0, 20.0, 10.0]
    summary_results = []
    
    log_dir = os.path.join(current_dir, "logs_multi_cap")
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    tsc_norm_path = os.path.join(current_dir, 'results', SCENARIO, 'models', 'last_vec_normalize.pkl')
    tsc_model_path = os.path.join(current_dir, 'results', SCENARIO, 'models', 'last_rl_model.zip')
    sumo_cfg = os.path.join(root_path, "sim_envs", config['SCENARIO_NAME'], f"{config['NETFILE']}.sumocfg")

    params = {
        'tls_id': config["JUNCTION_NAME"], 
        'num_seconds': 600, 
        'number_phases': config["PHASE_NUMBER"], 
        'sumo_cfg': sumo_cfg,
        'use_gui': False, 
        'log_file': log_dir
    }

    bs_norm_path = os.path.join(root_path, "vec_bs_hybrid_final.pkl")
    ppo_model_path = os.path.join(root_path, "ppo_bs_hybrid_final.zip")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cap in CAPACITIES:
        print(f"\n--- Evaluating Capacity: {cap} Mbps ---")
        
        tsc_env_fn = make_env(env_index=f'eval_{cap}', **params)
        tsc_env = DummyVecEnv([tsc_env_fn])
        tsc_env = VecNormalize.load(tsc_norm_path, tsc_env)
        tsc_env.training = False
        
        tsc_model = PPO.load(tsc_model_path, env=tsc_env, device=device)

        def make_bs(): return BSEnv(tsc_env, tsc_model, c_max=cap)
        eval_env_raw = DummyVecEnv([make_bs])
        eval_env = VecNormalize.load(bs_norm_path, eval_env_raw)
        eval_env.training = False
        eval_env.norm_reward = False

        ppo_agent = PPO.load(ppo_model_path, env=eval_env, device=device)

        obs = eval_env.reset()
        step_results = []

        for i in range(400):
            action_ppo, _ = ppo_agent.predict(obs, deterministic=True)
            
            raw_12_demand = eval_env.envs[0].unwrapped.get_12_lane_demand()
            controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(config["JUNCTION_NAME"])))
            
            dir_demand_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
            for j, lane in enumerate(controlled_lanes[:12]):
                d_name = eval_env.envs[0].unwrapped._get_lane_direction(lane)
                dir_demand_dict[d_name] += raw_12_demand[j]
            
            current_demand = np.array([dir_demand_dict[d] for d in ["North", "South", "East", "West"]])
            
            # Gestione pesi PPO
            # Assicuriamoci che l'azione sia un array e non una tupla (SB3 a volte lo fa)
            act = action_ppo[0] if isinstance(action_ppo, (list, np.ndarray)) else action_ppo
            weights_ppo = np.clip(act, 1e-6, None)
            c_ppo = (weights_ppo / np.sum(weights_ppo)) * cap
            
            sum_d = np.sum(current_demand)
            # Direct Strategy (KKT Optimal)
            c_dir = (current_demand / (sum_d + 1e-7)) * cap if sum_d > 0 else np.ones(4)*(cap/4)
            # Uniform Baseline
            c_uni = np.ones(4) * (cap / 4)

            u_ppo = get_util(c_ppo, current_demand)
            u_dir = get_util(c_dir, current_demand)
            u_uni = get_util(c_uni, current_demand)

            step_results.append({
                'step': i, 'u_ppo': u_ppo, 'u_direct': u_dir, 'u_uni': u_uni,
                'ppo_N': c_ppo[0], 'ppo_S': c_ppo[1], 'ppo_E': c_ppo[2], 'ppo_W': c_ppo[3],
                'dem_N': current_demand[0], 'dem_S': current_demand[1], 
                'dem_E': current_demand[2], 'dem_W': current_demand[3]
            })

            obs, _, done, _ = eval_env.step(action_ppo)
            if done[0]: break

        df = pd.DataFrame(step_results)
        
        # --- CALCOLO GAIN ---
        avg_ppo = df['u_ppo'].mean()
        avg_uni = df['u_uni'].mean()
        avg_dir = df['u_direct'].mean()
        
        gain_vs_uni = ((avg_ppo - avg_uni) / (avg_uni + 1e-9)) * 100
        gain_vs_dir = ((avg_ppo - avg_dir) / (avg_dir + 1e-9)) * 100 # Di solito sarà negativo o zero
        
        # Print veloce a console per ogni loop
        print(f"PPO vs Uniform Gain: {gain_vs_uni:+.2f}%")
        print(f"PPO vs Direct Gap:   {gain_vs_dir:+.2f}%")
        
        # --- PLOTTING ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
        
        ax1.plot(df['step'], df['u_ppo'], label='PPO Hybrid (RL)', color='forestgreen', linewidth=2)
        ax1.plot(df['step'], df['u_direct'], label='Direct KKT (Optimal)', color='royalblue', linestyle='--', alpha=0.7)
        ax1.plot(df['step'], df['u_uni'], label='Uniform Baseline', color='firebrick', linestyle=':', alpha=0.8)
        ax1.set_title(f"Utility Analysis | {cap} Mbps | Gain vs Uni: {gain_vs_uni:+.2f}% | Gap vs Dir: {gain_vs_dir:+.2f}%", fontsize=13)
        ax1.set_ylabel("Dixon-Log Utility")
        ax1.legend(loc='upper right', frameon=True, shadow=True); ax1.grid(True, linestyle='--', alpha=0.4)

        colors = {'N': 'blue', 'S': 'orange', 'E': 'green', 'W': 'red'}
        ax2.plot(df['step'], df['ppo_N'], label='Alloc North', color=colors['N'], linewidth=1.2)
        ax2.plot(df['step'], df['ppo_S'], label='Alloc South', color=colors['S'], linewidth=1.2)
        ax2.plot(df['step'], df['ppo_E'], label='Alloc East', color=colors['E'], linewidth=1.2)
        ax2.plot(df['step'], df['ppo_W'], label='Alloc West', color=colors['W'], linewidth=1.2)
        ax2.set_ylabel("Mbps"); ax2.legend(loc='upper right', ncol=2); ax2.grid(True, linestyle='--', alpha=0.4)

        ax3.plot(df['step'], df['dem_N'], label='North Demand', color=colors['N'], alpha=0.8)
        ax3.plot(df['step'], df['dem_S'], label='South Demand', color=colors['S'], alpha=0.8)
        ax3.plot(df['step'], df['dem_E'], label='East Demand', color=colors['E'], alpha=0.8)
        ax3.plot(df['step'], df['dem_W'], label='West Demand', color=colors['W'], alpha=0.8)
        ax3.set_ylabel("Vehicle Count"); ax3.set_xlabel("Simulation Step (s)")
        ax3.legend(loc='upper right', ncol=2); ax3.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"cardinal_analysis_cap_{int(cap)}.png"), dpi=150)
        plt.close()

        summary_results.append({
            'Capacity': cap, 
            'PPO_Avg': avg_ppo, 
            'Gain_vs_Uni_%': gain_vs_uni,
            'Gap_vs_Direct_%': gain_vs_dir
        })
        
        tsc_env.close()

    # --- FINAL TABLE ---
    print("\n" + "="*70)
    print(f"{'CAPACITY':<12} | {'PPO_AVG':<10} | {'GAIN vs UNI %':<15} | {'GAP vs DIR %':<15}")
    print("-" * 70)
    for res in summary_results:
        print(f"{res['Capacity']:<12.1f} | {res['PPO_Avg']:<10.4f} | {res['Gain_vs_Uni_%']:>+13.2f}% | {res['Gap_vs_Direct_%']:>+13.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_multi_capacity_comparison()