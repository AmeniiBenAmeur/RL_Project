import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# --- Windows DLL Patch for SUMO ---
try:
    import traci
    sys.modules["libsumo"] = traci
    os.environ['LIBSUMO_AS_TRACI'] = '1'
except ImportError:
    pass

# --- Project Imports ---
from utils.make_tsc_env import make_env
from bs_env import BSEnv
from _config import SCENARIO_CONFIGS

# --- PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, ".."))

SCENARIO = "Hongkong_YMT"
config = SCENARIO_CONFIGS.get(SCENARIO)

def run_multi_capacity_test():
    CAPACITIES = [50.0, 20.0, 10.0] 
    summary_results = []
    
    log_dir = os.path.join(root_path, "log_compare_final")
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    sumo_cfg = os.path.join(root_path, "sim_envs", config['SCENARIO_NAME'], f"{config['NETFILE']}.sumocfg")
    
    
    tsc_env_fn = make_env(
        env_index='eval', 
        tls_id=config["JUNCTION_NAME"],
        num_seconds=500,
        number_phases=config["PHASE_NUMBER"],
        sumo_cfg=sumo_cfg,
        use_gui=False,
        log_file=log_dir
    )
    
    tsc_env_base = DummyVecEnv([tsc_env_fn])
    tsc_norm_path = os.path.join(current_dir, 'results', SCENARIO, 'models', 'last_vec_normalize.pkl')
    tsc_model_path = os.path.join(current_dir, 'results', SCENARIO, 'models', 'last_rl_model.zip')
    
    tsc_env_base = VecNormalize.load(tsc_norm_path, tsc_env_base)
    tsc_env_base.training = False
    tsc_model = PPO.load(tsc_model_path, env=tsc_env_base) 


    ppo_agent_path = os.path.join(root_path, "ppo_bs_transformer_final.zip")
    bs_norm_path = os.path.join(root_path, "vec_bs_transformer_final.pkl")
    ppo_agent = PPO.load(ppo_agent_path)

    for cap in CAPACITIES:
        print(f"\n--- Evaluating Capacity: {cap} Mbps (6-Lane Aligned Plotting) ---")
        eval_env_raw = DummyVecEnv([lambda: BSEnv(tsc_env_base, tsc_model, c_max=cap)])
        eval_env = VecNormalize.load(bs_norm_path, eval_env_raw)
        eval_env.training = False
        eval_env.norm_reward = False 

        obs = eval_env.reset()
        
        
        all_controlled = traci.trafficlight.getControlledLanes(config["JUNCTION_NAME"])
        unique_lanes_6 = list(dict.fromkeys(all_controlled))[:6] 

        step_results = []
        for i in range(400): 
            # 1. Recupero Demand (TraCI)
            real_demand = [traci.lane.getLastStepVehicleNumber(lid) for lid in unique_lanes_6]
            real_demand = np.array(real_demand, dtype=np.float32)
            sum_d = np.sum(real_demand)
            
            # 2. PPO Strategy
            action_ppo_raw, _ = ppo_agent.predict(obs, deterministic=True)
            act_raw = np.clip(action_ppo_raw[0], 0.1, None)
            p_ppo = (act_raw / np.sum(act_raw)) * cap

            # 3. Direct KKT & Uniform
            p_direct = (real_demand / (sum_d + 1e-7)) * cap if sum_d > 0 else np.ones(6) * (cap/6)
            p_uni = np.ones(6) * (cap / 6)

            # 4. Utility calculation
            def get_util(p, d):
                u = 0.0
                for pi, di in zip(p, d):
                    if di >= 1.0: u += di * np.log1p(pi / di)
                return u

            u_ppo = get_util(p_ppo, real_demand)
            u_direct = get_util(p_direct, real_demand)
            u_uni = get_util(p_uni, real_demand)

            # --- Mapping Cardinali per il Plot (Esempio raggruppamento per J1) ---
            # Nota: adatta gli indici [0,1,2...] in base all'ordine reale delle tue lane
            d_plot = {
                'step': i, 'u_ppo': u_ppo, 'u_direct': u_direct, 'u_uni': u_uni,
                'total_veh': sum_d,
                # Esempio: Nord (0,1), Sud (2,3), Est (4), Ovest (5)
                'dem_N': real_demand[0]+real_demand[1], 'dem_S': real_demand[2]+real_demand[3],
                'dem_E': real_demand[4], 'dem_W': real_demand[5],
                'ppo_N': p_ppo[0]+p_ppo[1], 'ppo_S': p_ppo[2]+p_ppo[3],
                'ppo_E': p_ppo[4], 'ppo_W': p_ppo[5]
            }

            obs, _, done, _ = eval_env.step(action_ppo_raw)
            step_results.append(d_plot)
            if done[0]: break

        # Graphic Generation
        df = pd.DataFrame(step_results)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        
        # Plot 1: Network Utility
        ax1.plot(df['step'], df['u_ppo'], label='PPO Transformer', color='green', linewidth=2)
        ax1.plot(df['step'], df['u_direct'], label='Direct (Optimal KKT)', color='blue', linestyle='--')
        ax1.plot(df['step'], df['u_uni'], label='Uniform', color='red', linestyle=':')
        ax1.set_ylabel("Network Utility (U)")
        ax1.set_title(f"Performance Analysis at {cap} Mbps")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Dynamic Allocation (Mbps)
        ax2.plot(df['step'], df['ppo_N'], label='North', color='blue')
        ax2.plot(df['step'], df['ppo_S'], label='South', color='orange')
        ax2.plot(df['step'], df['ppo_E'], label='East', color='green')
        ax2.plot(df['step'], df['ppo_W'], label='West', color='red')
        ax2.set_ylabel("Allocated Capacity (Mbps)")
        ax2.set_title("Dynamic Bandwidth Allocation (PPO Decision)")
        ax2.legend(loc='upper right', ncol=4)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Real Traffic Demand (Fill style)
        ax3.fill_between(df['step'], df['dem_N'], label='North', alpha=0.2, color='blue')
        ax3.fill_between(df['step'], df['dem_S'], label='South', alpha=0.2, color='orange')
        ax3.fill_between(df['step'], df['dem_E'], label='East', alpha=0.2, color='green')
        ax3.fill_between(df['step'], df['dem_W'], label='West', alpha=0.2, color='red')
        ax3.set_ylabel("Number of Vehicles")
        ax3.set_xlabel("Simulation Steps (Seconds)")
        ax3.set_title("Real Traffic Demand (TraCI Counts)")
        ax3.legend(loc='upper right', ncol=4)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"full_analysis_cap_{int(cap)}.png"))
        plt.close()

        summary_results.append({
            'Capacity': cap, 'PPO_Avg': df['u_ppo'].mean(), 
            'Uniform_Avg': df['u_uni'].mean(), 
            'Gain_%': ((df['u_ppo'].mean() - df['u_uni'].mean()) / (df['u_uni'].mean() + 1e-6)) * 100
        })

    print("\n" + "="*50)
    print(pd.DataFrame(summary_results).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    run_multi_capacity_test()
