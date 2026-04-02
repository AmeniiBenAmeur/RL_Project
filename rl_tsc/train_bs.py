import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bs_env import BSEnv
from utils.custom_models import CustomModel
from utils.make_tsc_env import make_env
from _config import SCENARIO_CONFIGS
from tshub.utils.get_abs_path import get_abs_path

# Clean SUMO Environment Setup
SUMO_PATH = r'C:\Program Files (x86)\Eclipse\Sumo'
os.environ['SUMO_HOME'] = SUMO_PATH
if os.path.join(SUMO_PATH, 'bin') not in os.environ['PATH']:
    os.environ['PATH'] = os.path.join(SUMO_PATH, 'bin') + os.pathsep + os.environ['PATH']

try:
    import traci
    sys.modules["libsumo"] = traci
    os.environ['LIBSUMO_AS_TRACI'] = '1'
except ImportError:
    pass

path_convert = get_abs_path(__file__)
config = SCENARIO_CONFIGS.get("Hongkong_YMT")

def main():
    log_dir = path_convert("./log_train_bs/")
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    params = {
        'tls_id': config["JUNCTION_NAME"],
        'num_seconds': 3600, 
        'number_phases': config["PHASE_NUMBER"],
        'sumo_cfg': path_convert(f"../sim_envs/{config['SCENARIO_NAME']}/{config['NETFILE']}.sumocfg"),
        'use_gui': False,
        'log_file': log_dir,
    }

    # 1. Load TSC environment
    tsc_env_fn = make_env(env_index='train', **params)
    tsc_env = DummyVecEnv([tsc_env_fn])
    tsc_norm_path = path_convert(f'./results/{config["SCENARIO_NAME"]}/models/last_vec_normalize.pkl')
    tsc_env = VecNormalize.load(tsc_norm_path, tsc_env)
    tsc_env.training = False

    # 2. Load Pre-trained TSC model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tsc_model_path = path_convert(f'./results/{config["SCENARIO_NAME"]}/models/last_rl_model.zip')
    tsc_model = PPO.load(tsc_model_path, env=tsc_env, device=device)

    # 3. Initialize BS Environment
    env = DummyVecEnv([lambda: BSEnv(tsc_env, tsc_model, c_max=50.0)])
    # CRITICAL: norm_reward=False is essential for integer demand rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(
        features_extractor_class=CustomModel,
        features_extractor_kwargs=dict(features_dim=16),
    )

    # 4. Define PPO Agent
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log="./ppo_bs_logs/",
        device=device
    )

    print("\n[START] Training with Debug Prints active...")
    model.learn(total_timesteps=300000)

    model.save("ppo_bs_transformer_final")
    env.save("vec_bs_transformer_final.pkl")
    print("[END] Files saved.")

if __name__ == "__main__":
    main()