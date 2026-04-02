import os
import sys
import warnings
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# --- FIX: Windows DLL & TraCI ---
try:
    import traci
    sys.modules["libsumo"] = traci
    os.environ['LIBSUMO_AS_TRACI'] = '1'
except ImportError: pass
warnings.filterwarnings("ignore", category=UserWarning)

from bs_env_2 import BSEnv 
from utils.make_tsc_env import make_env
from utils.custom_models import CustomModel 
from _config import SCENARIO_CONFIGS
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)
SCENARIO = "Hongkong_YMT"
config = SCENARIO_CONFIGS.get(SCENARIO)

def train():
    # 1. Setup TSC Environment with its own Normalization
    params = {
        'tls_id': config["JUNCTION_NAME"],
        'num_seconds': 3600,
        'number_phases': config["PHASE_NUMBER"],
        'sumo_cfg': path_convert(f"../sim_envs/{config['SCENARIO_NAME']}/{config['NETFILE']}.sumocfg"),
        'use_gui': False,
        'log_file': path_convert('./logs_cardinal/'),
    }
    
    # Create Base TSC
    tsc_env_fn = make_env(env_index='train_bs', **params)
    tsc_env = DummyVecEnv([tsc_env_fn])
    
    # LOAD TSC NORMALIZER (Critical!)
    tsc_norm_path = path_convert(f'./results/{SCENARIO}/models/last_vec_normalize.pkl')
    tsc_env = VecNormalize.load(tsc_norm_path, tsc_env)
    tsc_env.training = False # Don't update traffic stats during BS training
    tsc_env.norm_reward = False
    
    # LOAD TSC MODEL
    tsc_model_path = path_convert(f'./results/{SCENARIO}/models/last_rl_model.zip')
    tsc_model = PPO.load(tsc_model_path, env=tsc_env)

    # 2. Setup BS Environment
    def make_bs_env():
        return BSEnv(tsc_env, tsc_model, c_max=50.0)
    
    env = DummyVecEnv([make_bs_env])
    # Normalize Observations (for Transformer) but NOT Rewards (for PF validity)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # 3. PPO with Custom Transformer
    policy_kwargs = dict(
        features_extractor_class=CustomModel,
        features_extractor_kwargs=dict(features_dim=16),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        tensorboard_log=path_convert('./logs_cardinal/')
    )

    print("--- Starting Hybrid Training: 12-Lane Input (TraCI) -> 4-Direction Output ---")
    model.learn(total_timesteps=300000, callback=CheckpointCallback(save_freq=5000, save_path='./models_cardinal/'))

    model.save("ppo_bs_hybrid_final")
    env.save("vec_bs_hybrid_final.pkl")
    print("Training Complete.")

if __name__ == "__main__":
    train()