import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

class BSEnv(gym.Env):
    def __init__(self, tsc_env, tsc_model, c_max=50.0):
        super(BSEnv, self).__init__()
        self.tsc_env = tsc_env
        self.tsc_model = tsc_model
        self.c_max = c_max
        self.junction_id = "J1"
        
        # Discover unique lanes immediately to set the correct action space
        # We use a temporary traci call or a safe default if traci isn't live yet
        try:
            raw_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
            self.unique_lanes = list(dict.fromkeys(raw_lanes))
            self.n_actions = len(self.unique_lanes)
        except Exception:
            # If traci is not yet started, we'll fix this during the first reset
            self.unique_lanes = []
            self.n_actions = 6 # Placeholder, will be updated in reset()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5, 12), dtype=np.float32
        )
        
        # Dynamic action space based on actual lanes
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        
        self.last_obs = None
        self._ep_reward = 0.0
        self._ep_length = 0

    def _update_unique_lanes(self):
        """Updates the list of unique lanes and ensures action consistency."""
        raw_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        self.unique_lanes = list(dict.fromkeys(raw_lanes))
        if len(self.unique_lanes) != self.n_actions:
            print(f"[WARN] Expected {self.n_actions} actions, but found {len(self.unique_lanes)} lanes. Check consistency.")

    def _get_movement_vehicle_counts(self):
        """Retrieves exact integer vehicle counts for each unique movement lane."""
        counts = np.zeros(len(self.unique_lanes), dtype=np.float32)
        try:
            for i, lane_id in enumerate(self.unique_lanes):
                counts[i] = traci.lane.getLastStepVehicleNumber(lane_id)
            return counts
        except Exception:
            return np.zeros(self.n_actions, dtype=np.float32)

    def step(self, action):
        # 1. TSC Brain Decision
        tsc_action, _ = self.tsc_model.predict(self.last_obs, deterministic=True)
        tsc_action_int = int(tsc_action.flatten()[0]) if hasattr(tsc_action, "shape") else int(tsc_action)

        # 2. Advance SUMO
        obs, _, dones, infos = self.tsc_env.step(np.array([tsc_action_int]))
        self.last_obs = obs[0] if len(obs.shape) == 3 else obs

        # 3. Capacity Allocation (Now perfectly sized to real_demand)
        clean_action = np.clip(action, 1e-6, None)
        allocation = (clean_action / (np.sum(clean_action) + 1e-7)) * self.c_max
        
        # 4. PF Reward
        real_demand = self._get_movement_vehicle_counts()
        reward = 0.0
        
        # Ensure we don't go out of bounds if TraCI lanes shifted
        loop_range = min(len(real_demand), len(allocation))
        for i in range(loop_range):
            d_i = real_demand[i]
            c_i = allocation[i]
            if d_i >= 1.0:
                reward += d_i * np.log1p(c_i / d_i)
        
        if self._ep_length % 100 == 0:
            print(f"[DEBUG] Step: {self._ep_length} | Demand Sum: {np.sum(real_demand)} | Reward: {reward:.4f}")

        self._ep_reward += reward
        self._ep_length += 1
        info = infos[0]
        terminated = bool(dones[0] if isinstance(dones, (list, np.ndarray)) else dones)
        if terminated:
            info['episode'] = {'r': self._ep_reward, 'l': self._ep_length}

        return self.last_obs, float(reward), terminated, False, info

    def reset(self, seed=None, options=None):
        self._ep_reward = 0.0
        self._ep_length = 0
        res = self.tsc_env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        self.last_obs = obs[0] if len(obs.shape) == 3 else obs
        # Update unique lanes on reset to ensure TraCI is active
        self._update_unique_lanes()
        return self.last_obs, {}