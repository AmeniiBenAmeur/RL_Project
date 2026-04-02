import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from collections import deque

class BSEnv(gym.Env):
    """
    Hybrid NSC Environment:
    - Input: 5x12 (Matches authors' Transformer architecture)
    - Action: 4 Cardinal Directions (Matches physical BS sectors)
    - Reward: Real-demand PF (Non-normalized for accuracy)
    """
    def __init__(self, tsc_env, tsc_model, c_max=50.0):
        super(BSEnv, self).__init__()
        # tsc_env here is the VecNormalize-wrapped DummyVecEnv
        self.tsc_env = tsc_env
        self.tsc_model = tsc_model
        self.c_max = c_max
        self.junction_id = "J1"
        
        self.n_directions = 4 
        self.directions = ["North", "South", "East", "West"]
        self.window_size = 5 
        
        # Buffer for Transformer: stores 12-lane raw counts
        self.state_buffer = deque(maxlen=self.window_size)
        
        # Current observation from the TSC (needed for model.predict)
        self.last_tsc_obs = None

        # Actions: 4 sectors
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation: 5 steps of 12 lanes
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(5, 12), dtype=np.float32)

    def _get_lane_direction(self, lane_id):
        """Maps SUMO lane to source cardinal direction."""
        shape = traci.lane.getShape(lane_id)
        dx, dy = shape[-1][0] - shape[0][0], shape[-1][1] - shape[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        if -45 <= angle < 45: return "West"
        elif 45 <= angle < 135: return "South"
        elif angle >= 135 or angle < -135: return "East"
        else: return "North"

    def get_12_lane_demand(self):
        """Fetches raw counts for 12 lanes with padding."""
        controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.junction_id)))
        counts = [float(traci.lane.getLastStepVehicleNumber(l)) for l in controlled_lanes]
        while len(counts) < 12: counts.append(0.0)
        return np.array(counts[:12], dtype=np.float32)

    def step(self, action):
        # 1. TSC move: use the last observation saved in reset/step
        tsc_action, _ = self.tsc_model.predict(self.last_tsc_obs, deterministic=True)
        # We step the VecNormalize env
        new_tsc_obs, _, done, info = self.tsc_env.step(tsc_action)
        self.last_tsc_obs = new_tsc_obs # Update for next step

        # 2. NSC Logic: Update buffer with raw TraCI counts for the Transformer
        # Note: BSEnv input will be normalized by ITS OWN VecNormalize in train script
        raw_12_demand = self.get_12_lane_demand()
        self.state_buffer.append(raw_12_demand)
        
        # 3. Aggregate into 4 directions for REWARD calculation (Real demand)
        dir_demand = {d: 0.0 for d in self.directions}
        controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.junction_id)))
        for i, lane in enumerate(controlled_lanes[:12]):
            d_name = self._get_lane_direction(lane)
            dir_demand[d_name] += raw_12_demand[i]
        
        # 4. Allocation & PF Reward (Pure math, no normalization here)
        weights = np.clip(action, 1e-6, None)
        normalized_weights = weights / np.sum(weights)
        allocated_capacity = normalized_weights * self.c_max
        
        reward = 0.0
        for i, d_name in enumerate(self.directions):
            di = dir_demand[d_name]
            ci = allocated_capacity[i]
            if di > 0:
                reward += di * np.log1p(ci / di)
        
        return np.array(self.state_buffer, dtype=np.float32), reward, done[0], False, {}

    def reset(self, seed=None, options=None):
        # Reset the underlying TSC env and capture first obs
        self.last_tsc_obs = self.tsc_env.reset()
        
        initial_raw = self.get_12_lane_demand()
        self.state_buffer.clear()
        for _ in range(self.window_size):
            self.state_buffer.append(initial_raw)
            
        return np.array(self.state_buffer, dtype=np.float32), {}