import numpy as np


class TelcoAllocator:
    """
    Advanced Network Slice Controller (NSC) using Alpha-Fairness and Dixon-log utility.
    Optimizes bandwidth allocation based on real-time traffic demand from SUMO.

    Utility function: U = sum_i [ d_i * log(1 + c_i / d_i) ]
    where:
        d_i = number of vehicles observed in lane i  (raw TraCI count)
        c_i = capacity allocated to lane i [Mbps]

    The per-user capacity seen by each vehicle is c_i / d_i.

    KKT closed-form solution (all active lanes always receive allocation):
        c_i = d_i * C_max / sum_j(d_j)
    i.e. allocation is proportional to demand, derived rigorously from utility maximisation.

    Alpha=1 -> Proportional fairness, assign power where there are more users,
               but considering also parts where there are fewer users.
    Alpha=0 -> Max-Throughput, assign power only where there are more users,
               not considering other directions.
    """

    DIRECTIONS = ["North", "South", "East", "West"]

    def __init__(self, c_max=50.0, alpha=1.0, epsilon=1e-6):
        """
        Initialize the allocator.
        :param c_max:    Maximum available capacity [Mbps].
        :param alpha:    Fairness parameter. 1.0 = Proportional Fairness.
        :param epsilon:  Small regularisation value to avoid log(0) when d_i = 0.
        """
        self.c_max    = c_max
        self.alpha    = alpha
        self.epsilon  = epsilon
        self.lane_ids = self.DIRECTIONS

    # ------------------------------------------------------------------
    # TraCI-based demand (raw vehicle counts)
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_to_direction(dx: float, dy: float) -> str:
        """
        Maps a 2D displacement vector to the cardinal direction that the lane
        is coming FROM (i.e. the origin side of the junction).

        SUMO coordinate system: x increases East, y increases North.
        arctan2(dy, dx) gives:
             90 deg -> lane points North  -> traffic comes from South
              0 deg -> lane points East   -> traffic comes from West
            -90 deg -> lane points South  -> traffic comes from North
            180 deg -> lane points West   -> traffic comes from East

        We invert to label by source direction.
        """
        angle = np.degrees(np.arctan2(dy, dx))

        if -45 <= angle < 45:
            return "West"    # lane heading East  -> vehicles from West
        elif 45 <= angle < 135:
            return "South"   # lane heading North -> vehicles from South
        elif angle >= 135 or angle < -135:
            return "East"    # lane heading West  -> vehicles from East
        else:
            return "North"   # lane heading South -> vehicles from North

    @staticmethod
    def get_traci_demand(tls_id: str) -> dict:
        """
        Reads raw vehicle counts directly from TraCI for each cardinal direction.

        For each lane controlled by the traffic light:
          1. Derives the travel direction from the lane geometry (shape vector).
          2. Inverts it to the source direction (where vehicles come FROM).
          3. Accumulates vehicle counts with traci.lane.getLastStepVehicleNumber().

        :param tls_id: SUMO traffic-light / junction ID (e.g. 'J1').
        :return:       dict {direction: raw_vehicle_count (int)}
        """
        try:
            import traci
        except ImportError:
            raise RuntimeError("TraCI is not available. Cannot read live vehicle counts.")

        counts = {d: 0 for d in TelcoAllocator.DIRECTIONS}

        # Retrieve the unique set of lanes controlled by this traffic light
        controlled = traci.trafficlight.getControlledLanes(tls_id)
        seen = set()
        unique_lanes = [l for l in controlled if not (l in seen or seen.add(l))]

        for lane_id in unique_lanes:
            shape = traci.lane.getShape(lane_id)
            if len(shape) < 2:
                continue  # skip degenerate lanes

            # Direction vector: from lane start to lane end
            dx = shape[-1][0] - shape[0][0]
            dy = shape[-1][1] - shape[0][1]

            direction = TelcoAllocator._angle_to_direction(dx, dy)
            counts[direction] += traci.lane.getLastStepVehicleNumber(lane_id)

        return counts

    # ------------------------------------------------------------------
    # Observation-based demand (fallback from normalised SUMO obs)
    # ------------------------------------------------------------------

    def _get_demand_from_obs(self, obs) -> dict:
        """
        Extracts and aggregates traffic demand from the observation matrix.
        Safely handles different observation shapes.
        Used as a fallback when TraCI is not available.

        :param obs: raw or pre-processed observation array from SUMO.
        :return:    dict {lane_id: demand_value}
        """
        obs = np.array(obs)

        if obs.size < 12:
            # Small / pre-processed observation (e.g. from a Transformer)
            demands = {
                "North": float(obs[0]) if obs.size > 0 else 0.0,
                "South": float(obs[1]) if obs.size > 1 else 0.0,
                "East":  float(obs[2]) if obs.size > 2 else 0.0,
                "West":  float(obs[3]) if obs.size > 3 else 0.0,
            }
        else:
            # Original logic for the 12-movement raw vector.
            # Take the last 12 values (most recent frame) and map to 4 BS directions.
            recent_obs = obs.flatten()[-12:]
            demands = {
                "North": float(np.mean(recent_obs[0:2])),
                "South": float(np.mean(recent_obs[2:4])),
                "East":  float(recent_obs[4]),
                "West":  float(recent_obs[5]),
            }
        return demands

    # ------------------------------------------------------------------
    # KKT allocation core
    # ------------------------------------------------------------------

    def _kkt_allocate(self, demands: np.ndarray) -> np.ndarray:
        """
        KKT closed-form solver for: max sum_i [ d_i * log(1 + c_i / d_i) ]
                                     s.t. sum(c_i) = C_max,  c_i >= 0

        Derivation:
            Lagrangian stationarity: d_i / (d_i + c_i) = lambda  for all active i
            => c_i = d_i / lambda - d_i = d_i * (1/lambda - 1)
            Summing over active lanes: C_max = (1/lambda - 1) * sum(d_active)
            => 1/lambda - 1 = C_max / sum(d_active)
            => c_i = d_i * C_max / sum(d_active)

        Lanes with d_i = 0 are replaced by epsilon so the ratio c_i/d_i remains
        finite; their allocation is effectively zero because epsilon << C_max.

        :param demands: numpy array of raw per-lane vehicle counts (shape: [4,])
        :return:        numpy array of allocated capacities in Mbps (shape: [4,])
        """
        # Guard: replace zero demands with epsilon to avoid division by zero
        d = np.where(demands <= 0, self.epsilon, demands.astype(float))

        # Closed-form proportional allocation derived from KKT conditions
        allocations = d * self.c_max / d.sum()
        return allocations

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_lane_allocation(self, obs) -> dict:
        """
        Allocation from normalised SUMO observation (fallback path).
        Used when TraCI direct access is not available.

        :param obs: raw or pre-processed observation from SUMO environment.
        :return:    dict {lane_id: allocated_Mbps}
        """
        demands_dict = self._get_demand_from_obs(obs)
        return self._allocate_from_dict(demands_dict)

    def get_lane_allocation_from_traci(self, tls_id: str) -> dict:
        """
        Allocation using raw TraCI vehicle counts (preferred path).

        Step logic:
            1. Observe d_i = raw integer vehicle count per lane from TraCI.
            2. Determine direction (N/S/E/W) from lane geometry automatically.
            3. Choose c_i for each lane via KKT: c_i = d_i * C_max / sum(d_j).
            4. Per-user capacity seen by each vehicle = c_i / d_i.

        :param tls_id: SUMO traffic-light / junction ID.
        :return:       dict {lane_id: allocated_Mbps}
        """
        demands_dict = self.get_traci_demand(tls_id)
        return self._allocate_from_dict(demands_dict)

    def _allocate_from_dict(self, demands_dict: dict) -> dict:
        """
        Shared KKT allocation logic given a pre-computed demand dict.

        :param demands_dict: {lane_id: demand_value}
        :return:             {lane_id: allocated_Mbps}
        """
        lanes   = list(demands_dict.keys())
        demands = np.array(list(demands_dict.values()))

        total_demand = np.sum(demands)

        # Zero-traffic edge case: no vehicles detected -> allocate nothing
        if total_demand <= 0:
            return {lane: 0.0 for lane in lanes}

        allocations = self._kkt_allocate(demands)
        return {lanes[i]: float(allocations[i]) for i in range(len(lanes))}

    def compute_utility(self, allocation_dict: dict, demand_dict: dict) -> float:
        """
        Evaluates the Dixon-log network utility:
            U = sum_i [ d_i * log(1 + c_i / d_i) ]

        Higher U means a better balance between total capacity and per-user fairness.
        The term c_i / d_i represents the per-vehicle capacity in lane i.

        :param allocation_dict: {lane: allocated_Mbps}
        :param demand_dict:     {lane: vehicle count (raw int or normalised float)}
        :return:                scalar utility value
        """
        utility = 0.0
        for lane in self.lane_ids:
            c      = allocation_dict.get(lane, 0.0)
            d      = demand_dict.get(lane, 0.0)
            d_safe = d if d > 0 else self.epsilon      # guard against d=0
            utility += d_safe * np.log1p(c / d_safe)  # d * log(1 + c/d)
        return utility

    def get_demand_snapshot(self, obs) -> dict:
        """
        Exposes per-lane demand from the normalised observation (fallback).

        :param obs: raw or pre-processed observation from SUMO environment.
        :return:    dict {lane_id: demand_value}
        """
        return self._get_demand_from_obs(obs)

    def get_per_user_capacity(self, allocation_dict: dict, demand_dict: dict) -> dict:
        """
        Computes the effective per-vehicle capacity c_i / d_i for each lane.
        This is the quantity that directly appears inside the log of the utility.

        :param allocation_dict: {lane: allocated_Mbps}
        :param demand_dict:     {lane: vehicle count}
        :return:                dict {lane: per_vehicle_Mbps}
        """
        result = {}
        for lane in self.lane_ids:
            c      = allocation_dict.get(lane, 0.0)
            d      = demand_dict.get(lane, 0.0)
            d_safe = d if d > 0 else self.epsilon
            result[lane] = c / d_safe
        return result