from dataclasses import dataclass
import numpy as np
from .quadcopter_manager import QuadcopterManager
from .task_progression import StageStatus, Stage
from abc import ABC

@dataclass(frozen=True)
class Offsets:
    distances: np.ndarray
    directions: np.ndarray
    pursuer_ids: list
    invader_ids: list
    pursuer_positions: np.ndarray   # Array of pursuer positions (pursuers x 3)
    invader_positions: np.ndarray   # Array of invader positions (invaders x 3)
    pursuer_velocities: np.ndarray


        
class L3Stage1(Stage):

    def __init__(self, quadcopter_manager:QuadcopterManager, dome_radius:float, max_step_calls:int = 300, debug_on:bool = False):
        self.dome_radius = dome_radius
        self.quadcopter_manager = quadcopter_manager
        self.debug_on = debug_on
        
        self.max_step_calls = 300 # 300 calls = 20 seconds for rl_frequency = 15
  
        self.init_constants()
        self.init_globals()
        
    def init_constants(self):
        self.MAX_REWARD = 1000
        self.PROXIMITY_THRESHOLD = 2
        self.PROXIMITY_PENALTY = 1000
        self.CAPTURE_DISTANCE = 0.2
        
    def init_globals(self):
        self.num_invaders = 0
        self.num_pursuers = 0
        self.step_calls = 0
        self.max_building_life = 3
        self.building_life = self.max_building_life
    
#===============================================================================
# On Calls
#===============================================================================
    def on_reset(self):
        self.on_episode_end()
        self.on_episode_start()
        
    def on_episode_start(self):
        self._stage_status = StageStatus.RUNNING
        self.spawn_invader_squad()
        self.spawn_pursuer_squad()
        
        self.update_offsets()
        self.update_last_offsets()
    
    def on_episode_end(self):
        self.quadcopter_manager.clear_invaders()
        self.quadcopter_manager.clear_pursuers()
        self.init_globals()
        
    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """
        
        self.drive_invaders()
        
    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """
        
        self.update_offsets()
        self.last_offsets = self.filter_last_offsets(self.current_offsets, self.last_offsets)
        self.update_building_life()

        # Compute shared operations here:
        pursuer_norms = np.linalg.norm(self.current_offsets.pursuer_positions, axis=1)
        invader_norms = np.linalg.norm(self.current_offsets.invader_positions, axis=1)
        self.num_pursuers_outside_dome = np.sum(pursuer_norms > self.dome_radius)
        self.invaders_in_origin = np.sum(invader_norms < self.CAPTURE_DISTANCE)
            
        reward = self.compute_reward()
        
        self.process_invaders_in_origin()
        self.process_captured_invaders()
        termination = self.compute_termination()
        return reward, termination

        
    def on_step_end(self):
        
        self.update_last_offsets()
        
    
    @property
    def status(self):
        return self._stage_status
        
        

#===============================================================================
# Reward and Termination
#===============================================================================
    #TODO: eu tenho que n~ao usar o offset e sim usar o lidar.
    #TALVEZ SEJA MELHOR NAO
    def compute_reward(self):
        # Calculate Bounding Reward: Penalizes agents based on the number outside a defined boundary
        BoundingReward = -self.MAX_REWARD * self.num_pursuers_outside_dome

        # Calculate Collision With Target Reward: Provides a reward based on how many invaders are within the capture distance
        CollisionWithTargetReward = self.MAX_REWARD * np.sum(self.current_offsets.distances < self.CAPTURE_DISTANCE)
        
        # Calculate Distance Reward for Pursuers: Penalizes based on the mean distance between pursuers and invaders
        # This encourages pursuers to get close to invaders
        DistanceRewardForPursuers = -np.nansum(self.current_offsets.distances) / max(1, np.count_nonzero(~np.isnan(self.current_offsets.distances)))

        # Determine invaders that are near the origin based on a threshold
        near_origin_mask = self.current_offsets.distances < self.PROXIMITY_THRESHOLD

        # Count the number of invaders that are near the origin
        invaders_near_origin = np.sum(near_origin_mask)

        # Calculate the minimum distance of invaders near the origin, clipping it to avoid division by zero
        min_distance_near_origin = np.clip(np.min(self.current_offsets.distances[near_origin_mask] if near_origin_mask.any() else [0]), 1e-10, None)
        
        # Calculate Penalty for Invaders Nearing the Origin: 
        # It combines penalties for invaders being too close to the origin, being at the origin, and if the building is destroyed
        PenaltyForInvadersNearingOrigin = -(invaders_near_origin / min_distance_near_origin + 
                                            self.MAX_REWARD * (self.invaders_in_origin + (self.building_life == 0) * self.max_building_life))

        # Return the sum of all rewards and penalties
        return (BoundingReward + CollisionWithTargetReward + DistanceRewardForPursuers + PenaltyForInvadersNearingOrigin)


    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""
        # Step count exceeds maximum.
        if self.step_calls > self.max_step_calls:
            self._stage_status = StageStatus.FAILURE
            return True

        # Building life depleted.
        if self.building_life == 0:
            self._stage_status = StageStatus.FAILURE
            return True

        # Using computed value from on_step_middle
        if self.num_pursuers_outside_dome > 0:
            self._stage_status = StageStatus.FAILURE
            return True

        # All invaders captured.
        if self.num_invaders == 0:
            self._stage_status = StageStatus.SUCCESS
            return True

        # If none of the above conditions met, return False
        return False



    
#===============================================================================
# Others Implementations
#===============================================================================
    

    def generate_positions(self, num_positions:int, r:float) -> np.ndarray:
        
        
        # Random azimuthal angles limited to 0 to pi for x > 0
        thetas = np.random.uniform(0, np.pi, num_positions)
        
        # Random polar angles limited to 0 to pi/2 for z > 0
        phis = np.random.uniform(0, np.pi/2, num_positions)
        
        # Convert to Cartesian coordinates using broadcasting
        xs = r * np.sin(phis) * np.cos(thetas)
        ys = r * np.sin(phis) * np.sin(thetas)
        zs = r * np.cos(phis)
        
        return np.column_stack((xs, ys, zs))
    
    def spawn_invader_squad(self):
        num_invaders = 5
        positions = self.generate_positions(num_invaders, self.dome_radius/2)
        self.quadcopter_manager.spawn_invader(positions, "invader")
        self.num_invaders = num_invaders
        
    def spawn_pursuer_squad(self):
        num_pursuers = 1
        positions = self.generate_positions(num_pursuers, 2)
        self.quadcopter_manager.spawn_pursuer(positions, "pursuer", lidar_radius=self.dome_radius)
        self.num_pursuers = num_pursuers
        
    
            
    def get_invaders_positions(self):
        invaders = self.quadcopter_manager.get_invaders()
        return np.array([invader.inertial_data["position"] for invader in invaders])
    
    def get_pursuers_position(self):
        pursuers = self.quadcopter_manager.get_pursuers()
        return np.array([pursuer.inertial_data["position"] for pursuer in pursuers])
    
    def calculate_invader_offsets_from_pursuers(self):
        
        """
        Computes the relative distances and directions between multiple pursuers and invaders, 
        preserving the identifiers of each entity.

        This method retrieves the spatial positions of all pursuers and invaders managed by the quadcopter manager.
        Using these positions, it computes the relative distances and directional vectors between each pursuer-invader pair.

        Returns:
        - A dictionary containing the following keys:
            * "distances": A 2D numpy array with shape (n_pursuers, n_invaders). Each entry [i, j] represents the 
            Euclidean distance between the i-th pursuer and the j-th invader.
            * "directions": A 3D numpy array with shape (n_pursuers, n_invaders, 3). Each entry [i, j, :] is a unit 
            vector pointing from the i-th pursuer to the j-th invader.
            * "pursuer_ids": A list of the IDs corresponding to each pursuer.
            * "invader_ids": A list of the IDs corresponding to each invader.

        Note:
        In cases where the distance between a pursuer and an invader is zero (indicating they occupy the same position),
        the directional vector for that pair will be a zero-vector (i.e., [0, 0, 0]).
        """

        # Get invaders and pursuers positions as arrays
        invaders = self.quadcopter_manager.get_invaders()
        invaders_positions = np.array([invader.inertial_data["position"] for invader in invaders])  # Shape (n_invaders, 3)
        invader_ids = [invader.id for invader in invaders]

        pursuers = self.quadcopter_manager.get_pursuers()
        pursuers_positions = np.array([pursuer.inertial_data["position"] for pursuer in pursuers])  # Shape (n_pursuers, 3)
        pursuer_ids = [pursuer.id for pursuer in pursuers]
        pursuers_velocities = np.array([pursuer.inertial_data["velocity"] for pursuer in pursuers])  # Shape (n_pursuers, 3)

        # Calculate as before
        relative_positions = pursuers_positions[:, np.newaxis, :] - invaders_positions[np.newaxis, :, :]
        distances = np.linalg.norm(relative_positions, axis=-1)
        # Compute the mask for non-zero distances
        mask = distances != 0

        # Create a zero-initialized directions array
        directions = np.zeros_like(relative_positions)

        # Apply the mask and compute directions
        non_zero_distances = distances[mask][:, np.newaxis]
        directions[mask] = relative_positions[mask] / non_zero_distances


        return Offsets(distances, directions, pursuer_ids, invader_ids, pursuers_positions, invaders_positions, pursuers_velocities)

    def identify_captured_invaders(self, current_offsets: Offsets) -> list:
        # Find where any distance is below the CATCH_DISTANCE
        capture_mask = current_offsets.distances < self.CAPTURE_DISTANCE
        
        # Use numpy's any() function along the pursuer axis to find which invaders are captured by any pursuer
        is_captured_by_any_pursuer = np.any(capture_mask, axis=0)
        
        # Extract the invader_ids of the captured invaders
        captured_invader_ids = [invader_id for idx, invader_id in enumerate(current_offsets.invader_ids) if is_captured_by_any_pursuer[idx]]
        #self.quadcopter_manager.delete_invader_by_id(captured_invader_ids)
        return captured_invader_ids
    
    def update_offsets(self):
        self.current_offsets: Offsets = self.calculate_invader_offsets_from_pursuers()
        
        
    def update_last_offsets(self):
        self.last_offsets: Offsets = self.current_offsets
    
    def filter_last_offsets(self, current_offsets:Offsets, last_offsets: Offsets) -> Offsets:
        # Filter last_offsets to keep only invaders present in current_offsets
        common_invader_ids = set(last_offsets.invader_ids) & set(current_offsets.invader_ids)
        indices_to_keep = [last_offsets.invader_ids.index(invader_id) for invader_id in common_invader_ids]

        distances = last_offsets.distances[:, indices_to_keep]
        directions = last_offsets.directions[:, indices_to_keep]
        pursuer_ids = [last_offsets.pursuer_ids[0]]  # since pursuer_ids seems to have always one element
        invader_ids = [last_offsets.invader_ids[i] for i in indices_to_keep]
        pursuer_positions = last_offsets.pursuer_positions  # seems to be a 2D array with one row, so keep as is
        invader_positions = last_offsets.invader_positions[indices_to_keep]
        pursuer_velocities = last_offsets.pursuer_velocities  # seems to be a 2D array with one row, so keep as is

        return Offsets(distances, directions, pursuer_ids, invader_ids, pursuer_positions, invader_positions, pursuer_velocities)

    def process_captured_invaders(self):
        captured_invader_ids = self.identify_captured_invaders(self.current_offsets)
        self.quadcopter_manager.delete_invader_by_id(captured_invader_ids)
        self.num_invaders = len(self.quadcopter_manager.get_invaders())

        
    def process_invaders_in_origin(self):
        invaders_in_origin_bool_array = np.linalg.norm(self.current_offsets.invader_positions, axis=1) < self.CAPTURE_DISTANCE
        invaders_ids_to_remove = np.array(self.current_offsets.invader_ids)[invaders_in_origin_bool_array]
        self.quadcopter_manager.delete_invader_by_id(invaders_ids_to_remove.tolist())

    def update_building_life(self):
        invaders_in_origin_bool_array = np.linalg.norm(self.current_offsets.invader_positions, axis=1) < self.CAPTURE_DISTANCE
        self.building_life -= np.sum(invaders_in_origin_bool_array)
        
        
    def drive_invaders(self):
        invaders:list = self.quadcopter_manager.get_invaders()
        if not invaders:
            return
        
        # Get all invader positions as a numpy array.
        positions = np.array([invader.inertial_data["position"] for invader in invaders])
        positions = np.atleast_2d(positions)


        # Calculate the direction to drive each invader towards the origin.
        directions = -positions
        
        # Create motion commands with the direction and an intensity of 0.2 for each invader.
        intensities = 0.2 * np.ones((len(invaders), 1))
        motion_commands = np.hstack((directions, intensities))

        # Drive each invader with the corresponding motion command
        for invader, command in zip(invaders, motion_commands):
            invader.drive(command)


"""
def compute_reward(self):
        bonus = 0
        penalty = 0
        score = 0

        current_offsets:Offsets = self.current_offsets
        distances = current_offsets.distances
        last_offsets:Offsets = self.last_offsets

        # ========================================================
        #  Base Reward
        # ========================================================
        # Calculate the mean distance across all pursuers and invaders
        total_distances = np.nansum(distances)  # We use nansum to ignore NaN values
        total_invaders = np.count_nonzero(~np.isnan(distances))  # Count non-NaN distances
        mean_distance = total_distances / total_invaders if total_invaders != 0 else 0
        score = - mean_distance
            
        # ========================================================
        #  Penalty for Invaders Near the Origin
        # ========================================================
        
        near_origin_mask = current_offsets.distances < self.PROXIMITY_THRESHOLD
        invaders_near_origin = np.sum(near_origin_mask)
            
        if near_origin_mask.any():
            min_distance_near_origin = np.clip(np.min(current_offsets.distances[near_origin_mask]), 1e-10, None)
            penalty += invaders_near_origin / min_distance_near_origin

        # ========================================================
        #  Bonus for moviment antecipation
        # ========================================================
        distance_variation = last_offsets.distances - current_offsets.distances

        # Identify which pursuers have reduced their distance to any invader
        pursuers_with_positive_variation = np.any(distance_variation > 0, axis=1)

        # Index the velocities using this identification
        velocities_to_bonify = current_offsets.pursuer_velocities[pursuers_with_positive_variation]

        if velocities_to_bonify.size > 0:
            bonus += 10 * np.sum(np.linalg.norm(velocities_to_bonify, axis=1))

        # ========================================================
        #  Bonus for:
        #   - Being close to the target
        #
        #  Penalty for:
        #   - Being outside the dome
        # ========================================================

        num_invasor_captured = np.sum(current_offsets.distances < self.CAPTURE_DISTANCE)
        if num_invasor_captured > 0:
            bonus += self.MAX_REWARD * num_invasor_captured

        # Using computed value from on_step_middle
        if self.num_pursuers_outside_dome > 0:
            penalty += self.MAX_REWARD * self.num_pursuers_outside_dome

        # Using computed value from on_step_middle
        if self.invaders_in_origin > 0:
            penalty += self.MAX_REWARD * self.invaders_in_origin
                
        if self.building_life == 0:
            penalty += self.MAX_REWARD * self.max_building_life

        return score + bonus - penalty

"""