from dataclasses import dataclass
import numpy as np
from .quadcopter_manager import QuadcopterManager
from .task_progression import StageStatus, Stage
from abc import ABC

from .offsets_handler import OffsetHandler
from ....events.message_hub import MessageHub
from typing import Dict, Optional


class L3Stage1(Stage):
    def __init__(
        self,
        quadcopter_manager: QuadcopterManager,
        dome_radius: float,
        debug_on: bool = False,
    ):
        self.dome_radius = dome_radius
        self.quadcopter_manager = quadcopter_manager
        self.debug_on = debug_on

        self.init_constants()
        self.init_globals()

        self.offset_handler = OffsetHandler(quadcopter_manager, self.dome_radius)

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic="SimulationStep", subscriber=self._subscriber_simulation_step
        )

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)

    def init_constants(self):
        self.MAX_REWARD = 1000
        self.PROXIMITY_THRESHOLD = 2
        self.PROXIMITY_PENALTY = 1000
        self.CAPTURE_DISTANCE = 0.2

        self.INVADER_EXPLOSION_RANGE = 0.2
        self.PURSUER_SHOOT_RANGE = 1

        self.MAX_STEP = 300  # 300 calls = 20 seconds for rl_frequency = 15
        # STEP_PENALTY_SMOOTHING ensures that at the end of the episode the accumulated penalty is about MAX_REWARD
        self.STEP_PENALTY_SMOOTHING = (
            2 * self.MAX_REWARD / (self.MAX_STEP * (self.MAX_STEP + 1))
        )

    def init_globals(self):
        self.current_step = 0
        # self.MAX_BUILDING_LIFE = 3
        # self.building_life = self.MAX_BUILDING_LIFE

    # ===============================================================================
    # On Notification
    # ===============================================================================

    # ===============================================================================
    # On Calls
    # ===============================================================================

    def on_env_init(self):
        self._stage_status = StageStatus.RUNNING
        # print("Spawning invaders and pursuers")
        self.spawn_invader_squad()
        self.spawn_pursuer_squad()

    def on_reset(self):
        # print("On Reset")
        self.on_episode_end()
        self.on_episode_start()

    def on_episode_start(self):
        # print("\n\nOn Episode Start\n\n")
        self.quadcopter_manager.arm_all()
        self.replace_invaders()
        self.replace_pursuers()
        self.offset_handler.on_episode_start()
        # print("episode started")

    def on_episode_end(self):
        self.init_globals()

    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """
        self.quadcopter_manager.drive_invaders()

    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """

        self.offset_handler.on_middle_step()
        # self.update_building_life()

        successful_shots, pursuers_exploded = self.process_nearby_invaders()
        self.process_invaders_in_origin()
        reward = self.compute_reward(successful_shots, pursuers_exploded)

        termination = self.compute_termination()
        return reward, termination

    def process_nearby_invaders(self):
        successful_shots = 0
        pursuers_exploded = 0

        successful_shots += self.process_shoot_range_invaders()
        pursuers_exploded += self.process_explosion_range_invaders()
        return successful_shots, pursuers_exploded

    def process_explosion_range_invaders(self):
        identified_invaders_nearby = self.offset_handler.identify_invaders_in_range(
            self.INVADER_EXPLOSION_RANGE
        )
        explosion = 0
        to_disarm: list = []
        for pursuer_id in list(identified_invaders_nearby.keys()):
            to_disarm.append(pursuer_id)
            to_disarm.append(identified_invaders_nearby[pursuer_id][0])

            self.quadcopter_manager.disarm_by_ids(to_disarm)
            explosion += 1

        return explosion

    def process_shoot_range_invaders(self):
        identified_invaders_nearby = self.offset_handler.identify_invaders_in_range(
            self.PURSUER_SHOOT_RANGE
        )
        successful_shots = 0

        for pursuer_id in list(identified_invaders_nearby.keys()):
            were_successful = self.quadcopter_manager.shoot_by_ids(
                pursuer_id, identified_invaders_nearby[pursuer_id][0]
            )

            if were_successful:
                successful_shots += 1

        return successful_shots

    def on_step_end(self):
        self.offset_handler.on_end_step()

    @property
    def status(self):
        return self._stage_status

    # ===============================================================================
    # Reward and Termination
    # ===============================================================================

    def compute_reward(self, successful_shots: int = 0, pursuers_exploded: int = 0):
        # Initialize reward components
        score, bonus, penalty = 0, 0, 0

        # Calculate the current and last closest distances
        current_closest_distance: int = (
            self.offset_handler.current_closest_pursuer_to_invader_distance
        )
        last_closest_distance: int = (
            self.offset_handler.last_closest_pursuer_to_invader_distance
        )

        # =======================================================================
        # Calculate Base Score
        #
        # Ideal Distance: Define this as the shoot range minus a small buffer (e.g., shoot range - 0.1). This gives the pursuer some leeway to maneuver without getting too close to the invader.
        # 1. Reward for Being Near the Ideal Distance: Provide a positive reward when the pursuer is near this ideal distance. The closer it is to the ideal distance, the higher the reward.
        # 2. Penalty for Being Too Far: If the pursuer is farther than the shoot range, it should receive a negative reward proportional to the excess distance.
        # 3. High Penalty for Getting Too Close: Since getting closer than 0.2 units results in destruction, the penalty should be significantly higher as the pursuer approaches this limit. A high value for
        # γ makes sense here.
        # =======================================================================

        variation = 0.2
        alpha = 1
        beta = 5
        gamma = 5
        # TODO: Ajustar a função abaixo de tal forma que a torne continua. A transição entre as duas últimas estão contínuas, menos a da primeira com a da segunda.
        if current_closest_distance > self.PURSUER_SHOOT_RANGE + variation:
            score = alpha * (
                self.PURSUER_SHOOT_RANGE + variation - current_closest_distance
            )

        elif current_closest_distance >= self.PURSUER_SHOOT_RANGE - variation:
            score = beta + self.PURSUER_SHOOT_RANGE - current_closest_distance

        else:
            score = (beta + variation) - np.exp(
                -gamma
                * (current_closest_distance - (self.PURSUER_SHOOT_RANGE - variation))
            )

        # =======================================================================
        # Bonuses
        # 1. Bonus for Getting Closer: If the pursuer gets closer to the invader, it should receive a positive reward proportional to the distance it has moved.
        # 2. Bonus for Capturing Invaders: If the pursuer captures an invader, it should receive a large positive reward.
        # =======================================================================

        # Bonus for getting closer to the target
        if current_closest_distance < last_closest_distance:
            bonus += -(current_closest_distance - last_closest_distance)

        # Bonus for capturing invaders
        bonus += self.MAX_REWARD * successful_shots

        # =======================================================================
        # Penalties
        # 1. Penalty for lost a pursuer: If the pursuer is destroyed, it should receive a large negative reward.
        # 2. Penalty for pursuers outside the dome: If the pursuer is outside the dome, it should receive a large negative reward.
        # 3. Penalty for touching the gound: If the pursuer touches the ground, it should receive a large negative reward.
        # 4. Penalty for time passing
        # =======================================================================

        penalty += self.MAX_REWARD * pursuers_exploded

        # Penalty for pursuers outside the dome
        num_pursuer_outside_dome = len(
            self.offset_handler.identify_pursuer_outside_dome()
        )
        if num_pursuer_outside_dome > 0:
            penalty += self.MAX_REWARD

        # Penalty for building life deviation
        # if self.building_life < self.max_building_life:
        #    penalty += self.MAX_REWARD * (self.max_building_life - self.building_life)

        pusuers_touched_ground = self.offset_handler.identify_pursuer_touched_ground()
        if len(pusuers_touched_ground) > 0:
            penalty += self.MAX_REWARD * len(pusuers_touched_ground)

        theta = 0.025  # this theta and the limit of 300 steps ensures that the max penalty at the end of episode is about 1000
        penalty += theta * self.current_step
        return score + bonus - penalty

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""
        # Step count exceeds maximum.
        if self.current_step > self.MAX_STEP:
            self._stage_status = StageStatus.FAILURE
            return True

        # Building life depleted.
        # if self.building_life == 0:
        #    self._stage_status = StageStatus.FAILURE
        #    return True

        # Using computed value from on_step_middle
        num_pursuers_outside_dome = len(
            self.offset_handler.identify_pursuer_outside_dome()
        )
        if num_pursuers_outside_dome > 0:
            self._stage_status = StageStatus.FAILURE
            return True

        num_invaders_outside_dome = len(
            self.offset_handler.identify_invader_outside_dome()
        )
        if num_invaders_outside_dome > 0:
            self._stage_status = StageStatus.FAILURE
            return True

        # All invaders captured.
        if len(self.quadcopter_manager.get_armed_invaders()) == 0:
            self._stage_status = StageStatus.SUCCESS
            return True

        # All pursuers destroyed.
        if len(self.quadcopter_manager.get_armed_pursuers()) == 0:
            # print("All pursuers destroyed")
            self._stage_status = StageStatus.FAILURE
            return True

        if len(self.offset_handler.identify_pursuer_touched_ground()) > 0:
            self._stage_status = StageStatus.FAILURE
            return True

        if len(self.offset_handler.identify_invader_touched_ground()) > 0:
            self._stage_status = StageStatus.FAILURE
            return True
        # If none of the above conditions met, return False
        return False

    # ===============================================================================
    # Others Implementations
    # ===============================================================================

    def generate_positions(self, num_positions: int, r: float) -> np.ndarray:
        # Random azimuthal angles limited to 0 to pi for x > 0
        thetas = np.random.uniform(0, np.pi, num_positions)

        # Random polar angles limited to 0 to pi/2 for z > 0
        phis = np.random.uniform(0, np.pi / 2, num_positions)

        # Convert to Cartesian coordinates using broadcasting
        xs = r * np.sin(phis) * np.cos(thetas)
        ys = r * np.sin(phis) * np.sin(thetas)
        zs = r * np.cos(phis)

        return np.column_stack((xs, ys, zs))

    def replace_invaders(self):
        # self.quadcopter_manager.disarm_all()
        invaders = self.quadcopter_manager.get_all_invaders()
        positions = self.generate_positions(len(invaders), self.dome_radius / 2)
        self.quadcopter_manager.replace_quadcopters(invaders, positions)

    def replace_pursuers(self):
        # self.quadcopter_manager.disarm_all()
        pursuers = self.quadcopter_manager.get_all_pursuers()
        positions = self.generate_positions(len(pursuers), 1)
        self.quadcopter_manager.replace_quadcopters(pursuers, positions)

    def spawn_invader_squad(self):
        num_invaders = 2
        positions = self.generate_positions(num_invaders, self.dome_radius)  # /2
        self.quadcopter_manager.spawn_invader(positions, "invader")

    def spawn_pursuer_squad(self):
        num_pursuers = 1
        positions = self.generate_positions(num_pursuers, 2)
        self.quadcopter_manager.spawn_pursuer(
            positions, "pursuer", lidar_radius=self.dome_radius
        )

    def get_invaders_positions(self):
        invaders = self.quadcopter_manager.get_all_invaders()
        return np.array([invader.inertial_data["position"] for invader in invaders])

    def get_pursuers_position(self):
        pursuers = self.quadcopter_manager.get_all_pursuers()
        return np.array([pursuer.inertial_data["position"] for pursuer in pursuers])

    def process_invaders_in_origin(self):
        invaders_ids_to_remove = self.offset_handler.identify_invaders_in_origin()

        self.quadcopter_manager.disarm_by_ids(invaders_ids_to_remove)

    def update_building_life(self):
        count = len(self.offset_handler.identify_invaders_in_origin())
        # self.building_life -= np.sum(count)
