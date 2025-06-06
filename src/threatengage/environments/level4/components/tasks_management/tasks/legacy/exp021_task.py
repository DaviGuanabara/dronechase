import numpy as np
from abc import ABC
from typing import Dict, Optional
from dataclasses import dataclass
import math
import random
from ...task_progression import TaskStatus, Task

from ....entities_management.entities_manager import EntitiesManager, Quadcopter
from ....entities_management.offsets_handler import OffsetHandler

from ......notification_system.message_hub import MessageHub
from ......notification_system.topics_enum import Topics_Enum
from ......entities.immovable_structures.immovable_structures import ImmovableStructures

from typing import Optional, Union, NamedTuple, Dict, List, cast
from ......entities.navigators.loitering_munition_navigator import KamikazeNavigator
from ......entities.navigators.loyalwingman_navigator import LoyalWingmanBehaviorTree


class Exp021_Task(Task):
    def __init__(
        self,
        # simulation,
        entities_manager: EntitiesManager,
        dome_radius: float,
        # debug_on: bool = False,
        building_position: np.ndarray = np.array([0, 0, 0.1]),  # type: ignore
    ):
        print("Exp021_Task init")

        self.dome_radius = dome_radius
        self.entities_manager = entities_manager

        self.init_constants()
        self.init_globals()

        self.offset_handler = OffsetHandler(
            self.dome_radius,
            entities_manager,
        )

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic=Topics_Enum.AGENT_STEP_BROADCAST.value,
            subscriber=self._subscriber_simulation_step,
        )

        self.kamikaze_navigator = KamikazeNavigator(building_position)
        self.loyalwingman_navigator = LoyalWingmanBehaviorTree(building_position)

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)

    def init_constants(self):
        self.NUM_PURSUERS = 1
        self.munition_per_defender = 20
        # self.MAX_NUMBER_OF_ROUNDS = 1  # self.calculate_rounds(
        # self.NUM_PURSUERS, self.munition_per_defender
        # )

        self.MAX_NUMBER_OF_INVADERS = 1
        self.MAX_NUMBER_OF_ROUNDS = 20
        self.NUM_INVADERS = self.MAX_NUMBER_OF_INVADERS  # self.MAX_NUMBER_OF_ROUNDS
        # self.NUM_INVADERS = 1

        self.current_round = 1
        self.building_position = np.array([0, 0, 1])

        self.MAX_REWARD = 1000
        self.PROXIMITY_THRESHOLD = 2
        self.PROXIMITY_PENALTY = 1000
        self.CAPTURE_DISTANCE = 0.2

        self.INVADER_EXPLOSION_RANGE = 0.2
        self.PURSUER_SHOOT_RANGE = 1

        self.STEP_INCREMENT = 300
        self.MAX_STEP = 300  # 300 calls = 20 seconds for rl_frequency = 15
        # STEP_PENALTY_SMOOTHING ensures that at the end of the episode the accumulated penalty is about MAX_REWARD
        self.STEP_PENALTY_SMOOTHING = (
            2 * self.MAX_REWARD / (self.MAX_STEP * (self.MAX_STEP + 1))
        )

    def init_globals(self):
        self.current_step = 0
        self.MAX_BUILDING_LIFE = 1
        self.building_life = self.MAX_BUILDING_LIFE

        self.kills = 0
        self.deads = 0
        self.is_building_destroyed = False
        # self.is_building_destroyed = False

        # self.MAX_BUILDING_LIFE = 3
        # self.building_life = self.MAX_BUILDING_LIFE

    # ===============================================================================
    # Rounds Management
    # ===============================================================================
    # TODO: [ ] i have to update the termination function
    # TODO: [ ] i have to verify if reward is being calculated correctly
    # TODO: [ ] i have to verify if the invaders are being created to arrive from any place.
    # TODO: [ ] i have to make a function to count the invaders killed.

    # TODO: [ ] Change the name rounds to waves
    def is_current_round_over(self) -> bool:
        return len(self.entities_manager.get_armed_invaders()) == 0

    def is_last_round(self) -> bool:
        return self.current_round >= self.MAX_NUMBER_OF_ROUNDS

    def is_all_rounds_over(self) -> bool:
        # print("test if all rounds is completed")
        return self.is_current_round_over() and self.is_last_round()

    def reset_rounds(self):
        self.current_round = 1
        # self.setup_round(self.current_round)

    def advance_round(self):
        self.MAX_STEP += self.STEP_INCREMENT
        self.current_round += (
            1
            if self.current_round < self.MAX_NUMBER_OF_ROUNDS
            else self.MAX_NUMBER_OF_ROUNDS
        )

        print(f"Round {self.current_round} of {self.MAX_NUMBER_OF_ROUNDS} Started")
        # self.entities_manager.disarm_all()
        # self.entities_manager.arm_all_pursuers()
        # self.entities_manager.disarm_all_invaders()

        self.setup_round(self.current_round)
        # print(f"Round {self.current_round} of {self.MAX_NUMBER_OF_ROUNDS} Started")
        # print(self.entities_manager.get_armed_invaders())
        # print(self.entities_manager.get_armed_pursuers())

        # print(
        #    f"Round {self.current_round} of {self.MAX_NUMBER_OF_ROUNDS} Started, {len(self.entities_manager.get_armed_invaders())} invaders and {len(self.entities_manager.get_armed_pursuers())} pursuers"
        # )
        self.offset_handler.on_episode_start()
        self.kamikaze_navigator.reset()
        self.loyalwingman_navigator.reset()

    def get_round(self) -> int:
        return self.current_round

    def setup_round(self, current_round):
        # invaders = self.entities_manager.get_armed_invaders()
        # for invader in invaders:
        #    self.entities_manager.disarm_by_quadcopter(invader)

        self.entities_manager.disarm_all_invaders()

        invaders = self.entities_manager.get_all_invaders()
        num_invaders_to_arm = (
            current_round
            if current_round <= self.MAX_NUMBER_OF_INVADERS
            else self.MAX_NUMBER_OF_INVADERS
        )
        for i in range(num_invaders_to_arm):
            # TODO: Find a better way to do this
            positions = self.generate_positions(1, 3, 2, 0.2)
            self.entities_manager.replace_quadcopter(invaders[i], positions[0])
            self.entities_manager.arm_by_quadcopter(invaders[i])

    def calculate_rounds(self, num_defenders, munition_per_defender):
        """
        Calculate the number of episodes based on the number of defenders and the munition each defender has.

        Args:
        num_defenders (int): The number of defenders.
        munition_per_defender (int): The munition each defender has.

        Returns:
        int: The number of episodes that can be created.
        """
        # Total munitions available across all defenders
        total_munitions = num_defenders * munition_per_defender

        # Solving the quadratic equation n^2 + n - 2 * total_munitions = 0
        a = 1
        b = 1
        c = -2 * total_munitions

        # Calculate the discriminant
        discriminant = b**2 - 4 * a * c

        # Calculate the two solutions
        sol2 = (-b + math.sqrt(discriminant)) / (2 * a)

        # We need the positive root and floor it since episodes are discrete
        n_episodes = math.ceil(sol2)

        return n_episodes

    # ===============================================================================
    # On Notification
    # ===============================================================================

    def drive_invaders(self):
        invaders: List[Quadcopter] = self.entities_manager.get_armed_invaders()

        for invader in invaders:
            # self.kamikaze_navigator.update(invader, self.offset_handler)
            invader.drive(np.array([0, 0, 0, 0.4]))
            position = invader.inertial_data["position"]
            # print(f"invader position: {position}")

    # TODO: para ter RL agent + BT, eu posso só pular o primeiro pursuer e deixar o resto como BT.
    def drive_loyalwingmen(self):
        pursuers: List[Quadcopter] = self.entities_manager.get_armed_pursuers()

        # Skip the first pursuer in the list
        for pursuer in pursuers[1:]:
            # print(f"loyalwingman: {pursuer.id}")
            self.loyalwingman_navigator.update(pursuer, self.offset_handler)

    # ===============================================================================
    # On Calls
    # ===============================================================================

    def on_env_init(self):
        self._stage_status = TaskStatus.RUNNING
        # print("Spawning invaders and pursuers")
        self.spawn_invader_squad()
        self.spawn_pursuer_squad()

    def on_reset(self):
        # print("On Reset")
        self.on_episode_end()
        self.on_episode_start()

    def on_episode_start(self):
        # print("\nOn Episode Start\n\n")
        # self.entities_manager.arm_all()
        # self.replace_invaders()

        self.replace_pursuers()

        self.reset_rounds()
        self.setup_round(self.current_round)
        self.entities_manager.arm_all_pursuers()
        self.offset_handler.on_episode_start()

        self.kamikaze_navigator.reset()
        self.loyalwingman_navigator.reset()

        # print("episode started")

    def on_episode_end(self):
        # print("On Episode End")
        self.init_globals()
        self.entities_manager.disarm_all()

    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """
        # print("on step start")
        self.drive_invaders()
        # self.drive_loyalwingmen()
        pass

    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """

        # print("on step middle")
        self.offset_handler.on_middle_step()
        self.update_building_life()

        successful_shots, pursuers_exploded = self.process_nearby_invaders()

        self.kills += successful_shots
        self.deads += pursuers_exploded

        self.process_invaders_in_origin()
        reward = self.compute_reward(successful_shots, pursuers_exploded)

        termination = self.compute_termination()
        return reward, termination

    def on_step_end(self):
        # print("on step end")
        if self.is_all_rounds_over():
            # print("(on step end) All rounds completed")
            return

        self.offset_handler.on_end_step()
        if (
            self.is_current_round_over()
            and len(self.entities_manager.get_armed_pursuers()) > 0
        ):
            # print(f"Round {self.current_round} of {self.MAX_NUMBER_OF_ROUNDS} Over")
            self.advance_round()

    def process_nearby_invaders(self):
        successful_shots = 0
        pursuers_exploded = 0

        successful_shots += self.process_shoot_range_invaders()
        # self.Enemy_Engagement_Success += successful_shots

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

            self.entities_manager.disarm_by_ids(to_disarm)
            explosion += 1

        return explosion

    def process_shoot_range_invaders(self):
        identified_invaders_nearby = self.offset_handler.identify_invaders_in_range(
            self.PURSUER_SHOOT_RANGE
        )
        successful_shots = 0

        for pursuer_id in list(identified_invaders_nearby.keys()):
            were_successful = self.entities_manager.shoot_by_ids(
                pursuer_id, identified_invaders_nearby[pursuer_id][0]
            )

            if were_successful:
                successful_shots += 1

        return successful_shots

    @property
    def status(self):
        return self._stage_status

    # ===============================================================================
    # Reward, Termination and INFO
    # ===============================================================================

    def compute_reward(self, successful_shots: int = 0, pursuers_exploded: int = 0):
        # Initialize reward components
        score, bonus, penalty = 0, 0, 0
        pursuers = self.entities_manager.get_armed_pursuers()

        if len(pursuers) == 0:
            return -self.MAX_REWARD

        pursuer = pursuers[0]
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

        score = self.kills - current_closest_distance
        # print(f"score: {score}")

        # =======================================================================
        # Bonuses
        # 1. Bonus for Getting Closer: If the pursuer gets closer to the invader, it should receive a positive reward proportional to the distance it has moved.
        # 2. Bonus for Capturing Invaders: If the pursuer captures an invader, it should receive a large positive reward.
        # =======================================================================

        # O que acontece se o pursuer tiver recarregando ?
        # Ele ainda vai ser recompensado por se aproximar do alvo ?
        if current_closest_distance < last_closest_distance:
            velocity = pursuer.inertial_data.get("velocity", np.zeros(3))
            bonus += 10 * np.linalg.norm(velocity)

        # Bonus for capturing invaders
        if successful_shots > 0:
            bonus += (
                self.MAX_REWARD * successful_shots
            ) + self.kills * self.MAX_REWARD / 10
            # print(
            #    f"bonus for successful shots: {self.MAX_REWARD * successful_shots + self.kills * self.MAX_REWARD / 10}"
            # )

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
        if self.building_life < self.MAX_BUILDING_LIFE:
            penalty += self.MAX_REWARD * (self.MAX_BUILDING_LIFE - self.building_life)

        # pusuers_touched_ground = self.offset_handler.identify_pursuer_touched_ground()
        # if len(pusuers_touched_ground) > 0:
        #    penalty += self.MAX_REWARD * len(pusuers_touched_ground)

        # theta = 0.025  # this theta and the limit of 300 steps ensures that the max penalty at the end of episode is about 1000
        # penalty += theta * self.current_step
        return score + bonus - penalty

    def compute_termination(self) -> bool:
        """Calculate if the simulation is done."""
        # Step count exceeds maximum.
        if self.current_step > self.MAX_STEP:
            self._stage_status = TaskStatus.FAILURE
            return True

        if self.is_all_rounds_over():
            # print("(compute termination) All rounds completed")
            self._stage_status = TaskStatus.SUCCESS
            return True

        # Building life depleted.
        if self.building_life == 0:
            # print("Building life depleted")
            self.is_building_destroyed = True
            self._stage_status = TaskStatus.FAILURE
            return True

        # Using computed value from on_step_middle
        num_pursuers_outside_dome = len(
            self.offset_handler.identify_pursuer_outside_dome()
        )
        if num_pursuers_outside_dome > 0:
            self._stage_status = TaskStatus.FAILURE
            return True

        num_invaders_outside_dome = len(
            self.offset_handler.identify_invader_outside_dome()
        )
        if num_invaders_outside_dome > 0:
            self._stage_status = TaskStatus.FAILURE
            return True

        # All invaders captured.
        # if len(self.entities_manager.get_armed_invaders()) == 0:
        #    self._stage_status = TaskStatus.SUCCESS
        #    return True

        # All pursuers destroyed.
        if len(self.entities_manager.get_armed_pursuers()) == 0:
            # print("All pursuers destroyed")
            self._stage_status = TaskStatus.FAILURE
            return True

        # if len(self.offset_handler.identify_pursuer_touched_ground()) > 0:
        #    self._stage_status = StageStatus.FAILURE
        #    return True

        # if len(self.offset_handler.identify_invader_touched_ground()) > 0:
        #    self._stage_status = StageStatus.FAILURE
        #    return True
        # If none of the above conditions met, return False
        return False

    def compute_info(self):
        return {
            "kills": self.kills,
            "deads": self.deads,
            "is_building_destroyed": self.is_building_destroyed,
            "current_wave": self.current_round,
        }

    # ===============================================================================
    # Others Implementations
    # ===============================================================================

    import numpy as np

    def generate_positions(
        self, num_positions: int, r_max: int, r_min: int = 5, z_min: float = 2
    ) -> np.ndarray:
        # Adjust r_min and r_max to ensure r_min is not greater than r_max
        r_min = min(r_min, r_max)
        r_max = max(r_min, r_max)

        # Adjust z_min to be less than or equal to r_min
        z_min = min(z_min, r_min)

        # Random azimuthal angles from 0 to 2pi for full coverage around the Z-axis
        thetas = np.random.uniform(0, 2 * np.pi, num_positions)

        # Vectorized random radius selection for each position
        rs = np.random.uniform(r_min, r_max, num_positions)

        # Calculate minimum phi for each position based on z_min and radius
        phi_min = np.arccos(z_min / rs)

        # Random polar angles from phi_min to pi/2 for z >= z_min
        phis = np.random.uniform(phi_min, np.pi / 2, num_positions)

        # Convert to Cartesian coordinates using broadcasting
        xs = rs * np.sin(phis) * np.cos(thetas)
        ys = rs * np.sin(phis) * np.sin(thetas)
        zs = rs * np.cos(phis)

        return np.column_stack((xs, ys, zs))

    # def replace_invaders(self):
    #    invaders = self.entities_manager.get_all_invaders()
    #    positions = self.generate_positions(len(invaders), 7, 5, 3)
    #    self.entities_manager.replace_quadcopters(invaders, positions)

    def replace_pursuers(self):
        pursuers = self.entities_manager.get_all_pursuers()
        positions = self.generate_positions(
            self.NUM_PURSUERS, 1, 1, 1
        )  # np.array([[0, 0, 2]])
        self.entities_manager.replace_quadcopters(pursuers, positions)

    def spawn_invader_squad(self):
        positions = self.generate_positions(self.NUM_INVADERS, 3, 2, 0.2)
        invaders: List[Quadcopter] = self.entities_manager.spawn_invader(
            positions, "invader"
        )

        for invader in invaders[1:]:
            self.entities_manager.disarm_by_quadcopter(invader)

    def spawn_pursuer_squad(self):
        # num_pursuers = 1
        positions = self.generate_positions(self.NUM_PURSUERS, 1, 1, 1)
        # np.array([[0, 5, 2]])  # self.generate_positions(self.NUM_PURSUERS, 2)

        self.pursuers_positions = positions
        pursuers = self.entities_manager.spawn_pursuer(
            positions, "pursuer", lidar_radius=2 * self.dome_radius
        )

        for pursuer in pursuers:
            pursuer.set_munition(self.munition_per_defender)

    def get_invaders_positions(self):
        invaders = self.entities_manager.get_all_invaders()
        return np.array([invader.inertial_data["position"] for invader in invaders])

    def get_pursuers_position(self):
        pursuers = self.entities_manager.get_all_pursuers()
        return np.array([pursuer.inertial_data["position"] for pursuer in pursuers])

    def process_invaders_in_origin(self):
        invaders_ids_to_remove = self.offset_handler.identify_invaders_in_origin()

        self.entities_manager.disarm_by_ids(invaders_ids_to_remove)

    def update_building_life(self):
        count = len(self.offset_handler.identify_invaders_in_origin())
        self.building_life -= np.sum(count)
