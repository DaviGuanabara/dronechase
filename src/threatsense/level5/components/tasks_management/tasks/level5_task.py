import numpy as np
from abc import ABC
from typing import Dict, Optional
from dataclasses import dataclass
import math


from threatsense.level5.components.normalization import normalize_inertial_data
from threatsense.level5.components.tasks_management.task_progression import TaskStatus, Task
from threatsense.level5.components.entities_manager import EntitiesManager, Quadcopter
from core.context.offsets_handler import OffsetHandler

from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import TopicsEnum
from core.entities.immovable_structures.immovable_structures import ImmovableStructures

# Problema aqui. LoyalwingmanBehaviorTree vem do core, que aponta pro level4.
from core.entities.navigators.loyalwingman_navigator import LoyalWingmanBehaviorTree
from core.entities.navigators.loitering_munition_navigator_air_combat_only import (
    KamikazeNavigator as KamikazeNavigator_KamikazeNavigator_Air_Combat_Only,
)

from typing import Optional, Union, NamedTuple, Dict, List, cast


from typing import Tuple


print("[ThreatSense] Importing Level 5 Task...")


class Level5_Task(Task):
    def __init__(
        self,
        # simulation,
        entities_manager: EntitiesManager,
        dome_radius: float,
        # debug_on: bool = False,
        building_position: np.ndarray = np.array([0, 0, 0.1]),
        use_fused_lidar=False
    ):
        print("Level 5 - Task Init")

        self.dome_radius = dome_radius
        self.entities_manager = entities_manager
        self.use_fused_lidar = use_fused_lidar

        print("init constants and globals")
        self.init_constants()
        self.init_globals()

        self.offset_handler = OffsetHandler(
            self.dome_radius,
            entities_manager,
        )

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic=TopicsEnum.AGENT_STEP_BROADCAST,
            subscriber=self._subscriber_simulation_step,
        )

        self.kamikaze_navigator = KamikazeNavigator_KamikazeNavigator_Air_Combat_Only(
            building_position
        )
        self.loyalwingman_navigator = LoyalWingmanBehaviorTree(
            building_position)

        print(f"[DEBUG] Level5_Task init - NUM_PURSUERS = {self.NUM_PURSUERS}")

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)

    def init_constants(self):
        self.NUM_PURSUERS = 6
        self.MUNITION_PER_DEFENDER = 20
        self.NUM_INVADERS = 12  # 👈 pode ser > MAX_NUMBER_OF_ROUNDS

        self.MAX_REWARD = 1000

        self.ENEMY_BORN_RADIUS = 6
        self.INVADER_EXPLOSION_RANGE = 0.2
        self.PURSUER_SHOOT_RANGE = 1

        self.STEP_INCREMENT = 100
        self.MAX_STEP = 300

        # Round inicial arbitrário, pode ser >= 1
        self.INITIAL_ROUND = 1
        # Total máximo de rounds possíveis, baseados em munição
        self.MAX_NUMBER_OF_ROUNDS = self.calculate_max_rounds(
            self.NUM_PURSUERS, self.MUNITION_PER_DEFENDER, self.NUM_INVADERS)
        # Número de inimigos pode ser arbitrário (ex: para gerar oclusão, complexidade)

        print(f"[INIT] MAX_NUMBER_OF_ROUNDS: {self.MAX_NUMBER_OF_ROUNDS}")
        print(f"[INIT] NUM_INVADERS: {self.NUM_INVADERS}")

    def init_globals(self):
        self.current_step = 0
        self.current_round = self.INITIAL_ROUND
        # self.building_life = self.MAX_BUILDING_LIFE

        self.agent_kills = 0
        self.allies_kills = 0
        self.deads = 0
        self.is_building_destroyed = False

        self.last_closest_distance = self.dome_radius
        self.current_closest_distance = self.dome_radius

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
        self.current_round = self.INITIAL_ROUND
        self.setup_round(self.current_round)

    def increment_max_step(self, successful_shots: int):
        if successful_shots > 0:
            self.MAX_STEP += self.STEP_INCREMENT
            # print("new max step:", self.MAX_STEP)

    def advance_round(self):
        self.current_round += (
            1
            if self.current_round < self.MAX_NUMBER_OF_ROUNDS
            else self.MAX_NUMBER_OF_ROUNDS
        )
        # self.entities_manager.disarm_all()
        # self.entities_manager.arm_all_pursuers()
        # self.entities_manager.disarm_all_invaders()

        self.setup_round(self.current_round)
        self._extracted_from_on_episode_start_19()

    def get_round(self) -> int:
        return self.current_round

    def setup_round(self, current_round):
        # invaders = self.entities_manager.get_armed_invaders()
        # for invader in invaders:
        #    self.entities_manager.disarm_by_quadcopter(invader)

        self.entities_manager.disarm_all_invaders()

        invaders = self.entities_manager.get_all_invaders()
        # print(current_round, len(invaders))

        positions = self.generate_positions(
            current_round, self.ENEMY_BORN_RADIUS)
        for i in range(current_round):
            # TODO: Find a better way to do this
            # positions = self.generate_positions(1, 6)

            self.entities_manager.replace_quadcopter(invaders[i], positions[i])
            self.entities_manager.arm_by_quadcopter(invaders[i])

    def calculate_max_rounds(self, num_pursuers: int, munition_per_defender: int, initial_invaders: int) -> int:
        """
        Calculates the maximum number of rounds given:

        - The number of defenders (pursuers),
        - The amount of ammunition per defender,
        - The initial number of invaders in round 1.

        The number of invaders increases incrementally with each round:
            Round 1: a
            Round 2: a + 1
            Round 3: a + 2
            ...
            Round R: a + (R - 1)

        The total number of invaders over R rounds is:
            S = (R / 2) * [2a + (R - 1)]

        Let:
            R     = number of rounds,
            a     = number of invaders in the first round,
            T_ab  = total number of possible kills = num_pursuers * munition_per_defender + 1

        Then we have the inequality:
            R * (2a + R - 1) <= 2 * T_ab
            => R² + (2a - 1)R - 2T_ab <= 0

        Solving the quadratic equation:
            Δ = (2a - 1)² + 8 * T_ab
            R = [-(2a - 1) + sqrt(Δ)] / 2

        Only the positive root is considered since the number of rounds must be positive.

        Parameters:
        ----------
        num_pursuers : int
            Total number of allied drones (defenders).
        munition_per_defender : int
            Ammunition available per pursuer.
        initial_invaders : int
            Number of invaders in the first round.

        Returns:
        -------
        int
            The maximum number of possible rounds (rounded down).
        """
        total_abates = num_pursuers * munition_per_defender + \
            1  # +1 to account for the suicidal behavior of the agent
        a = initial_invaders

        b = 2 * a - 1
        delta = b**2 + 8 * total_abates

        sol = (-b + math.sqrt(delta)) / 2
        return math.ceil(sol)

    # ===============================================================================
    # On Notification
    # ===============================================================================

    def drive_invaders(self):
        invaders: List[Quadcopter] = self.entities_manager.get_armed_invaders()

        for invader in invaders:
            self.kamikaze_navigator.update(invader, self.offset_handler)

    def drive_loyalwingmen(self):
        allies = self.entities_manager.get_allies(armed=True)

        for ally in allies:
            self.loyalwingman_navigator.update(ally, self.offset_handler)

    # ===============================================================================
    # On Calls
    # ===============================================================================

    def on_env_init(self):
        self._stage_status = TaskStatus.RUNNING
        print("level 5_Task - on_env_init - Spawning invaders and pursuers")
        self.spawn_invader_squad()
        self.spawn_pursuer_squad()

    def on_reset(self):

        print("level 5 task - on reset")
        self.on_episode_end()
        self.on_episode_start()

    def on_episode_start(self):
        self.reset_rounds()
        # TODO: o reset pursuer est'a muito longe do reset invaders. Isso remove uma certa ordem no c'odigo

        self.entities_manager.arm_all_pursuers()
        self.replace_pursuers()
        self._extracted_from_on_episode_start_19()

    # TODO Rename this here and in `advance_round` and `on_episode_start`
    def _extracted_from_on_episode_start_19(self):
        self.offset_handler.on_episode_start()
        self.kamikaze_navigator.reset()
        self.loyalwingman_navigator.reset()

    def on_episode_end(self):
        self.init_constants()
        self.init_globals()

        self.entities_manager.disarm_all()

    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """
        # print("on step start")
        self.drive_invaders()
        self.drive_loyalwingmen()

    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """

        # print("on step middle")
        self.offset_handler.on_middle_step()
        # self.update_building_life()

        (
            successful_rl_agent_shots,
            successful_allies_shots,
            pursuers_exploded,
            pursuer_suicided,
            agent_suicided,
        ) = self.process_nearby_invaders()

        self.agent_kills += successful_rl_agent_shots
        self.allies_kills += successful_allies_shots
        self.deads += pursuers_exploded

        self.process_invaders_in_origin()
        reward = self.compute_reward(
            successful_rl_agent_shots,
            successful_allies_shots,
            pursuers_exploded,
            pursuer_suicided,
            agent_suicided,
        )

        self.increment_max_step(
            successful_rl_agent_shots + successful_allies_shots)
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
        # successful_rl_agent_shots = 0
        # successful_allies_shots = 0
        # pursuers_exploded = 0

        (
            successful_rl_agent_shots,
            successful_allies_shots,
        ) = self.process_shoot_range_invaders()
        # self.Enemy_Engagement_Success += successful_shots

        (
            pursuers_exploded,
            pursuer_suicided,
            agent_suicided,
        ) = self.process_explosion_range_invaders()
        return (
            successful_rl_agent_shots,
            successful_allies_shots,
            pursuers_exploded,
            pursuer_suicided,
            agent_suicided,
        )

    def process_explosion_range_invaders(self) -> Tuple[int, int, int]:
        identified_invaders_nearby = self.offset_handler.identify_invaders_in_range(
            self.INVADER_EXPLOSION_RANGE
        )

        explosion = 0
        pursuer_suicided = 0
        agent_suicided = 0

        agent_id = self.entities_manager.agent_id

        to_disarm: list = []
        for pursuer_id in list(identified_invaders_nearby.keys()):
            to_disarm.extend(
                (pursuer_id, identified_invaders_nearby[pursuer_id][0]))
            self.entities_manager.disarm_by_ids(to_disarm)
            pursuer_munition = self.entities_manager.get_quadcopters(id=pursuer_id)[
                0
            ].gun.munition

            # If there is no munition, then the LW suicided to kill the kamikaze
            if pursuer_munition == 0 and pursuer_id == agent_id:
                agent_suicided += 1

            elif pursuer_munition == 0:
                pursuer_suicided += 1

            else:
                explosion += 1

        return explosion, pursuer_suicided, agent_suicided

    def process_shoot_range_invaders(self) -> Tuple[int, int]:
        identified_invaders_nearby = self.offset_handler.identify_invaders_in_range(
            self.PURSUER_SHOOT_RANGE
        )

        agent = self.entities_manager.get_agent()

        successful_rl_agent_shots = 0
        successful_allies_shots = 0

        for pursuer_id in list(identified_invaders_nearby.keys()):
            if were_successful := self.entities_manager.shoot_by_ids(
                pursuer_id, identified_invaders_nearby[pursuer_id][0]
            ):
                if pursuer_id == agent.id:
                    successful_rl_agent_shots += 1

                if pursuer_id != agent.id:
                    successful_allies_shots += 1

        return successful_rl_agent_shots, successful_allies_shots

    @property
    def status(self):
        return self._stage_status

    # ===============================================================================
    # Reward, Termination and INFO
    # ===============================================================================

    def compute_reward(
        self,
        successful_rl_agent_shots=0,
        successful_allies_shots=0,
        pursuers_exploded=0,
        pursuer_suicided=0,
        agent_suicided=0,
    ):  # sourcery skip: low-code-quality
        # Initialize reward components
        score, bonus, penalty = 0, 0, 0
        agent = self.entities_manager.get_agent()

        munition = agent.gun_state[0]
        reload_progress = agent.gun_state[1]
        gun_available = agent.gun_state[2]

        position = agent.inertial_data["position"]
        velocity = agent.inertial_data["velocity"]

        distance_to_origin = float(np.linalg.norm(position))

        target_id = -1
        closest_ally_id = self.offset_handler.identify_closest_ally(agent.id)

        # Handling the case when there is no ally
        if closest_ally_id == -1:
            target_id = self.offset_handler.identify_closest_invader(agent.id)

        else:
            target_id = self.offset_handler.identify_closest_invader(
                closest_ally_id)

        target_position = np.array([0, 0, 0])
        if target_id > -1:
            target = self.entities_manager.get_quadcopters(id=target_id)[0]
            target_position = target.inertial_data["position"]

        self.current_closest_distance = float(
            np.linalg.norm(position - target_position)
        )
        score -= self.current_closest_distance

        if 0.01 < self.last_closest_distance - self.current_closest_distance and (
            gun_available == 1 or munition == 0
        ):
            velocity = agent.inertial_data.get("velocity", np.zeros(3))
            bonus += np.linalg.norm(velocity)  # type: ignore

        self.last_closest_distance = self.current_closest_distance

        if gun_available == 1 or munition == 0:
            score = -self.current_closest_distance

        else:
            # score = current_closest_distance * reload_progress
            score = self.current_closest_distance * (2 * reload_progress - 1)

        if successful_rl_agent_shots > 0 or agent_suicided > 0:
            bonus += (
                successful_rl_agent_shots + agent_suicided  # + self.agent_kills / 10
            ) * self.MAX_REWARD

        # Initially, i tried 0.5, now i am going to 1.
        # WOW, 80% were really good, much better than 200% (24.01.2024 p08).
        # Now, i am going to try 0.5
        if successful_allies_shots > 0 or pursuer_suicided > 0:
            bonus += (
                0.5 * (successful_allies_shots +
                       pursuer_suicided) * self.MAX_REWARD
            )

        elif pursuers_exploded > 0:
            # print("pursuer exploded")
            penalty += self.MAX_REWARD * pursuers_exploded

        if position[2] < -5:  # 0.1:
            # Scale the penalty based on the position
            # The penalty is 0 at position 1 and MAX_REWARD at position 0.01
            # 0.01, it is touching the ground (more or less this number)
            scaling_factor = (-5 - position[2]) / (-5 + 6)
            penalty += scaling_factor * self.MAX_REWARD

        # Penalty for pursuers outside the dome
        num_pursuer_outside_dome = len(
            self.offset_handler.identify_pursuer_outside_dome()
        )
        if num_pursuer_outside_dome > 0:
            penalty += self.MAX_REWARD

        # Penalty for going outside combat zone.
        # distance_to_origin = float(np.linalg.norm(agent.inertial_data["position"]))
        if distance_to_origin > self.ENEMY_BORN_RADIUS - 2:
            penalty += distance_to_origin - self.ENEMY_BORN_RADIUS - 2

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
        # if self.building_life <= 0:
        # print("Building life depleted")
        #    self.is_building_destroyed = True
        #    self._stage_status = TaskStatus.FAILURE
        #    return True

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

        # All pursuers destroyed.
        if len(self.entities_manager.get_armed_pursuers()) == 0:
            # print("All pursuers destroyed")
            self._stage_status = TaskStatus.FAILURE
            return True

        # This is an training environment, which means that it has to stop when the agent dies.
        agent = self.entities_manager.get_agent()

        if not agent.armed:
            self._stage_status = TaskStatus.FAILURE
            return True

        # O QUE DEU MELHOR ANTES NÃO TINHA ISSO
        position = agent.inertial_data["position"]
        return position[2] < -5.99

    def compute_info(self):
        return {
            "agent_kills": self.agent_kills,
            "allies_kills": self.allies_kills,
            "deads": self.deads,
            # "is_building_destroyed": self.is_building_destroyed,
            "current_wave": self.current_round,
        }

    # ===============================================================================
    # Others Implementations
    # ===============================================================================

    def generate_positions(
        self, num_positions: int, r: float, min_z: float = 4
    ) -> np.ndarray:
        # Random azimuthal angles limited to 0 to pi for x > 0
        thetas = np.random.uniform(0, np.pi, num_positions)

        # Use min function to ensure that the lower bound for z is not greater than r
        lower_bound_z = min(min_z, r)
        min_phi = np.arccos(
            lower_bound_z / r
        )  # Adjust phi range based on lower bound for z

        # Ensure that phis range is valid even when r < min_z
        phis = (
            np.random.uniform(min_phi, np.pi / 2, num_positions)
            if r >= min_z
            else np.random.uniform(0, np.pi / 2, num_positions)
        )

        # Convert to Cartesian coordinates using broadcasting
        xs = r * np.sin(phis) * np.cos(thetas)  # type: ignore
        ys = r * np.sin(phis) * np.sin(thetas)
        zs = r * np.cos(phis)

        return np.column_stack((xs, ys, zs))

    def replace_invaders(self):
        # self.entities_manager.disarm_all()
        invaders = self.entities_manager.get_all_invaders()
        positions = self.generate_positions(
            len(invaders), self.ENEMY_BORN_RADIUS)
        # (self.invaders_positions)  # self.generate_positions(len(invaders), self.dome_radius / 2)
        self.entities_manager.replace_quadcopters(invaders, positions)

    def replace_pursuers(self):
        pursuers = self.entities_manager.get_all_pursuers()

        positions = self.generate_positions(len(pursuers), 2)
        self.entities_manager.replace_quadcopters(pursuers, positions)

    def spawn_invader_squad(self):
        # num_invaders = 2
        positions = self.generate_positions(
            self.NUM_INVADERS, self.ENEMY_BORN_RADIUS)
        # np.array([[0, 0, 10]])  #   # /2
        invaders: List[Quadcopter] = self.entities_manager.spawn_invader(
            positions, "invader"
        )

        for invader in invaders[1:]:
            self.entities_manager.disarm_by_quadcopter(invader)

        # self.invaders_positions = positions

    def spawn_pursuer_squad(self):
        positions = self.generate_positions(self.NUM_PURSUERS, 2)

        # Separa posição e nome do agente
        agent_pos = positions[:1]
        agent_name = ["Agent"]

        # Separa posições e nomes dos aliados
        allies_pos = positions[1:]
        ally_names = [f"Ally{i}" for i in range(1, self.NUM_PURSUERS)]

        # Spawna agente com FusedLiDAR
        agent = self.entities_manager.spawn_pursuer(
            positions=agent_pos, names=agent_name, lidar_radius=2 * self.dome_radius, use_fused_lidar=self.use_fused_lidar,
        )

        # Spawna aliados com LiDAR padrão
        allies = self.entities_manager.spawn_pursuer(
            allies_pos, ally_names, lidar_radius=2 * self.dome_radius
        )

        # Junta todos
        pursuers = agent + allies

        for pursuer in pursuers:
            print(
                f"pursuer {pursuer.id} spawned at {pursuer.inertial_data['position']}")
            pursuer.set_munition(self.MUNITION_PER_DEFENDER)

        print(
            f"[DEBUG] Total pursuers spawned: {len(pursuers)} of {self.NUM_PURSUERS}")

        self.entities_manager.set_agent()

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
        self.building_life = max(self.building_life, 0)
