from dataclasses import dataclass
import numpy as np

from threatengage.environments.level3.components.quadcopter_manager import QuadcopterManager, Quadcopter
from threatengage.environments.level3.components.task_progression import StageStatus, Stage
from abc import ABC

from threatengage.environments.level3.components.offsets_handler import OffsetHandler
from core.notification_system.message_hub import MessageHub
from core.notification_system.topics_enum import Topics_Enum
from typing import Dict, Optional
import random


class L3Stage1(Stage):

    """
    Level 3 P1 - Raio de nascimento dos perseguidores = 1, o que os deixa bem próximo do perseguidor.
    Ele tem que aprender a ignorar o outro loyalwingman e focar no invasor.
    Esse estágio é:
    - 1 invasor
    - 1 perseguidor Agente RL
    - 1 perseguidor de suporte (aqui não faz nada. Está mais para estimular a rede, mostrando que ele existe)

    - score = (
            5 - current_closest_distance if gun_available else current_closest_distance
        )


    self.MAX_STEP = 400
    self.dome_radius = 20
    """

    def __init__(
        self,
        quadcopter_manager: QuadcopterManager,
        dome_radius: float,
        debug_on: bool = False,
    ):
        self.dome_radius = dome_radius  # dome_radius

        self.quadcopter_manager = quadcopter_manager
        self.debug_on = debug_on

        self.init_constants()
        self.init_globals()

        self.offset_handler = OffsetHandler(quadcopter_manager, self.dome_radius)

        self.messageHub = MessageHub()
        self.messageHub.subscribe(
            topic=Topics_Enum.AGENT_STEP_BROADCAST.value,
            subscriber=self._subscriber_simulation_step,
        )

        # print("dome_radius", dome_radius)
        print("L3Stage1 - P1 - instatiated")

        # print("dome_radius", dome_radius)

    def _subscriber_simulation_step(self, message: Dict, publisher_id: int):
        self.current_step = message.get("step", 0)
        self.timestep = message.get("timestep", 0)

    def init_constants(self):
        print("NUM PURSUERS = 1. ONLY THE RL AGENT")
        self.NUM_PURSUERS = 2  # 1  # 2  # (1 for the RL Agent, and the other to simulate the support pursuer)
        self.NUM_INVADERS = 5

        self.MAX_REWARD = 1000
        # self.PROXIMITY_THRESHOLD = 2
        self.PROXIMITY_PENALTY = 1000
        # self.CAPTURE_DISTANCE = 0.2

        self.INVADER_EXPLOSION_RANGE = 0.2
        self.PURSUER_SHOOT_RANGE = 1

        # self.MAX_STEP = 300  # 300 calls = 20 seconds for rl_frequency = 15
        self.MAX_STEP = 600  # 600 calls = 40 seconds for rl_frequency = 15
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

        # Eles já ficam reaparecendo, então não precisa limitar a munição para 5. Ou talvez eu limite mesmo, para ver se ele aprende a suicidar.
        # não sei se é uma boa ideia ou não... pq
        # munition_values = [8]
        # new_munition = random.choice(munition_values)
        self.quadcopter_manager.get_all_pursuers()[0].gun.set_munition(4)

        # print("All quadcopters armed")

        self.offset_handler.on_episode_start()
        # print("episode started")

    def on_episode_end(self):
        self.init_globals()
        self.quadcopter_manager.disarm_all()
        self.replace_disarmed_invaders()
        self.replace_pursuers()

        # print("All quadcopters replaced")

    def on_step_start(self):
        """
        Here lies the methods that should be executed BEFORE the STEP.
        It aims to set the environment to the simulation step execution.
        """

        # print(self.current_step)
        self.quadcopter_manager.drive_invaders()
        self.quadcopter_manager.drive_support_pursuers()
        pass

    def on_step_middle(self):
        """
        Here lies the methods that should be executed BEFORE observation and reward.
        So, pay attention to the order of the methods. Do not change the position of the quadcopters
        before the observation and reward.
        """

        self.offset_handler.on_middle_step()
        # self.update_building_life()

        successful_shots, pursuers_exploded = self.process_nearby_invaders()
        # self.process_invaders_in_origin()

        gun_state: np.ndarray = self.agent_gun_state()

        reward = self.compute_reward(successful_shots, pursuers_exploded, gun_state)

        termination = self.compute_termination()

        # armed_invaders = self.quadcopter_manager.get_armed_invaders()

        # if len(armed_invaders) == 0:
        #    self.replace_invaders()
        #    invaders = self.quadcopter_manager.get_all_invaders()
        #    self.quadcopter_manager.arm_by_quadcopter(invaders[0])

        disarmed_invaders = self.quadcopter_manager.get_disarmed_invaders()
        if len(disarmed_invaders) > 0:
            self.replace_disarmed_invaders()
            for invader in disarmed_invaders:
                self.quadcopter_manager.arm_by_quadcopter(invader)

            # invaders = self.quadcopter_manager.get_all_invaders()
            # self.quadcopter_manager.arm_by_quadcopter(invaders[0])

        return reward, termination

    def agent_gun_state(self) -> np.ndarray:
        pursuer: Quadcopter = self.quadcopter_manager.get_all_pursuers()[0]
        return pursuer.gun_state

    def process_nearby_invaders(self):
        # TODO: Account for LW suicide
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

        # if self.debug_on and self.current_step % 2 == 0:
        #    pursuers = self.quadcopter_manager.get_all_pursuers()
        #    for pursuer in pursuers:
        #        pursuer.lidar.debug_sphere()

    @property
    def status(self):
        return self._stage_status

    # ===============================================================================
    # Reward and Termination
    # ===============================================================================

    def compute_reward(
        self,
        successful_shots: int = 0,
        pursuers_exploded: int = 0,
        gun_state: np.ndarray = np.array([1, 0, 1]),
    ):
        score, bonus, penalty = 0, 0, 0
        munition = gun_state[0]
        reload_progress = gun_state[1]
        gun_available = gun_state[2]

        # Calculate the current and last closest distances

        current_closest_distance = (
            self.offset_handler.current_closest_pursuer_to_invader_distance
        )

        last_closest_distance = (
            self.offset_handler.last_closest_pursuer_to_invader_distance
        )

        # =======================================================================
        # Calculate Base Score
        # =======================================================================

        if gun_available == 1:
            score = -current_closest_distance

        elif munition == 0:
            score = -current_closest_distance

        else:
            # score = current_closest_distance * reload_progress
            score = current_closest_distance * (2 * reload_progress - 1)

        # bonification for getting closer to the invader
        # 0.01 works like an threshold to avoid moviment oscilation trigger the bonus
        # I identified oscilations at maximun about 0.003. So, 0.01 is a good value (1/20 the dimension of the quadcopter that is already small).
        if 0.01 < last_closest_distance - current_closest_distance and (
            gun_available == 1 or munition == 0
        ):
            agent = self.quadcopter_manager.get_all_pursuers()[0]
            velocity = agent.inertial_data.get("velocity", np.zeros(3))
            bonus += 10 * np.linalg.norm(velocity)  # type: ignore
            # bonus += 10 * (last_closest_distance - current_closest_distance)
        bonus += self.MAX_REWARD * successful_shots
        penalty += self.MAX_REWARD * pursuers_exploded

        # Penalty for pursuers outside the dome
        num_pursuer_outside_dome = len(
            self.offset_handler.identify_pursuer_outside_dome()
        )
        if num_pursuer_outside_dome > 0:
            penalty += self.MAX_REWARD

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
        # if len(self.quadcopter_manager.get_armed_invaders()) == 0:
        #    self._stage_status = StageStatus.SUCCESS
        #    return True

        # All pursuers destroyed.
        if len(self.quadcopter_manager.get_armed_pursuers()) < self.NUM_PURSUERS:
            # print("All pursuers destroyed")
            self._stage_status = StageStatus.FAILURE
            return True

        # if len(self.offset_handler.identify_pursuer_touched_ground()) > 0:
        #    self._stage_status = StageStatus.FAILURE
        #    return True

        # if len(self.offset_handler.identify_invader_touched_ground()) > 0:
        #    self._stage_status = StageStatus.FAILURE
        #    return True
        # If none of the above conditions met, return False
        return False

    # ===============================================================================
    # Others Implementations
    # ===============================================================================

    def generate_positions(
        self, num_positions: int, r: float, r_max: float = 0
    ) -> np.ndarray:
        if r > r_max:
            r_max = r

        radius = np.random.uniform(r, r_max, num_positions)
        # Random azimuthal angles limited to 0 to 2 * pi for x > 0
        thetas = np.random.uniform(0, 2 * np.pi, num_positions)

        # Random polar angles limited to 0 to pi/2 for z > 0
        phis = np.random.uniform(0, np.pi / 2, num_positions)

        # Convert to Cartesian coordinates using broadcasting
        xs = radius * np.sin(phis) * np.cos(thetas)
        ys = radius * np.sin(phis) * np.sin(thetas)
        zs = radius * np.cos(phis)

        return np.column_stack((xs, ys, zs))

    def replace_disarmed_invaders(self):
        # self.quadcopter_manager.disarm_all()
        invaders = (
            self.quadcopter_manager.get_disarmed_invaders()
        )  # self.quadcopter_manager.get_all_invaders()
        positions = self.generate_positions(len(invaders), 2, 6)  # 6)
        self.quadcopter_manager.replace_quadcopters(invaders, positions)

    def replace_pursuers(self):
        # self.quadcopter_manager.disarm_all()
        pursuers = self.quadcopter_manager.get_all_pursuers()
        positions = self.generate_positions(len(pursuers), 1)
        self.quadcopter_manager.replace_quadcopters(pursuers, positions)

    def spawn_invader_squad(self):
        # num_invaders = 2
        positions = self.generate_positions(self.NUM_INVADERS, 2)  # 6)  # /2
        self.quadcopter_manager.spawn_invader(positions, "invader")

    def spawn_pursuer_squad(self):
        """
        Lidar radius deve ser 2 * dome_radius
        """

        positions = self.generate_positions(self.NUM_PURSUERS, 1)

        self.quadcopter_manager.spawn_pursuer(
            positions,
            ["RL Agent", "supporter"],
            lidar_radius=2 * self.dome_radius,  # NÃO MEXER AQUI
        )

        # pursuers = self.quadcopter_manager.get_all_pursuers()
        # for pursuer in pursuers:
        #    pursuer.lidar.debug = True

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
