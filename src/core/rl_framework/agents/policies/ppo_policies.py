import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from gymnasium import spaces
import numpy as np
from typing import Tuple
from typing import NamedTuple

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from core.rl_framework.agents.policies.fused_lidar_extractor import FLIAExtractor

class MatrixInputShape(NamedTuple):
    channel: int  # can be 0 (distance) or 1 (target type)
    theta: int  # spherical coordinate system - theta
    phi: int  # spherical coordinate system - phi


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: spaces.Box, features_dim: int = 256, activation="ReLU"
    ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            getattr(nn, activation)(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),
            getattr(nn, activation)(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param hidden_dims: (list) number of units for each hidden layer
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list = [64, 64],
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = self.create_mlp(feature_dim, hidden_dims, last_layer_dim_pi)

        # Value network
        self.value_net = self.create_mlp(feature_dim, hidden_dims, last_layer_dim_vf)

    def create_mlp(self, input_dim, hidden_dims, output_dim):
        """
        Creates a MLP with the specified dimensions.

        :param input_dim: (int) The dimension of the input layer
        :param hidden_dims: (list) A list of integers where each integer is the dimension of a hidden layer
        :param output_dim: (int) The dimension of the output layer
        :return: a MLP as a nn.Sequential instance
        """
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        net_arch = kwargs.get("net_arch")
        default_hiddens = [256, 256, 256]

        if net_arch is None:
            warnings.warn(f"net_arch is None. Using default hiddens: {default_hiddens}")
            self.hiddens = default_hiddens
        elif not isinstance(net_arch, dict):
            warnings.warn(
                f"net_arch is not a dictionary. Using default hiddens: {default_hiddens}"
            )
            self.hiddens = default_hiddens
        else:
            # Get the 'pi' key from the dictionary, if it exists
            self.hiddens = net_arch.get("pi", default_hiddens)

        print(f"CustomActorCritic Hiddens: {self.hiddens}")
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.hiddens)

    def __str__(self):
        return f"CustomActorCriticPolicy (hidden layers: {self.hiddens})"


# ===================================================================================================
# =================================== MixObservationNN ==============================================
# ===================================================================================================


class CustomActorCriticPolicyMixedObservation(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        net_arch = kwargs.get("net_arch")
        default_hiddens = [256, 256, 256]

        if net_arch is None:
            warnings.warn(f"net_arch is None. Using default hiddens: {default_hiddens}")
            self.hiddens = default_hiddens
        elif not isinstance(net_arch, dict):
            warnings.warn(
                f"net_arch is not a dictionary. Using default hiddens: {default_hiddens}"
            )
            self.hiddens = default_hiddens
        else:
            # Get the 'pi' key from the dictionary, if it exists
            self.hiddens = net_arch.get("pi", default_hiddens)

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.observation_space = observation_space

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LidarInertialActionExtractor(self.observation_space)


# ===================================================================================================
# PPO Extractor
#
# The official extractor class is LidarInertialActionExtractor for ThreatEngage up to level 4
#
# ===================================================================================================


class LidarInertialActionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # super(LidarInertialActionExtractor, self).__init__()
        print("LidarInertialActionExtractor")
        print("[Warning] Gun observation key is not processed here (the key is not, but is expected Gun observation together in inertial_data)")
        super().__init__(observation_space, features_dim)

        lidar = observation_space["lidar"]
        inertial = observation_space["inertial_data"]
        action = observation_space["last_action"]

        self.lidar_feature_extractor, n_flatten_lidar = self._preprocess_lidar(lidar)
        (
            self.inertial_feature_extractor,
            n_flatten_inertial,
        ) = self._preprocess_inertial_data(inertial)
        self.action_feature_extractor, nflatten_action = self._preprocess_action(action)

        # Concatenate both feature extractors and pass through a linear layer
        concatenated_dim = n_flatten_lidar + n_flatten_inertial + nflatten_action
        self.final_layer = nn.Sequential(
            nn.Linear(concatenated_dim, features_dim), nn.ReLU()
        )

    def _preprocess_action(self, action) -> Tuple[nn.Sequential, int]:
        # Feature extractor for the tuple observation (using Linear layers)
        action_feature_extractor = nn.Sequential(
            nn.Linear(action.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass for the tuple feature extractor
        with torch.no_grad():
            n_flatten_action = action_feature_extractor(
                torch.rand(1, action.shape[0])
            ).shape[1]

        return action_feature_extractor, n_flatten_action

    def _preprocess_lidar(self, lidar_data) -> Tuple[nn.Sequential, int]:
        # Feature extractor for the matrix observation (using Conv2d layers)
        lidar_feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=lidar_data.shape[0],
                out_channels=32,
                kernel_size=4,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass for the matrix feature extractor
        with torch.no_grad():
            n_flatten_lidar = lidar_feature_extractor(
                torch.rand(1, *lidar_data.shape)
            ).shape[1]

        return lidar_feature_extractor, n_flatten_lidar

    def _preprocess_inertial_data(self, inertial) -> Tuple[nn.Sequential, int]:
        # Feature extractor for the tuple observation (using Linear layers)
        inertial_feature_extractor = nn.Sequential(
            nn.Linear(inertial.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Compute shape by doing one forward pass for the tuple feature extractor
        with torch.no_grad():
            n_flatten_inertial = inertial_feature_extractor(
                torch.rand(1, inertial.shape[0])
            ).shape[1]

        return inertial_feature_extractor, n_flatten_inertial

    def forward(self, observations: Dict[str, th.Tensor]) -> torch.Tensor:
        

        lidar_observation = observations["lidar"]
        inertial_data = observations["inertial_data"]
        action = observations["last_action"]

        #print("[DEBUG] PPO POLICIES LIDAR INERTIAL ACTION EXTRACTOR inertial_data.shape:", inertial_data.shape)

        lidar_features = self.lidar_feature_extractor(lidar_observation)
        inertial_features = self.inertial_feature_extractor(
            inertial_data.flatten(start_dim=1)
        )
        action_features = self.action_feature_extractor(action)

        concatenated_features = torch.cat(
            (lidar_features, inertial_features, action_features),
            dim=1,
        )
        return self.final_layer(concatenated_features)
    

class StudentWithFLIA(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, action_dim=4):
        super().__init__(observation_space, features_dim)
        self.feature_extractor = FLIAExtractor(
            observation_space, features_dim=features_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
        )

    def forward(self, obs):
        features = self.feature_extractor(obs)
        return self.mlp_head(features)
    




class StudentPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(StudentPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs)

        # usa sua StudentWithFLIA como backbone
        self.backbone = StudentWithFLIA(observation_space)

        # carrega os pesos supervisionados aqui
        self.backbone.load_state_dict(torch.load(
            "C:\\Users\\davi_\\Documents\\GitHub\\dronechase\\apps\\threatsense_runner\\09.09.2025_trained_models_all_3\\models\\student_last_model_0.pt"))

        # policy head (ações)
        # 4 -> sua dimensão da ação
        self.action_net = nn.Linear(4, action_space.shape[0])
        # value head (valor do estado)
        self.value_net = nn.Linear(4, 1)

    def forward(self, obs, deterministic=False):
        features = self.backbone(obs)  # [B, 4]

        # valor do estado
        values = self.value_net(features)

        # distribuição gaussiana diagonal
        mean_actions = self.action_net(features)
        distribution = self.action_dist.proba_distribution(
            mean_actions, self.log_std)

        # amostra ação
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        features = self.backbone(obs)
        mean_actions = self.action_net(features)
        distribution = self.action_dist.proba_distribution(
            mean_actions, self.log_std)
        return distribution.get_actions(deterministic=deterministic)
