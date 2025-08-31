from torch import Tensor, nn
import torch
from typing import Dict, Tuple
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """
    Basic MLP block with batch normalization, linear layer, optional activation, and optional dropout regularization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prob (float): Dropout probability. Ignored if use_dropout is False.
        activation_function (callable, optional): Activation function applied after the linear layer.
            If None, no activation is applied. Default: None.
        use_dropout (bool): Whether to apply dropout regularization. Default: True.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization for stable training.

        
        |-> Although batch norm is commomly used, it requires large mini batches for its
        |-> math works properly. Normaly this happens in Supervised Learning, 
        |-> but in RL, we normally have smaller batches (e.g., 32, 64, maybe 1).
        |-> So, I replaced it with LayerNorm to be safer.
        |-> batch_norm uses the whole batch for calculate mean and std,
        |-> layer_norm uses the features of each sample for calculate mean and std.

        linear (nn.Linear): Linear transformation of input features.
        activation (callable): User-defined activation function.
        dropout (nn.Dropout or None): Dropout regularization to prevent overfitting. None if use_dropout is False.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        prob=0.1,
        activation_function=None,
        use_dropout=False,
    ):
        super(MLPBlock, self).__init__()
        #self.batch_norm = nn.BatchNorm1d(num_features=input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation_function
        self.dropout = nn.Dropout(p=prob) if use_dropout else None

    def forward(self, x):
        """
        Performs the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after batch normalization, linear transformation,
            activation (if applicable), and dropout (if applicable).
        """

        x = self.layer_norm(x) #if x.size(0) > 1 else x
        x = self.linear(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.dropout(x) if self.dropout is not None else x

        return x


class MLPphi(nn.Module):
    """
    MLP responsible for aggregating information from neighbors.

    This module consists of three MLP blocks with residual connections between them,
    transforming the input data hierarchically.

    Args:
        hidden (int): Hidden layer size in each block.
        features (int): Number of input features.
        prob (float): Dropout probability.

    Attributes:
        phi_block_1 (MLPBlock): First MLP block.
        phi_block_2 (MLPBlock): Second MLP block.
        phi_block_3 (MLPBlock): Third MLP block.
    """

    def __init__(self, input_dim, output_dim, prob):
        super(MLPphi, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.phi_block_1 = MLPBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.phi_block_2 = MLPBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.phi_block_3 = MLPBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )

    def forward(self, u):
        """
        Performs the forward pass of the MLPphi.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, num_neighbors, features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_neighbors, hidden),
            after the transformations of the blocks and residual summation.
        """
        batch_size, num_neighbors, _ = u.shape
        # (batch * neighbors, features)
        u_flat = u.reshape(-1, self.input_dim)

        input_block_1 = u_flat
        output_block_1 = self.phi_block_1(input_block_1)

        input_block_2 = output_block_1
        output_block_2 = self.phi_block_2(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.phi_block_3(input_block_3)

        final_output = output_block_3 + input_block_3

        return final_output.reshape(batch_size, num_neighbors, self.output_dim)


class MLPomega(nn.Module):
    """
    MLP responsible for computing attention weights.

    This module uses three MLP blocks to process input data and applies a softmax function
    to generate normalized attention weights.

    Args:
        hidden (int): Hidden layer size in each block.
        features (int): Number of input features.
        prob (float): Dropout probability.

    Attributes:
        omega_block_1 (MLPBlock): First MLP block.
        omega_block_2 (MLPBlock): Second MLP block.
        omega_block_3 (MLPBlock): Third MLP block.
    """

    def __init__(self, input_dim, output_dim, prob):
        super(MLPomega, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.omega_block_1 = MLPBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.omega_block_2 = MLPBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.omega_block_3 = MLPBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            use_dropout=True,
            prob=prob,
        )

    def apply_mask(
        self, tensor: Tensor, mask: Tensor
    ) -> Tensor:
        """
        Applies a mask to a tensor, setting masked elements to a specific value.

        Args:
            tensor (torch.Tensor): The tensor to which the mask will be applied. [B*N, hidden]
            mask (torch.Tensor): The mask tensor (same or broadcastable shape as `tensor`). [B, N] ou [B*N]
            mask_value (float, optional): The value to assign to masked elements. Default: -float("inf").

        Returns:
            torch.Tensor: The tensor with the mask applied.
        """

        # [batch_size, num_neighbors] -> [batch_size * num_neighbors]
        mask = mask.to(tensor.device).bool()
        #print(f"[DEBUG] ApplyMask -> Mask {mask.shape}, Tensor {tensor.shape}")

        mask_expanded = mask.view(-1, 1).expand_as(tensor)
        fill_value = torch.finfo(tensor.dtype).min
        return tensor.masked_fill(~mask_expanded, fill_value)

        #mask_flat = mask.view(-1)

        # [batch_size * num_neighbors, 1]
        #mask_flat = mask_flat.unsqueeze(1)

        # [batch_size * num_neighbors, hidden]
        #mask_expanded = mask_flat.expand(-1, tensor.size(1))

        #return tensor.masked_fill(mask_expanded == 0, mask_value)

    def _grant_at_least_one_valid(self, mask: Tensor) -> Tensor:
        """
        Ensures that at least one element in each batch is valid in the mask to avoid
        all NaNs in the softmax.

        Args:
            mask (torch.Tensor): The mask tensor of shape (batch_size, num_neighbors).

        Returns:
            torch.Tensor: The modified mask tensor with at least one valid element per batch.
        """
        all_invalid = (mask.sum(dim=1) == 0)
        if all_invalid.any():
            mask = mask.clone()
            mask[all_invalid, 0] = True

        return mask

    def forward(self, Observation, mask):
        """
        Performs the forward pass of the MLPomega.

        Args:
            Observation [B, N, cnn_out]
            Mask [B, N]


        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_neighbors, hidden),
            containing normalized attention weights.
        """

        #print(f"[DEBUG] Omega forward -> Observation shape: {Observation.shape}, Mask shape: {mask.shape}")
        if mask is not None:
            mask = self._grant_at_least_one_valid(mask)

        # (batch * neighbors, features)
        batch_size, num_neighbors, _ = Observation.shape

        
        Observation_flat = Observation.reshape(-1, self.input_dim) # [B, N, cnn_out] -> [B*N, input_dim]

        input_block_1 = Observation_flat
        output_block_1 = self.omega_block_1(input_block_1)

        input_block_2 = output_block_1
        output_block_2 = self.omega_block_2(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.omega_block_3(input_block_3)  # [B*N, hidden]

        #print(f"[DEBUG] Omega forward -> Output block 3 shape: {output_block_3.shape}")
        output_block_3_masked = self.apply_mask(output_block_3, mask)  # [B*N, 1]

        output_reviewed = output_block_3_masked.reshape(
            batch_size, num_neighbors, self.output_dim)

        return torch.softmax(output_reviewed, dim=1).nan_to_num(0.0)


class MLPtheta(nn.Module):
    """
    MLP responsible for final regression.

    This module consists of four MLP blocks with residual connections, where the final block
    reduces the dimensionality to a single value for each instance.

    Args:
        hidden (int): Hidden layer size in each block.
        prob (float): Dropout probability.

    Attributes:
        theta_block_1 (MLPBlock): First MLP block.
        theta_block_2 (MLPBlock): Second MLP block.
        theta_block_3 (MLPBlock): Third MLP block.
        theta_block_4 (MLPBlock): Fourth MLP block for dimensionality reduction to a single output.
    """

    def __init__(self, input_dim, hidden, output_dim, prob):
        super(MLPtheta, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_features = output_dim

        self.theta_block_1 = MLPBlock(
            input_dim=input_dim,
            output_dim=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.theta_block_2 = MLPBlock(
            input_dim=hidden,
            output_dim=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.theta_block_3 = MLPBlock(
            input_dim=hidden,
            output_dim=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )

        self.theta_block_4 = MLPBlock(
            input_dim=hidden,
            output_dim=output_dim,  # 1,  # Reduces to a single output
            activation_function=None,
            use_dropout=False,
        )

    def forward(self, aggregated_features): #, num_neighbors):
        """
        Performs the forward pass of the MLPtheta.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, hidden).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """

        #batch_size, _ = aggregated_features.shape

        # (batch * neighbors, features)
        u_flat = aggregated_features.reshape(-1, self.input_dim)

        input_block_1 = u_flat
        output_block_1 = self.theta_block_1.forward(input_block_1)

        input_block_2 = output_block_1 + input_block_1
        output_block_2 = self.theta_block_2.forward(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.theta_block_3.forward(input_block_3)

        input_block_4 = output_block_3 + input_block_3
        return self.theta_block_4.forward(input_block_4)




class DeepSetAttentionNet(nn.Module):
    """
    Neural network model implementing a Deep Set Attention mechanism with three main components:
    MLP_phi, MLP_omega, and MLP_theta.

    This architecture is designed to process input data with a neighbor dimension, extract latent features
    using MLP_phi, compute attention weights using MLP_omega, and perform final regression using MLP_theta.
    It supports flexible input and output dimensions, making it suitable for tasks involving feature
    aggregation across neighbors.

    Args:
        input_dim (int): The number of input features per neighbor.
        output_dim (int): The number of output features from the model.
        hidden (int, optional): The size of the hidden layers for all MLP components. Default: 32.
        prob (float, optional): Dropout probability for regularization in MLP blocks. Default: 0.5.

    Attributes:
        mlp_phi (MLPphi): Module responsible for processing features into a latent space.
        mlp_omega (MLPomega): Module responsible for computing attention weights for each neighbor.
        mlp_theta (MLPtheta): Module responsible for final regression after feature aggregation.
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output features.
        hidden (int): The size of the hidden layers for all MLP components.
        prob (float): Dropout probability used for regularization.

    Methods:
        forward(u, mask):
            Processes input data through MLP_phi to extract latent features, applies MLP_omega to compute
            attention weights, aggregates features using weighted summation, and performs regression
            with MLP_theta. The mask is used to ignore padded elements during the computation.

    Forward Pass:
        1. Input tensor `u` of shape (batch_size, num_neighbors, input_dim) is reshaped to combine
        the batch and neighbor dimensions for processing.
        2. Latent features are extracted by MLP_phi from the flattened input.
        3. Attention weights are computed by MLP_omega and normalized using a mask.
        4. Element-wise multiplication combines latent features and attention weights.
        5. Weighted features are aggregated across the neighbor dimension using summation.
        6. Aggregated features are passed to MLP_theta for final regression, producing the output.

    Returns:
        torch.Tensor: Final output tensor of shape (batch_size, output_dim), representing the regression results.
    """

    def __init__(self, input_dim, hidden, output_dim, prob=0.5):
        super(DeepSetAttentionNet, self).__init__()

        #TODO: Mudar para termos input_dim e output_dim
        #PHi e Omega devem ter output_dim ?
        self.mlp_phi = MLPphi(input_dim=input_dim, output_dim=hidden, prob=prob)
        self.mlp_omega = MLPomega(input_dim=input_dim, output_dim=hidden, prob=prob)
        self.mlp_theta = MLPtheta(input_dim=hidden, hidden=hidden, output_dim=output_dim, prob=prob)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.prob = prob

    # TODO: I have to find a better way to abstract "observation" to extract
    # "mask" and "u" of it.
    def forward(self, observation, mask):
        """
        Forward pass of the DeepSetAttentionNet.

        Args:

           Observation [B, N, cnn_out]
           Mask [B, N]

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """

        batch_size, num_neighbors, _ = observation.shape

        output_mlp_phi = self.mlp_phi(observation)

        output_mlp_omega = self.mlp_omega(observation, mask)

        weighted_features = output_mlp_phi * output_mlp_omega
        aggregated_features = weighted_features.sum(
            dim=1).reshape(batch_size, -1)

        return self.mlp_theta.forward(aggregated_features)


class HybridPadConv2d(nn.Module):
    """Conv2d with circular padding on W axis (Azimuth)."""
    #TODO: Reler sobre como isso aqui funciona

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), bias=True):
        super().__init__()

        self.NO_VALUE = 1  # Its like a "no data" value for LIDAR, used in padding
        kh, kw = (kernel_size if isinstance(kernel_size, tuple)else (kernel_size, kernel_size))
        sh, sw = (stride if isinstance(stride, tuple) else (stride, stride))

        self.pad_w = (kw - 1) // 2
        self.pad_h = (kh - 1) // 2

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), stride=(sh, sw), padding=(0, 0), bias=bias)

    def forward(self, x):

        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")

        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h),mode="constant", value=self.NO_VALUE)
        return self.conv(x)

class SphereCNN(nn.Module):
    """CNN shared for each sphere.
    - Uses circular padding on width (azimuthal, phi) dimension.
    - Focus on use same process for any sphere, so the NN looks for patterns in each sphere independently.
    |-> Expects that Batch and Neighbors are already merged in a sigle dimension outside this class.
    """

    def __init__(self, shape: Tuple[int, int, int], output_dim=128, conv_channels=(32, 64)):
        """
        - shape = (input_channels, H, W):
        |-> input_channels: number of input channels (features) per sphere (e.g., 3 if using [normalized_r, entity_type, delta step]).
        |-> H: height of the matrix (n_theta, polar angle resolution).
        |-> W: width of the matrix (n_phi, azimuthal angle resolution).

        - output_dim: dimension of the output feature vector for each sphere.
        
        """
        super().__init__()

        input_channels, height, width = shape
        conv1_output = conv_channels[0]  # 32 maps
        conv2_output = conv_channels[1]  # 64 maps

        self.block1 = nn.Sequential(HybridPadConv2d(
            in_ch=input_channels, out_ch=conv1_output, kernel_size=(5, 5), stride=(2, 2)), nn.ReLU())
        self.block2 = nn.Sequential(HybridPadConv2d(
            in_ch=conv1_output, out_ch=conv2_output, kernel_size=(3, 3), stride=(2, 2)), nn.ReLU())
        self.flatten = nn.Flatten()
        

        # descobrir dimensão dinamicamente
        with torch.no_grad():
            sample = torch.rand(1, input_channels, height, width)  # H,W só exemplo
            output_block1 = self.block1(sample)
            output_block2 = self.block2(output_block1)
            n_flatten = self.flatten(output_block2).shape[1]
        self.linear = nn.Linear(n_flatten, output_dim)

    def forward(self, x):  # [B, C, H, W]
        output_block1 = self.block1(x)
        output_block2 = self.block2(output_block1)
        output_flatten = self.flatten(output_block2)
        return self.linear(output_flatten)   # [B, out_dim]


class CNNDeepSetAttentionNet(nn.Module):
    """
    DeepSet com CNN por vizinho.
    """

    def __init__(self, input_shape: Tuple[int, int, int], hidden=512, output_dim=512, prob=0.5):
        """
        Input Shape = (input_channels, H, W):
        |-> input_channels: number of input channels (features) per sphere (e.g., 3 if using [normalized_r, entity_type, delta step]).
        |-> H: height of the matrix (n_theta, polar angle resolution).
        |-> W: width of the matrix (n_phi, azimuthal angle resolution).

        
        """
        super().__init__()
        self.output_dim = output_dim
        self.sphere_cnn = SphereCNN(input_shape, output_dim=hidden)
        self.deepset = DeepSetAttentionNet(input_dim=hidden, hidden=hidden, output_dim=output_dim, prob=prob)
        

    def forward(self, observation, mask):
        """
        observation: [B, N, C, H, W]
        mask: [B, N]

        B: batch size
        N: number of neighbors
        C: input channels
        H: height -> polar angle
        W: width -> azimuthal angle
        """
        #B, N, C, H, W = observation.shape

        #print(f"[DEBUG] CNNDeepSetAttentionNet 1 -> Observation shape: {observation.shape}, Mask shape: {mask.shape}")

        # Aceita [B, N, C, H, W] ou [N, C, H, W]
        if observation.dim() == 5:
            # já está no formato certo
            B, N, C, H, W = observation.shape
        elif observation.dim() == 4:
            # só aceito como [N,C,H,W] se for B=1 (não vetorizado)
            N, C, H, W = observation.shape
            B = 1
            observation = observation.unsqueeze(0)  # -> [1,N,C,H,W]
            if mask.dim() == 1:          # [N]
                mask = mask.unsqueeze(0)  # -> [1,N]
            elif mask.dim() == 2 and mask.shape[0] == 1:
                pass                     # já está [1,N]
            else:
                raise ValueError(f"Inconsistent mask for B=1,N={N}: {mask.shape}")
        else:
            raise ValueError(f"Unexpected lidar shape: {observation.shape}")

        #print(f"mask: {mask}")        
        # 1 sphere by time -> batch x neighbors
        #print(f"[DEBUG] CNNDeepSetAttentionNet 2 -> Observation shape: {observation.shape}, Mask shape: {mask.shape}")
        x = observation.reshape(B*N, C, H, W)
        sphere_features = self.sphere_cnn(x).reshape(B, N, -1)  # [B*N, cnn_out] -> [B, N, cnn_out]

        #print(
            #f"[DEBUG] CNNDeepSetAttentionNet 3 -> sphere_features shape: {sphere_features.shape}, Mask shape: {mask.shape}")
        return self.deepset.forward(sphere_features, mask)  # mask = [B, N]

       



class FLIAExtractor(BaseFeaturesExtractor):
    #Fused Lidar, Inertial and Action Extractor
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # super(LidarInertialActionExtractor, self).__init__()
        #print("FLIAExtractor: Using Fused Lidar, Inertial and Action Extractor")
        #print("[Warning] Gun observation key is not processed here (the key is not, but is expected Gun observation together in inertial_data)")
        super().__init__(observation_space, features_dim)

        lidar = observation_space["stacked_spheres"]  # [B, N, C, H, W]
        mask = observation_space["validity_mask"]   # [B, N]

        #print(f"[DEBUG] FLIAExtractor -> Lidar shape: {lidar.shape}, Mask shape: {mask.shape}")
        inertial = observation_space["inertial_data"] 
        action = observation_space["last_action"]

        self.lidar_feature_extractor, n_flatten_lidar = self._preprocess_lidar(lidar)
        (self.inertial_feature_extractor,n_flatten_inertial,) = self._preprocess_inertial_data(inertial)
        self.action_feature_extractor, nflatten_action = self._preprocess_action(action)

        # Concatenate both feature extractors and pass through a linear layer
        concatenated_dim = n_flatten_lidar + n_flatten_inertial + nflatten_action
        self.final_layer = nn.Sequential(nn.Linear(concatenated_dim, features_dim), nn.ReLU())

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

    def _preprocess_lidar(self, lidar_space, output_dim=512) -> Tuple[CNNDeepSetAttentionNet, int]:
        # Supondo que cada vizinho tem input_dim features (+ máscara no final)

        _, C, H, W = lidar_space.shape  # stack_size (N neighbors), channels (distance, entity type, delta step), n_theta, n_phi
        extractor = CNNDeepSetAttentionNet(input_shape=(C, H, W), hidden=output_dim, output_dim=output_dim, prob=0.5)
     


        # o output já é (batch_size, output_dim)
        return extractor, output_dim

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

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        lidar_observation = observations["stacked_spheres"]
        lidar_mask = observations["validity_mask"]

        #print(f"[DEBUG] FLIAExtractor forward -> Lidar shape: {lidar_observation.shape}, Mask shape: {lidar_mask.shape}")
        inertial_data = observations["inertial_data"]
        action = observations["last_action"]

        lidar_features = self.lidar_feature_extractor(lidar_observation, lidar_mask)
        inertial_features = self.inertial_feature_extractor(
            inertial_data.flatten(start_dim=1)
        )
        action_features = self.action_feature_extractor(action)

        concatenated_features = torch.cat(
            (lidar_features, inertial_features, action_features),
            dim=1,
        )
        return self.final_layer(concatenated_features)

