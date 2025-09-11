import numpy as np
import torch.nn as nn
from torch import Tensor, nn
import torch
from typing import Dict, Tuple
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from pytorch3d.ops import knn_points

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


class CellGNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int, int], k, hidden_dim=64):
        """
        input_shape: [B, N, C, M]
            B = batch
            N = vizinhos
            C = canais/features do LiDAR
            M = células por vizinho
        hidden_dim: H_gnn -> nº de features latentes por célula após a GNN
        """
        super().__init__()
        self.input_shape = input_shape
        _, _, C, _ = input_shape

        self.k = k
        self.hidden_dim = hidden_dim

        self.gnn1 = GATConv(C, hidden_dim, heads=1, concat=True)
        self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=True)

    def build_edges(self, coords, B, N, M, k):
        """
        Constrói arestas KNN de forma vetorizada (sem loop).
        coords: [B, N*M, d]
        return edge_index: [2, E]

        # edge_index contém todos os grafos concatenados,
        # mas com offsets para que cada batch seja isolado.
        # Assim temos B grafos independentes em uma única chamada.

        """
        num_nodes_per_batch = N * M

        # distâncias: [B, N*M, N*M]
        knn = knn_points(coords, coords, K=k+1)  # retorna dists e idx
        knn_idx = knn.idx[:, :, 1:]  # ignora self
        # top-k índices por batch: [B, N*M, k]

        # cria índices base [0..N*M-1], expandido no batch
        base_idx = torch.arange(num_nodes_per_batch, device=coords.device).view(
            1, -1, 1)  # [1, N*M, 1]
        base_idx = base_idx.expand(B, -1, k)  # [B, N*M, k]

        # offset por batch
        batch_offsets = (torch.arange(B, device=coords.device)
                         * num_nodes_per_batch).view(B, 1, 1)
        batch_offsets = batch_offsets.expand_as(base_idx)  # [B, N*M, k]

        # aplica offset
        src = base_idx + batch_offsets       # [B, N*M, k]
        dst = knn_idx + batch_offsets        # [B, N*M, k]

        # concatena
        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # [2, E]
        return edge_index.to(coords.device)

    def forward(self, x):
        """
        x: [B, N, C, M] -> features
        Saída: [B, N, M, H_gnn]
        """
        B, N, C, M = x.shape

        # rearranja: cada célula = vetor C
        nodes = x.permute(0, 1, 3, 2).reshape(B, N*M, C)  # [B, N*M, C]

        # edges (vetorizado por batch)
        edge_index = self.build_edges(nodes, B, N, M, self.k)

        # concatena batches em [B*N*M, C]
        nodes_flat = nodes.reshape(B*N*M, C)

        # aplica GNN
        out = F.elu(self.gnn1(nodes_flat, edge_index))
        out = self.gnn2(out, edge_index)  # [B*N*M, H_gnn]

        # volta para [B, N, M, H_gnn]
        return out.view(B, N, M, -1)

    def output_shape(self):
        """
        input_shape: (B, N, C, M)
        output:      (B, N, M, H_gnn)
        """
        B, N, _, M = self.input_shape
        return (B, N, M, self.hidden_dim)


class AttentionPooling(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        """
        input_shape = [B, N, M, H_in]
        hidden_dim  = dimensão das features de saída (H_out)
        """
        super().__init__()

        self.input_shape = input_shape
        _, N, M, H_in = input_shape
        self.hidden_dim = hidden_dim

        # MLP para calcular scores de atenção por vizinho
        self.att_mlp = nn.Sequential(
            nn.Linear(H_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)   # score escalar por vizinho
        )

        # Projeção final das features (H_in -> H_out)
        self.proj = nn.Linear(H_in, hidden_dim)

    def output_shape(self):
        """
        input_shape: (B, N, M, H_in)
        output:      (B, M, H_out)
        """
        B, N, M, _ = self.input_shape
        return (B, M, self.hidden_dim)

    def forward(self, h, mask=None):
        """
        h:    [B, N, M, H_in]
        mask: [B, N]  (True = válido, False = inválido)
        return: [B, M, H_out]
        """
        B, N, M, H_in = h.shape

        # Score por vizinho (agregado nas células)
        scores = self.att_mlp(h.mean(dim=2))  # [B, N, 1]
        scores = scores.squeeze(-1)           # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)

        alpha = torch.softmax(scores, dim=1)  # [B, N]

        # aplica atenção
        alpha_expanded = alpha.view(B, N, 1, 1)     # [B, N, 1, 1]
        pooled = torch.sum(h * alpha_expanded, dim=1)  # [B, M, H_in]

        return self.proj(pooled)  # [B, M, H_out]



class DeepSetSingleWorldView(nn.Module):
    """
    Entrada: [B, M, H_pool]
    Saída:   [B, output_dim]

    A entrada já reflete uma visão de mundo unificada
    (vizinho colapsado), restando apenas agregar as células.
    """

    def __init__(self, input_dim, hidden, output_dim, prob=0.0):
        super().__init__()

        self.mlp_phi = MLPphi(input_dim=input_dim,
                              output_dim=hidden, prob=prob)
        self.mlp_omega = MLPomega(input_dim=input_dim,
                                  output_dim=hidden, prob=prob)
        self.mlp_theta = MLPtheta(input_dim=hidden,
                                  hidden=hidden,
                                  output_dim=output_dim,
                                  prob=prob)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.prob = prob

    def forward(self, observation: torch.Tensor):
        """
        Forward pass.

        Args:
            observation: [B, M, H_pool]

        Returns:
            torch.Tensor: [B, output_dim]
        """
        B, M, H = observation.shape

        # φ transforma cada célula
        phi_out = self.mlp_phi(observation)           # [B, M, hidden]

        # ω gera pesos de atenção (aqui todos válidos -> máscara = None)
        mask = torch.ones(B, M, dtype=torch.bool, device=observation.device)
        omega_out = self.mlp_omega(observation, mask=mask)  # [B, M, hidden]

        # aplica pesos
        weighted = phi_out * omega_out                # [B, M, hidden]

        # soma sobre células
        agg = weighted.sum(dim=1)                     # [B, hidden]

        # aplica θ para saída final
        return self.mlp_theta(agg)                    # [B, output_dim]



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


class SphereCNNCells(nn.Module):
    """
    CNN que extrai embeddings por célula, preservando o C original (ex: 3 canais do LiDAR).
    Apenas reduz o espaço HxW para M células.

    Entrada:  [B, N, C, H, W]
    Saída:    [B, N, C, M]
    """

    def __init__(self, input_shape: Tuple[int, int, int, int, int], M: int,
                 conv_channels=(32, 64)):
        """
        input_shape = (B, N, C, H, W)
        M = nº de células de saída por vizinho (H' * W')
        conv_channels = nº de filtros intermediários
        """
        super().__init__()
        self.input_shape = input_shape
        _, _, C, H, W = input_shape

        self.M = M
        self.C = C

        self.block1 = nn.Sequential(
            HybridPadConv2d(C, conv_channels[0],
                            kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            HybridPadConv2d(
                conv_channels[0], conv_channels[1], kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )
        # volta para C original
        self.proj = nn.Conv2d(conv_channels[1], C, kernel_size=1)

        # checa quantas células saem naturalmente
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            feat = self.block1(dummy)
            feat = self.block2(feat)
            feat = self.proj(feat)
            _, _, H2, W2 = feat.shape
            flatten_dim = H2 * W2

        # se não bate em M, adapta
        self.adapt = nn.Linear(flatten_dim, M)

    def output_shape(self):
        """
        input_shape: (B, N, C, H, W)
        output:      (B, N, C, M)
        """
        B, N, _, _, _ = self.input_shape
        return (B, N, self.C, self.M)

    def forward(self, x):
        """
        x: [B, N, C, H, W]
        return: [B, N, C, M]
        """
        B, N, C, H, W = x.shape
        # junta batch e vizinhos
        x = x.view(B * N, C, H, W)        # [B*N, C, H, W]

        feat = self.block1(x)             # [B*N, C1, H1, W1]
        feat = self.block2(feat)          # [B*N, C2, H2, W2]
        feat = self.proj(feat)            # [B*N, C, H2, W2]
        feat = feat.flatten(2)            # [B*N, C, H2*W2]


        feat = self.adapt(feat)       # [B*N, C, M]

        # separa de volta
        feat = feat.view(B, N, C, self.M)  # [B, N, C, M]
        return feat




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


class SingleWorldViewNet(nn.Module):
    """
    CNN -> GNN (célula a célula) -> Pooling (colapsa vizinhos) -> DeepSet (agrega células)
    """

    def __init__(self, input_shape: Tuple[int, int, int, int, int],
                 M: int = 16, hidden=512, output_dim=512, prob=0.0):
        """
        Input Shape = (B, N, C, H, W):
        | -> B: batch size
        | -> N: número de vizinhos
        | -> C: canais/features por célula (ex: LiDAR)
        | -> H: polar angle resolution
        | -> W: azimuthal angle resolution
        """
        super().__init__()
        self.output_dim = output_dim

        # CNN gera embeddings por célula (mantém C, reduz HxW -> M células)
        self.sphere_cnn = SphereCNNCells(input_shape, M=M, conv_channels=(32, 64))
        sphere_cnn_output_shape = self.sphere_cnn.output_shape()  # (B, N, C, M)

        # GNN relaciona células (intra-vizinhos e inter-vizinhos)
        self.cell_gnn = CellGNN(input_shape=sphere_cnn_output_shape, k=5, hidden_dim=hidden)
        self.cell_gnn_output_shape = self.cell_gnn.output_shape()  # (B, N, M, H_gnn) h_gnn = hidden

        # Pooling colapsa os vizinhos (N -> 1)
        self.cell_pool = AttentionPooling(input_shape=self.cell_gnn_output_shape, hidden_dim=hidden)
        self.cell_pool_output_shape = self.cell_pool.output_shape()  # (B, M, H_pool) h_pool = hidden

        # DeepSet agrega as células (M -> embedding final)
        self.deepset = DeepSetSingleWorldView(
            input_dim=self.cell_pool_output_shape[-1],  # H_pool
            hidden=hidden,
            output_dim=output_dim,
            prob=prob
        )

    def forward(self, x, mask):
        """
        x:           [B, N, C, H, W]   -> LiDAR esférico
        mask:        [B, N]            -> vizinhos válidos
        """
        # CNN -> embeddings por célula
        # [B, N, C, M]
        # Aglutina a posição 2D para 1D (H,W -> M)
        # Saída: [B, N, C, M]
        # Propósito: reduzir o tamanho espacial.
        cnn_out = self.sphere_cnn(x)  

        # GNN -> células interconectadas
        # Contrói grafo KNN entre células (intra e inter-vizinhos)
        # Saída: [B, N, M, H_gnn]
        # Proósito: relacionar células as diferentes visões do mundo.
        gnn_out = self.cell_gnn(cnn_out)  # [B, N, M, H_gnn]

        # Pooling -> colapsa vizinhos
        # Saída: [B, M, H_pool]
        # M é o número de células, e H_pool é são as features latentes de uma visão unificada.
        # Propósito: criar uma única visão do mundo.
        pooled = self.cell_pool(gnn_out, mask)  # [B, M, H_pool]

        # Saída: [B, output_dim]
        # Propósito: a partir da visão unificada, extrair features relevantes.
        # É como se aqui ele tivesse que escolher o que é importante no mundo.
        return self.deepset(pooled)  # [B, output_dim]


class SingleWorldViewExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # --- Lidar ---
        lidar_space = observation_space["stacked_spheres"]   # (N, C, H, W)
        N, C, H, W = lidar_space.shape # type: ignore
        self.lidar_feature_extractor, n_flatten_lidar = self._preprocess_lidar(
            (1, N, C, H, W), output_dim=128
        )

        # --- Inertial ---
        inertial = observation_space["inertial_data"]
        action = observation_space["last_action"]


        self.unified_extractor, n_flatten_unified = self._unified_preprocess(inertial, action)

        # --- Final Layer ---
        concatenated_dim = n_flatten_lidar + n_flatten_unified
        self.final_layer = nn.Sequential(
            nn.Linear(concatenated_dim, features_dim),
            nn.ReLU()
        )

    def _preprocess_lidar(self, input_shape, output_dim):
        """
        input_shape: (1, N, C, H, W)  -> adicionamos B=1 só para construir a rede
        """
        extractor = SingleWorldViewNet(
            input_shape=input_shape,
            M=9,
            hidden=output_dim,
            output_dim=output_dim,
            prob=0.0
        )
        return extractor, output_dim  # já sai [B, output_dim]

    def _unified_preprocess(self, inertial, action):
        extractor = nn.Sequential(
            nn.Linear(inertial.shape[0] + action.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        with torch.no_grad():
            out_dim = extractor(torch.rand(1, inertial.shape[0] + action.shape[0])).shape[1]
        return extractor, out_dim
    

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        lidar_observation = observations["stacked_spheres"]   # [B, N, C, H, W]
        lidar_mask = observations["validity_mask"]            # [B, N]
        inertial_data = observations["inertial_data"]         # [B, D_inertial]
        action = observations["last_action"]                  # [B, D_action]


        lidar_features = self.lidar_feature_extractor(lidar_observation, lidar_mask)
        unified_features = self.unified_extractor(torch.cat((inertial_data, action), dim=1))


        concatenated_features = torch.cat(
            (lidar_features, unified_features),
            dim=1,
        )
        return self.final_layer(concatenated_features)




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

