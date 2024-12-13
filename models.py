from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F
import copy

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class Encoder(nn.Module):
    def __init__(self, input_channels=2, input_size=(65, 65), repr_dim=256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv_net(sample_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, repr_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        return self.mlp(x)

def energy_distance(pred_state, target_state):
    """
    计算预测状态和目标状态之间的能量距离
    使用负余弦相似度作为距离度量
    """
    # 归一化向量
    pred_norm = F.normalize(pred_state, p=2, dim=-1)
    target_norm = F.normalize(target_state, p=2, dim=-1)
    
    # 计算余弦相似度并转换为距离
    distance = 1 - F.cosine_similarity(pred_norm, target_norm, dim=-1)
    return distance

class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=128, action_dim=2, momentum=0.99):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.momentum = momentum

        # Encoder with spatial attention
        self.online_encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # First spatial attention block
            SpatialAttentionBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Second spatial attention block
            SpatialAttentionBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Final processing
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, repr_dim),
            nn.LayerNorm(repr_dim)
        ).to(device)

        # Predictor with stronger state-action fusion
        self.predictor = StateActionFusion(repr_dim, action_dim).to(device)

        # Target networks
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, states, actions):
        """Forward pass with better state-action integration"""
        B, T, C, H, W = states.shape
        
        # Get all encodings
        online_states = self.online_encoder(states.view(-1, C, H, W)).view(B, T, -1)
        with torch.no_grad():
            target_states = self.target_encoder(states.view(-1, C, H, W)).view(B, T, -1)
        
        # Prepare predictions
        predictions = []
        current_state = online_states[:, 0]
        predictions.append(current_state.unsqueeze(1))
        
        # Generate predictions
        for t in range(T-1):
            action = actions[:, t]
            pred_next = self.predictor(current_state, action)
            predictions.append(pred_next.unsqueeze(1))
            
            if self.training:
                current_state = online_states[:, t+1]
            else:
                current_state = pred_next
        
        predictions = torch.cat(predictions, dim=1)
        return predictions, online_states, target_states

    def predict_future(self, init_states, actions):
        """Predict future states given initial states and actions sequence"""
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1
        
        # Get initial state representation
        initial_repr = self.online_encoder(init_states[:, 0])  # [B, D]
        predicted_reprs = [initial_repr.unsqueeze(0)]  # List to store all predictions
        
        # Current state for unrolling
        current_repr = initial_repr
        
        # Predict future states autoregressively
        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            next_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(next_repr.unsqueeze(0))  # Add to predictions
            current_repr = next_repr
        
        # Stack all predictions
        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        return predicted_reprs

    @torch.no_grad()
    def update_target_encoder(self):
        tau = 1 - self.momentum
        for online_params, target_params in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_params.data.mul_(self.momentum).add_(tau * online_params.data)

class SpatialAttentionBlock(nn.Module):
    """Spatial attention mechanism to focus on important regions"""
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv_spatial(x))
        return x * attention

class StateActionFusion(nn.Module):
    """Better state-action integration"""
    def __init__(self, repr_dim, action_dim):
        super().__init__()
        self.state_proj = nn.Linear(repr_dim, repr_dim)
        self.action_proj = nn.Linear(action_dim, repr_dim)
        self.fusion = nn.Sequential(
            nn.Linear(repr_dim * 2, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim)
        )

    def forward(self, state, action):
        state_feat = self.state_proj(state)
        action_feat = self.action_proj(action)
        combined = torch.cat([state_feat, action_feat], dim=-1)
        return self.fusion(combined)