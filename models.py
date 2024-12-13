from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F


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

class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=512, proj_dim=256, action_dim=2, momentum=0.996):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.momentum = momentum  # 提高momentum以获得更稳定的目标表示

        # 增强编码器网络
        self.online_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, repr_dim),
            nn.LayerNorm(repr_dim)
        ).to(device)

        # 增强投影器网络
        self.online_projector = nn.Sequential(
            nn.Linear(repr_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        ).to(device)

        # 简化预测器以减少过拟合
        self.online_predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        ).to(device)

        # 动作预测器
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim)
        ).to(device)

        # 创建目标网络
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # 冻结目标网络参数
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        tau = 1 - self.momentum  # 使用更高的momentum值
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data.mul_(self.momentum).add_(tau * online_params.data)
        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data.mul_(self.momentum).add_(tau * online_params.data)

    def forward(self, states, actions):
        B, T, C, H, W = states.shape
        device = states.device

        predictions = []
        online_preds = []
        targets = []

        if self.training:
            # 编码所有状态并应用BYOL
            state_reprs = self.online_encoder(states.view(B * T, C, H, W))
            state_reprs = state_reprs.view(B, T, -1)  # [B, T, D]
            
            # BYOL投影和预测
            online_preds = self.online_projector(state_reprs)
            online_preds = self.online_predictor(online_preds)
            
            with torch.no_grad():
                target_reprs = self.target_encoder(states.view(B * T, C, H, W))
                target_reprs = target_reprs.view(B, T, -1)
                targets = self.target_projector(target_reprs)

            # 当前状态表示
            current_repr = state_reprs[:, 0]
            predictions.append(current_repr.unsqueeze(1))

            # 预测未来状态
            for t in range(T - 1):
                action = actions[:, t]
                pred_repr = self.predictor(torch.cat([current_repr, action], dim=-1))
                predictions.append(pred_repr.unsqueeze(1))
                current_repr = state_reprs[:, t + 1]

        else:
            # 推理模式
            current_repr = self.online_encoder(states[:, 0])
            predictions.append(current_repr.unsqueeze(1))

            for t in range(T - 1):
                action = actions[:, t]
                pred_repr = self.predictor(torch.cat([current_repr, action], dim=-1))
                predictions.append(pred_repr.unsqueeze(1))
                current_repr = pred_repr

        predictions = torch.cat(predictions, dim=1)
        return predictions, online_preds, targets

    def predict_future(self, init_states, actions):
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1] 
        T = T_minus1 + 1
        
        predicted_reprs = []
        
        # 初始状态
        current_repr = self.online_encoder(init_states[:, 0])
        predicted_reprs.append(current_repr.unsqueeze(0))
        
        # 预测未来状态
        for t in range(T_minus1):
            action = actions[:, t]
            pred_repr = self.predictor(torch.cat([current_repr, action], dim=-1))
            predicted_reprs.append(pred_repr.unsqueeze(0))
            current_repr = pred_repr
            
        predicted_reprs = torch.cat(predicted_reprs, dim=0)
        return predicted_reprs