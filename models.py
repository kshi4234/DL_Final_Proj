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
    def __init__(self, device="cuda", repr_dim=256, momentum=0.99):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.momentum = momentum
        
        # 编码器网络 - 使用残差连接增强特征提取
        self.online_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, repr_dim),
            nn.LayerNorm(repr_dim)
        ).to(device)
        
        # 预测器网络 - 用于预测下一个状态表示
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + 2, repr_dim),  # +2 for action
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim)
        ).to(device)
        
        # 目标编码器
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
    def compute_system_energy(self, states, actions):
        """
        计算整个观察-动作序列的系统能量
        """
        B, T, C, H, W = states.shape
        
        # 获取所有状态的目标表示
        with torch.no_grad():
            target_states = self.target_encoder(
                states.view(-1, C, H, W)
            ).view(B, T, -1)
        
        # 获取初始状态表示
        current_state = self.online_encoder(states[:, 0])
        total_energy = 0
        predictions = [current_state]
        
        # 计算每个时间步的能量
        for t in range(T-1):
            # 预测下一个状态
            action = actions[:, t]
            pred_next_state = self.predictor(
                torch.cat([current_state, action], dim=-1)
            )
            predictions.append(pred_next_state)
            
            # 计算与目标状态的距离
            energy = energy_distance(pred_next_state, target_states[:, t+1])
            total_energy = total_energy + energy.mean()
            
            # 更新当前状态
            if self.training:
                current_state = self.online_encoder(states[:, t+1])
            else:
                current_state = pred_next_state
                
        predictions = torch.stack(predictions, dim=1)
        return total_energy / (T-1), predictions

    def forward(self, states, actions):
        """
        前向传播，返回系统能量和预测序列
        """
        energy, predictions = self.compute_system_energy(states, actions)
        return energy, predictions
        
    @torch.no_grad()
    def update_target_encoder(self):
        """
        更新目标编码器的参数
        """
        tau = 1 - self.momentum
        for online_params, target_params in zip(
            self.online_encoder.parameters(), 
            self.target_encoder.parameters()
        ):
            target_params.data.mul_(self.momentum).add_(
                tau * online_params.data
            )
            
    def predict_future(self, init_states, actions):
        """
        预测未来状态序列
        """
        B, _, C, H, W = init_states.shape
        T = actions.shape[1] + 1
        
        predictions = []
        current_state = self.online_encoder(init_states[:, 0])
        predictions.append(current_state.unsqueeze(0))
        
        for t in range(T-1):
            action = actions[:, t]
            pred_next = self.predictor(
                torch.cat([current_state, action], dim=-1)
            )
            predictions.append(pred_next.unsqueeze(0))
            current_state = pred_next
            
        return torch.cat(predictions, dim=0)

class ResidualBlock(nn.Module):
    """残差块用于增强特征提取"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out