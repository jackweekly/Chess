import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style policy/value network.
    - Input: 119 planes (8-frame history + metadata), shape (B, 119, 8, 8)
    - Policy head: 73 planes (8x8x73 = 4672 flat actions)
    - Value head: scalar in [-1, 1]
    """

    def __init__(self, input_planes: int = 119, channels: int = 256, blocks: int = 19):
        super().__init__()
        self.input_planes = input_planes
        self.channels = channels
        self.blocks = blocks

        self.conv_input = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head -> 73 planes
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 73, 1),
        )

        # Value head -> scalar
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_input(x)
        h = self.res_tower(h)
        pi = self.policy_head(h)  # (B,73,8,8)
        v = self.value_head(h)    # (B,1)
        pi = pi.view(pi.size(0), -1)  # flatten to (B,4672)
        return pi, v
