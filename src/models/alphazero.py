import torch
import torch.nn as nn
from src.models.supervised_baseline import ConvPolicy


class AlphaZeroNet(ConvPolicy):
    """ResNet policy with value head for AlphaZero-style training."""

    def __init__(self, channels: int = 256, blocks: int = 20, n_classes: int = 4672, input_channels: int = 119):
        super().__init__(channels=channels, blocks=blocks, n_classes=n_classes, input_channels=input_channels)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.input_channels, 8, 8)
        out = self.stem(x)
        out = self.resblocks(out)
        pi = self.head(out)
        v = self.value_head(out)
        return pi, v
