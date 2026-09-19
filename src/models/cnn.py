"""PyTorch CNN baseline for brain MRI slice classification."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class BrainMRICNN(nn.Module):
    def __init__(self, in_channels: int = 1, base_ch: int = 16, n_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.block1 = ConvBlock(in_channels, base_ch)
        self.block2 = ConvBlock(base_ch, base_ch * 2)
        self.block3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.block4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding_dim = base_ch * 8
        self.classifier = nn.Linear(self.embedding_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.gap(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
