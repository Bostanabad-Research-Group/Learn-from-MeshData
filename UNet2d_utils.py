import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d→BatchNorm→SiLU blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with ConvTranspose2d then DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2
            ]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpWithPixelShuffle(nn.Module):
    """Upscaling with PixelShuffle then DoubleConv."""
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.pixel_shuffle(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2
            ]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    """1×1 convolution to map to the desired number of output channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net with pixel-shuffle upsampling."""
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=32,
        num_hidden_layers=4,
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Down path
        channels = hidden_channels
        for _ in range(num_hidden_layers):
            self.down_layers.append(Down(channels, channels * 2))
            channels *= 2

        # Up path
        for _ in range(num_hidden_layers):
            self.up_layers.append(
                UpWithPixelShuffle(channels, channels // 2, upscale_factor=2)
            )
            channels //= 2

        self.outc = OutConv(hidden_channels, out_channels)
        # self.lmask = nn.Parameter(torch.ones(1, 1, res, res, device='cuda:0'))

    def forward(self, grid):
        mask = grid[:, 0:1, ...].clone()
        x = self.inc(grid)

        skip_connections = []
        for down in self.down_layers:
            skip_connections.append(x)
            x = down(x)

        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.up_layers, skip_connections):
            x = up(x, skip)
        return self.outc(x) * mask

class UNet_mo(nn.Module):
    """U-Net variant with multi-output mask parameter."""
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=32,
        num_hidden_layers=4,
        res=200
    ):
        super().__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        channels = hidden_channels
        for _ in range(num_hidden_layers):
            self.down_layers.append(Down(channels, channels * 2))
            channels *= 2

        for _ in range(num_hidden_layers):
            self.up_layers.append(
                UpWithPixelShuffle(channels, channels // 2, upscale_factor=2)
            )
            channels //= 2

        self.outc = OutConv(hidden_channels, out_channels)
        self.lmask = nn.Parameter(
            torch.ones(1, 3, res, res, device='cuda:0')
        )

    def forward(self, grid):
        mask = grid[:, 0:1, ...].clone()
        x = self.inc(grid)

        skip_connections = []
        for down in self.down_layers:
            skip_connections.append(x)
            x = down(x)

        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.up_layers, skip_connections):
            x = up(x, skip)

        return self.outc(x) * mask * self.lmask
