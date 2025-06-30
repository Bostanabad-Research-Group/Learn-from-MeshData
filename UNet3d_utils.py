
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),#, groups=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.up = up


    def forward(self, x):
        x = self.double_conv(x)

        return x



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2 )#, groups = out_channels)
        self.conv = DoubleConv(in_channels, out_channels )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handling size mismatch for 3D volumes
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [
            diff_w // 2, diff_w - diff_w // 2,  # Width padding
            diff_h // 2, diff_h - diff_h // 2,  # Height padding
            diff_d // 2, diff_d - diff_d // 2   # Depth padding
        ])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class OutConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)  # Intermediate convolution
        self.bn1 = nn.BatchNorm3d(in_channels // 2)  # Batch normalization
        self.act1 = nn.SiLU(inplace=True)  # Activation

        self.conv2 = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)  # Final convolution
        self.bn2 = nn.BatchNorm3d(out_channels)
        #self.act2 = nn.Sigmoid()  # Output activation for normalization (can be replaced)

    def forward(self, x):
        x = self.conv1(x)  # First convolution
        x = self.bn1(x)  # Batch normalization
        x = self.act1(x)  # Activation
        x = self.conv2(x)  # Second convolution
        x = self.bn2(x)

        return x



class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, depth=4 , res = 50):
        super().__init__()
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
 
        dim = 2
        self.strain_conv = nn.Sequential(

            
            nn.Conv3d(4, 16, kernel_size=1), 
            nn.BatchNorm3d(16),
            nn.SiLU(inplace=True),            

            nn.ConvTranspose3d(
                16, 1, 
                kernel_size=[dim,dim,dim], 
                stride=3,
                output_padding=0 
            )
        )

        channels = hidden_channels
        for _ in range(depth):
            self.down_layers.append(Down(channels, channels * 2))
            channels *= 2

        # Up path
        for _ in range(depth):
            self.up_layers.append(Up(channels, channels // 2))
            channels //= 2

        self.outc = OutConv(hidden_channels, out_channels)

    def forward(self, grid, maxs):
        mask = grid[:, 0:1, ...].clone()
        strain = grid[:, 1:2, 1:2, 1:2, 1:2]  # (B, 1, 1, 1, 1)
        cats = torch.cat([strain , maxs] , -1).permute(0 , 4 , 1 , 2 , 3)
        
        STRAIN = self.strain_conv(cats) 
        
        x = mask
        x = self.inc(x)

        # Down path
        skip_connections = []
        for down in self.down_layers:
            skip_connections.append(x)
            x = down(x)

        x = STRAIN * x

        # Up path
        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.up_layers, skip_connections):
            x = up(x, skip)

        out = self.outc(x) * mask
        return out
    
