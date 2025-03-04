import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class DilatedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super(DilatedUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        
        # Bridge with dilated convolutions
        self.bridge = nn.Sequential(
            ResidualBlock(512, 512, dilation=2),
            ResidualBlock(512, 512, dilation=4),
            ResidualBlock(512, 512, dilation=8)
        )
        
        # Decoder
        self.dec4 = ResidualBlock(1024, 256)
        self.dec3 = ResidualBlock(512, 128)
        self.dec2 = ResidualBlock(256, 64)
        self.dec1 = ResidualBlock(128, out_channels)
        
        # Up-sampling layers
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        b = self.bridge(e4)
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return torch.sigmoid(d1)
