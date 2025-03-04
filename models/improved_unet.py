import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut(x))
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        return x + shortcut


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.reduce_channels = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.residual = ResidualBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.bn(x)

        # Resize to match skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        x = torch.cat([x, skip], dim=1)  # Now both tensors have the same spatial size
        x = self.reduce_channels(x)
        return self.residual(x)



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.residual(x)
        x = self.pool(skip)
        return x, skip


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in [2, 4, 8]
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for conv in self.convs:
            x = F.relu(conv(x))
        return x


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.initial = ResidualBlock(in_channels, 16)

        self.enc1 = EncoderBlock(16, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        self.bottleneck = DilatedConvBlock(256, 512)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        output = self.final_conv(x)
        
        batch_size, num_classes, H, W = output.shape

        # Flatten H and W dimensions
        output_flat = output.view(batch_size, num_classes, H * W)

        # Compute softmax along spatial dimensions (H*W)
        soft_acts = F.softmax(output_flat, dim=-1)  # [batch_size, num_classes, H*W]

        # Precompute grid coordinates (only once)
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device=output.device), 
                                        torch.arange(H, device=output.device), indexing='xy')

        grid_x = grid_x.flatten().float()  # Shape: [H*W]
        grid_y = grid_y.flatten().float()  # Shape: [H*W]

        # Compute softmax-weighted positions for all batch & classes at once
        norm_x = (soft_acts * grid_x).sum(dim=-1) / W  # [batch_size, num_classes]
        norm_y = (soft_acts * grid_y).sum(dim=-1) / H  # [batch_size, num_classes]

        # Stack the results into shape [batch_size, num_classes, 2]
        max_pixels = torch.stack([norm_x, norm_y], dim=-1)  # Shape: [batch_size, num_classes, 2]

        return max_pixels

