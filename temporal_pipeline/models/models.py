from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from monai.networks.nets import UNet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):

    def __init__(self, seq_len=1, pretrained=True):
        super(Model, self).__init__()

        print("---- Initializing model ----")

        network = EncoderDecoder(seq_len, pretrained)

        self.network = network

    def forward(self, x):
        x = self.network(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, device):
        super(UNet3D, self).__init__()

        self.device=device

        print("---- Initializing 3D_UNet ----")

        network = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                ).to(self.device)

        self.network = network

    def forward(self, x):
        x = self.network(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, seq_len=1):
        super(EncoderDecoder, self).__init__()

        self.seq_len = seq_len

        # determine if input is a sequence (3D) or a single image (2D)
        self.three_dim = False if seq_len == 1 else True

        # load a resnet50 backbone (not pretrained) TODO: could try with pretrained
        resnet = models.resnet152(weights = None)
        
        # remove the classification layers by replacing them with identity
        resnet.avgpool = Identity()
        resnet.fc = Identity()

        if self.three_dim:
            # use 3D convolution to handle temporal sequences
            # input shape: (batch, 1, seq_len, height, width)
            resnet.conv1 = nn.Conv3d(1, 64, kernel_size=(seq_len, 7, 7), stride=(1, 2, 2),
            padding=(0, 3, 3), bias=False)
            self.conv1 = list(resnet.children())[0]
            # store the rest of resnet layers
            self.resnet = nn.Sequential(*list(resnet.children()))[1:]

        # batch normalization layers for the decoder path
        self.bn5 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(1024)

        # transpose convolutions to upsample the feature maps back to original size
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)

        # final convolution to produce seq_len*3 output channels (one for each frame and one for each landmark)
        self.outconv = nn.Conv2d(64, 3*seq_len, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        # if using sequences, apply the 3D convolution first
        if self.three_dim:
            x = self.conv1(x)
            
        # reshape/flatten the temporal or depth dimension to process as 2D spatial features
        x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

        # pass through the resnet encoder
        x = self.resnet(x)

        # pass through the decoder (upsampling + activation + normalization)
        x = F.relu(self.bn1(self.upconv1(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn4(self.upconv4(x)))
        x = F.relu(self.bn5(self.upconv5(x)))

        # final output projection
        x = self.outconv(x)
        # reshape from (B, 3*seq__len, H, W) to (B, 3, seq_len, H, W)
        x = x.view(x.shape[0], 3, self.seq_len, x.shape[2], x.shape[3])
        
        return x


if __name__ == "__main__":

    # Test 3D mode (sequence of images)
    print("\nTesting 3D mode (seq_len=16)...")
    model_3d = EncoderDecoder(seq_len=16)
    x_3d = torch.randn(1, 1, 16, 224, 224)
    output_3d = model_3d(x_3d)
    print(f"Input shape: {x_3d.shape} -> Output shape: {output_3d.shape}")

    print("\nTest completed successfully!")


