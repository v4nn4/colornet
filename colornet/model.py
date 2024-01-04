import logging

import torch
from torch import nn
import torch.nn.functional as F

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ColorizationUNet(nn.Module):
    def __init__(self, mean, std):
        super(ColorizationUNet, self).__init__()
        self.mean = mean
        self.std = std
        # Define the encoder path
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # ... (more layers as needed)

        # Define the decoder path using ConvTranspose2d for upsampling
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.dec_conv2 = nn.ConvTranspose2d(
            64 + 64, 32, kernel_size=3
        )  # Notice the channel size adjustment for concatenation
        self.dec_conv3 = nn.ConvTranspose2d(
            32 + 32, 3, kernel_size=3
        )  # Notice the channel size adjustment for concatenation
        # ... (more layers as needed)

        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.apply(initialize_weights)

    def forward(self, x):
        x = (x - self.mean) / self.std
        # Encoder
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(enc1))
        enc3 = F.relu(self.enc_conv3(enc2))
        # ... (more layers as needed)

        # Decoder
        dec1 = self.dec_conv1(enc3)  # Upsample
        dec1 = torch.cat((dec1, enc2), dim=1)  # Concatenate skip connection
        dec1 = F.relu(dec1)
        dec2 = self.dec_conv2(dec1)  # Upsample
        dec2 = torch.cat((dec2, enc1), dim=1)  # Concatenate skip connection
        dec2 = F.relu(dec2)

        dec3 = self.dec_conv3(dec2)
        # ... (more layers and upsampling as needed)

        # Final layer should map back to the original number of channels
        return self.sigmoid(dec3)


class Encoder(nn.Module):
    def __init__(self, mean: float, std: float):
        super(Encoder, self).__init__()
        self.mean = mean
        self.std = std
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2, return_indices=True)

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.conv1.weight, gain)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight, gain)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_normal_(self.conv3.weight, gain)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsample from 32x32x4 to the original image size 256x256x1
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=5
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=5
        )
        self.upconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To ensure the output is in the range [0, 1]

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.upconv1.weight, gain)
        nn.init.zeros_(self.upconv1.bias)
        nn.init.xavier_normal_(self.upconv2.weight, gain)
        nn.init.zeros_(self.upconv2.bias)
        nn.init.xavier_normal_(self.upconv3.weight, gain)
        nn.init.zeros_(self.upconv3.bias)

    def forward(self, x):
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.sigmoid(self.upconv3(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self, mean: float, std: float):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(mean, std)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
