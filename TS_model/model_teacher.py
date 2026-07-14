import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Segmentation(nn.Module):
    def __init__(self, num_classes=2, pretrained=False, dropout_p=0.2):
        super().__init__()

        weights = None
        backbone = resnet18(weights=weights)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)

        self.enc_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.enc_pool = backbone.maxpool
        self.enc_layer1 = backbone.layer1
        self.enc_layer2 = backbone.layer2
        self.enc_layer3 = backbone.layer3
        self.enc_layer4 = backbone.layer4

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec_stem = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Dropout2d(dropout_p),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_adapter(x)

        e1 = self.enc_stem(x)      # /2
        e2 = self.enc_pool(e1)     # /4
        e3 = self.enc_layer1(e2)   # /4
        e4 = self.enc_layer2(e3)   # /8
        e5 = self.enc_layer3(e4)   # /16
        e6 = self.enc_layer4(e5)   # /32

        d4 = self.dec4(e6)         # /16
        d3 = self.dec3(d4)         # /8
        d2 = self.dec2(d3)         # /4
        d1 = self.dec1(d2)         # /4
        d1 = self.up(d1)           # /2
        d0 = self.dec_stem(d1)     # /1

        out = self.head(d0)
        return out


