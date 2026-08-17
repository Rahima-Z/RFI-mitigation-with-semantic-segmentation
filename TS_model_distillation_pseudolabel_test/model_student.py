import torch
import torch.nn as nn


class AEStudent(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.2):
        super().__init__()

        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)

        # --- Encodeur ---
        self.enc_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   # /4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # /8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # /16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # /32 (bottleneck)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # --- Décodeur (miroir symétrique) ---
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # /16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # /8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # /4
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),    # /2, kernel plus grand
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.dec_stem = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=7, stride=2, padding=3, output_padding=1),    # /1, kernel large pour affiner les contours
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Dropout2d(dropout_p),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )


    def forward(self, x):
        x = self.input_adapter(x)

        e0 = self.enc_stem(x)   # /2
        e1 = self.enc1(e0)      # /4
        e2 = self.enc2(e1)      # /8
        e3 = self.enc3(e2)      # /16
        e4 = self.enc4(e3)      # /32  <- bottleneck

        d4 = self.dec4(e4)      # /16
        d3 = self.dec3(d4)      # /8
        d2 = self.dec2(d3)      # /4
        d1 = self.dec1(d2)      # /2
        d0 = self.dec_stem(d1)  # /1

        out = self.head(d0)
        return out
    