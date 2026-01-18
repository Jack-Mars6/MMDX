import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU())
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.DoubleConv = DoubleConv(in_channels, 32)
        self.Down = nn.ModuleList()
        self.Up = nn.ModuleList()
        for i in range(4):
            Down_layer = Down(32 * (2 ** i), 32 * (2 ** (i + 1)))
            self.Down.append(Down_layer)
        for i in range(4):
            Up_layer = Up(32 * (2 ** (4 - i)), 32 * (2 ** (3 - i)))
            self.Up.append(Up_layer)
        self.OutConv = OutConv(32, out_channels)

    def forward(self, x):
        identity = x
        cache = []
        x = self.DoubleConv(x)
        cache.append(x)
        for i in range(4):
            x = self.Down[i](x)
            if i != 3:
                cache.append(x)
        for i in range(4):
            x = self.Up[i](x, cache[3 - i])
        x = self.OutConv(x)
        return x + identity


if __name__ == '__main__':
    import torch

    x = torch.zeros((9, 1, 512, 512))
    print(x.shape)
    net = UNet(in_channels=1, out_channels=2)
    y = net(x)
    print(y.shape)
    # print(net)

