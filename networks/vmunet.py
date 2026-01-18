from .vmamba import VSSM_conv
import torch
from torch import nn


class VMUNet_conv(nn.Module):
    def __init__(self,
                 input_channels,
                 num_classes,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path='vmamba_small_e238_ema.pth',
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM_conv(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits


if __name__ == '__main__':
    device = torch.device("cuda:1")
    x = torch.randn(4, 2, 512, 512).to(device)
    print(x.shape)
    net = VMUNet_conv(input_channels=2, num_classes=4).to(device)
    y = net(x)
    print(y.shape)
