import torch
import torch.nn as nn

from ..registry import DISCRIMINATOR

@DISCRIMINATOR.register_module
class DCDiscriminator(nn.Sequential):
    def __init__(self, in_channel=3, start_channel=64, output_channel=1, levels=4, sigmoid=True):
        super(DCDiscriminator, self).__init__()
        self.add_module('base_conv', nn.Conv2d(in_channel, start_channel, 4, 2, 1, bias=False))
        channel = start_channel
        for _i in range(levels):
            out_channel = channel*2
            self.add_module('bn{}'.format(_i), nn.BatchNorm2d(channel))
            self.add_module('leaky_relu{}'.format(_i), nn.LeakyReLU(0.2, inplace=True))
            if _i == levels -1:
                self.add_module('conv{}'.format(_i), nn.Conv2d(channel, output_channel, 4, 1, 0, bias=False))
            else:
                self.add_module('conv{}'.format(_i), nn.Conv2d(channel, out_channel, 4, 2, 1, bias=False))
            channel = out_channel
        
        if sigmoid:
            self.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = super(DCDiscriminator, self).forward(x)
        return x.view(x.size(0))