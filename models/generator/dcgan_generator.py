import torch
import torch.nn as nn

from ..registry import GENERATOR

class View2D(nn.Module):
    def __init__(self, size=(1, 1)):
        super(View2D, self).__init__()
        self.size = size
    
    def forward(self, x):
        return x.view(x.size(0), -1, *self.size)

@GENERATOR.register_module
class UniformNoisyGenerator(nn.Module):
    def __init__(self, batch=1, shape=(100,), bandwidth=1.0):
        super(UniformNoisyGenerator, self).__init__()
        self.batch = batch
        self.shape = shape
        self.bandwidth = bandwidth
    
    def forward(self, placeholder=None):
        device=self.device if hasattr(self, 'device') else None
        return torch.rand((self.batch,) + self.shape, device=device)*self.bandwidth

@GENERATOR.register_module
class DCGenerator(nn.Sequential):
    def __init__(self, input_size=100, start_size=(4, 4), start_channel=1024, output_channel=3, levels=4):
        super(DCGenerator, self).__init__()
        self.add_module('linear', nn.Linear(input_size, start_size[0]*start_size[1]*start_channel))
        self.add_module('view2d', View2D(start_size))
        channel = start_channel
        for _i in range(levels):
            out_channel = channel//2
            if _i == levels -1:
                out_channel = output_channel
            self.add_module('bn{}'.format(_i), nn.BatchNorm2d(channel))
            self.add_module('relu{}'.format(_i), nn.ReLU(inplace=True))
            self.add_module('transposed_conv2d{}'.format(_i), nn.ConvTranspose2d(channel, out_channel, 4, 2, 1, bias=False))
            channel = out_channel
        
        self.add_module('tanh', nn.Tanh())
        