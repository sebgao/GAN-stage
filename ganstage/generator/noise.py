import torch
import torch.nn as nn

from ..registry import GENERATOR

@GENERATOR.register_module
class UniformNoisyGenerator(nn.Module):
    def __init__(self, batch=1, shape=(100,), bandwidth=1.0, device='cpu'):
        super(UniformNoisyGenerator, self).__init__()
        self.batch = batch
        self.shape = shape
        self.bandwidth = bandwidth
        self.device = device
    
    def forward(self, placeholder=None, *args, **kwargs):
        device = self.device if hasattr(self, 'device') else None
        return torch.rand((self.batch,) + self.shape, device=device)*self.bandwidth
    
    def __repr__(self):
        return '{}(shape={}, bandwidth={})'.format(type(self).__name__, self.shape, self.bandwidth)

@GENERATOR.register_module
class GaussianNoisyGenerator(nn.Module):
    def __init__(self, batch=1, shape=(100,), bandwidth=1.0, device='cpu'):
        super(GaussianNoisyGenerator, self).__init__()
        self.batch = batch
        self.shape = shape
        self.bandwidth = bandwidth
        self.device = device
    
    def forward(self, placeholder=None, *args, **kwargs):
        device = self.device if hasattr(self, 'device') else None
        return torch.randn((self.batch,) + self.shape, device=device).clamp_(-self.bandwidth, self.bandwidth)
    
    def __repr__(self):
        return '{}(shape={}, bandwidth={})'.format(type(self).__name__, self.shape, self.bandwidth)