import torch
import torch.nn as nn

from ..registry import FACTORY, LIBRARY
from ..utils import instantiate

class TwoStageGAN(nn.Module):
    def __init__(self):
        super(TwoStageGAN, self).__init__()
    
    def train(self):
        pass

@FACTORY.register_module
class DCGAN(TwoStageGAN):
    def __init__(self, noise=None, generator=None, discriminator=None, loss=None):
        super(DCGAN, self).__init__()
        noise = LIBRARY.UniformNoisyGenerator if noise is None else noise
        generator = LIBRARY.DCGenerator if generator is None else generator
        discriminator = LIBRARY.DCDiscriminator if discriminator is None else discriminator
        loss = nn.NLLLoss if loss is None else loss

        self.noise = instantiate(noise)
        self.generator = instantiate(generator)
        self.discriminator = instantiate(discriminator)
        self.loss = instantiate(loss)
    