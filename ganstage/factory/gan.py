from functools import partial

import torch
import torch.nn as nn

from ..registry import FACTORY, LIBRARY
from ..utils import instantiate

REAL_LABEL = 1.0
FAKE_LABEL = 0.0

class TwoStageGAN(nn.Module):
    def __init__(self):
        super(TwoStageGAN, self).__init__()
    

@FACTORY.register_module
class DCGAN(TwoStageGAN):
    def __init__(self, noise=None, generator=None, discriminator=None, loss=None):
        super(DCGAN, self).__init__()

        noise = LIBRARY.UniformNoisyGenerator if noise is None else noise
        generator = LIBRARY.DCGenerator if generator is None else generator
        discriminator = LIBRARY.DCDiscriminator if discriminator is None else discriminator
        loss = nn.BCELoss if loss is None else loss

        self.noise = instantiate(noise)
        self.generator = instantiate(generator)
        self.discriminator = instantiate(discriminator)
        self.loss = instantiate(loss)

    
    def config(self, dataset, batch_size, optimizer=None):
        device = next(self.parameters()).device

        self.dataset = instantiate(dataset)
        self.batch_size = batch_size

        self.device = device
        self.noise.batch = batch_size
        self.noise.device = device

        optimizer = partial(torch.optim.Adam, lr=0.0002, betas=(0.5, 0.999), weight_decay=5*1e-4) if optimizer is None else optimizer
        self.g_optimizer = instantiate(optimizer, params=self.generator.parameters())
        self.d_optimizer = instantiate(optimizer, params=self.discriminator.parameters())

    def train_data(self, data):
        device = self.device

        data = data.to(device)
        batch_size = self.batch_size

        info = dict()

        label = torch.full((batch_size, ), REAL_LABEL, device=device)
        
        # Step 1: training the discriminator

        # Step 1.1: using true samples
        label.fill_(REAL_LABEL)
        output = self.discriminator(data)
        error = self.loss(output, label)
        error.backward()
        info['loss_d_real'] = error.mean().item()

        # Step 1.2: using fake samples
        label.fill_(FAKE_LABEL)
        fake = self.generator(self.noise())
        output = self.discriminator(fake.detach())
        error = self.loss(output, label)
        error.backward()
        info['loss_d_fake'] = error.mean().item()

        self.d_optimizer.step()

        # Step 2: training the generator
        self.generator.zero_grad()
        label.fill_(REAL_LABEL)
        output = self.discriminator(fake)
        error = self.loss(output, label)
        error.backward()
        info['loss_g'] = error.mean().item()

        self.g_optimizer.step()
        self.discriminator.zero_grad()

        return info

    def start_train(self):
        loader = self.dataset.loader(self.batch_size)
        len_loader = len(loader)
        losses = []
        for idx, (data, _) in enumerate(loader):
            self.train()
            info = self.train_data(data)
            loss = sum([info[key] for key in info if key.startswith('loss')])
            losses.append(loss)

        print(sum(losses)/len(losses))


    def forward(self):
        return self.generator(self.noise())