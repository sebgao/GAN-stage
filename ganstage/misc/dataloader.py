import torch
import torchvision

from ..registry import MISC

def init(i):
    import numpy as np
    np.random.seed(0xFFFFFFFF & (torch.initial_seed()))
    pass

@MISC.register_module
class StandardImageFolder:
    def __init__(self, root, size=(64, 64), preprocess=None):
        if preprocess is None:
            preprocess = []
        transform = torchvision.transforms.Compose(preprocess + [
            torchvision.transforms.Resize(size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = torchvision.datasets.ImageFolder(root=root, transform=transform)

    def loader(self, batch_size):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=init, drop_last=True, pin_memory=True)
