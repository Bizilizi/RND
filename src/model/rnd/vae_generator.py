import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.rnd.generator import ImageGenerator


class MNISTVaeLinearGenerator(nn.Module):
    def __init__(
        self,
        x_dim: int,
        h_dim1: int,
        h_dim2: int,
        z_dim: int,
        transforms=None,
    ) -> None:
        super().__init__()

        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        self.z_dim = z_dim
        self.transforms = transforms

    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = torch.sigmoid(self.fc6(h))
        h = h.reshape(-1, 1, 28, 28)

        return h

    def generate(
        self, num_samples: int, device: t.Optional[torch.device] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(device)
            generated = self.forward(z)

        if self.transforms:
            generated = self.transforms(generated)

        return generated


class MNISTVaeCNNGenerator(nn.Module, ImageGenerator):
    """Gan generated MNIST images"""

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 28 * 28

        self.fc1 = nn.Linear(input_dim, 25)
        self.upscale_module = nn.Sequential(
            nn.ConvTranspose2d(1, 3, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 5),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.reshape(-1, 1, 5, 5)
        x = self.upscale_module(x)
        x = torch.sigmoid(x)
        x = x.reshape(-1, 1, 28, 28)

        return x

    def generate(
        self, num_samples: int, device: t.Optional[torch.device] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            z = Variable(torch.randn(num_samples, self.input_dim).to(device))
            generated = self.forward(z)

        if self.transforms:
            generated = self.transforms(generated)

        return generated
