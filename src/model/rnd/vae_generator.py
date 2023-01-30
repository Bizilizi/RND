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
        apply_sigmoid=True,
    ) -> None:
        super().__init__()

        self.apply_sigmoid = apply_sigmoid
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        self.z_dim = z_dim
        self.transforms = transforms

    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)

        if self.apply_sigmoid:
            h = torch.sigmoid(h)

        return h

    def generate(
        self, num_samples: int, device: t.Optional[torch.device] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim).to(device)
            generated = self.forward(z)
            generated = generated.reshape(-1, 1, 28, 28)

        if self.transforms:
            generated = self.transforms(generated)

        return generated


class MNISTVaeCNNGenerator(nn.Module, ImageGenerator):
    """Gan generated MNIST images"""

    def __init__(
        self,
        input_dim: int,
        apply_sigmoid: bool,
        transforms=None,
    ) -> None:
        super().__init__()

        self.apply_sigmoid = apply_sigmoid
        self.transforms = transforms
        self.input_dim = input_dim
        self.output_dim = 28 * 28

        self.fc1 = nn.Linear(input_dim, 5 * 5 * 32)
        self.upscale_module = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(-1, 32, 5, 5)
        x = self.upscale_module(x)

        if self.apply_sigmoid:
            x = torch.sigmoid(x)

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
