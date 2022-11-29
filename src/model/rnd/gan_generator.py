import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.model.rnd.generator import ImageGenerator


class MNISTGanGenerator(nn.Module, ImageGenerator):
    """Gan generated MNIST images"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

    def generate(
        self, num_samples: int, device: t.Optional[torch.device] = None
    ) -> torch.Tensor:

        with torch.no_grad():
            z = Variable(torch.randn(num_samples, self.input_dim).to(device))
            generated = self.forward(z)

        return generated.resize(generated.shape[0], 1, 28, 28)
