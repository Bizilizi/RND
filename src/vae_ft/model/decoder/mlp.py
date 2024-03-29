import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        h_dim1: int,
        h_dim2: int,
        z_dim: int,
        transforms=None,
        apply_sigmoid=True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.apply_sigmoid = apply_sigmoid

        self.module = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim1, input_dim),
        )

        self.z_dim = z_dim
        self.transforms = transforms

    def forward(self, z):
        h = self.module(z)

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
