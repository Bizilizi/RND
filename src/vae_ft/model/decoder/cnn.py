import typing as t

import torch
import torch.nn as nn


class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        apply_sigmoid: bool,
        transforms=None,
        dropout: float = 0.0,
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
            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(dropout),
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
            z = torch.randn(num_samples, self.input_dim).to(device)
            generated = self.forward(z)
            generated = generated.reshape(-1, 1, 28, 28)

        if self.transforms:
            generated = self.transforms(generated)

        return generated
