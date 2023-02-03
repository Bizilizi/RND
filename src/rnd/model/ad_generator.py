import itertools
import typing as t

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm


class MNISTAutoDecoderLinearGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        transforms=None,
        apply_sigmoid=True,
    ) -> None:
        super().__init__()

        self.apply_sigmoid = apply_sigmoid
        self.z_dim = z_dim
        self.transforms = transforms

        # model
        self.decoder_1 = nn.Sequential(
            weight_norm(nn.Linear(self.z_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(128, 128 - self.z_dim)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decoder_2 = nn.Sequential(
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.output_layer = weight_norm(nn.Linear(128, 28 * 28))

    def forward(self, z):

        skip_connection = z
        x = self.decoder_1(z)
        x = torch.cat([x, skip_connection], dim=-1)
        x = self.decoder_2(x)
        x = self.output_layer(x)

        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        return x

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
