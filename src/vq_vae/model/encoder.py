from torch import nn
from src.vq_vae.model.resnet import ResidualStack


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        regularization_dropout,
    ):
        super().__init__()

        self._model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_hiddens // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hiddens // 2,
                out_channels=num_hiddens,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ResidualStack(
                in_channels=num_hiddens,
                num_hiddens=num_hiddens,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                regularization_dropout=regularization_dropout,
            ),
        )

    def forward(self, inputs):
        return self._model(inputs)
