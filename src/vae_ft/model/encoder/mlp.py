from torch import nn


class MLPEncoder(nn.Module):
    """Gan generated MNIST images"""

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout == 0.0:
            self.module = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_dim, 512),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        else:
            self.module = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )

        self.mu_head = nn.Linear(256, self.output_dim)
        self.sigma_head = nn.Linear(256, self.output_dim)

    def forward(self, x):
        x = self.module(x)

        return self.mu_head(x), self.sigma_head(x)
