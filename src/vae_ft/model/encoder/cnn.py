from torch import nn


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        input_chanel: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.input_chanel = input_chanel

        if dropout == 0.0:
            self.module = nn.Sequential(
                nn.Conv2d(self.input_chanel, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv2d(64, 16, kernel_size=2),
                nn.AvgPool2d(5),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(self.input_chanel, 3, kernel_size=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Conv2d(3, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=2),
                nn.AvgPool2d(5),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.mu_head = nn.Linear(16, self.output_dim)
        self.sigma_head = nn.Linear(16, self.output_dim)

    def forward(self, x):
        x = self.module(x)

        return self.mu_head(x), self.sigma_head(x)
