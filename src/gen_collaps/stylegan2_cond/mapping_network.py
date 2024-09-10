import torch
from torch import nn
import torch.nn.functional as F
from labml_nn.gan.stylegan import EqualizedLinear

class MappingNetwork(nn.Module):
    """
    <a id="mapping_network"></a>

    ## Mapping Network

    ![Mapping Network](mapping_network.svg)

    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, in_features: int, out_features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for _ in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(in_features, in_features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers, EqualizedLinear(in_features, out_features))

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)