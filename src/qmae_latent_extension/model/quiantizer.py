import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim, _freeze=True
        )
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer(
            "_ema_w", torch.Tensor(num_embeddings, self._embedding_dim)
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        input_shape = inputs.shape
        """BTC shape"""

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        """(flat_input - self._embedding.weight)^2"""

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw,
                requires_grad=False,
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1),
                requires_grad=False,
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape distances
        distances = distances.reshape(*quantized.shape[:2], -1)
        """
        distances shape - T x B x num_emb
        """

        distances = rearrange(distances, "t b c -> b t c")
        """distances shape - B x T x num_emb"""

        return loss, quantized, perplexity, encoding_indices, distances
