import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        num_embeddings_per_step,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_frozen_embeddings = 0

        self._num_embeddings_per_step = num_embeddings_per_step

        self._embedding = nn.Embedding(
            self.num_embeddings, self.embedding_dim, _freeze=True
        )
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        self.register_buffer(
            "_ema_cluster_size", torch.zeros(num_embeddings, requires_grad=False)
        )
        self.register_buffer(
            "_ema_w",
            torch.zeros(
                (num_embeddings, self.embedding_dim), requires_grad=False
            ).normal_(),
        )

        self._decay = decay
        self._epsilon = epsilon

    def extend_codebook(self) -> None:
        # Extend embeddings
        embedding_copy = self._embedding.weight.data.clone()
        num_embedding = embedding_copy.shape[0]

        self._embedding = nn.Embedding(
            num_embedding + self._num_embeddings_per_step,
            self.embedding_dim,
            _freeze=True,
        )
        self._embedding.weight.data.normal_()
        self._embedding.weight.data[:num_embedding] = embedding_copy

        # Extend ema vector
        ema_w_copy = self._ema_w.data.clone()
        self._ema_w = torch.zeros(
            (num_embedding + self._num_embeddings_per_step, self.embedding_dim),
            requires_grad=False,
        ).normal_()
        self._ema_w.data[:num_embedding] = ema_w_copy

        # Extend ema cluster size
        self._ema_cluster_size = torch.cat(
            [
                self._ema_cluster_size,
                torch.zeros(self._num_embeddings_per_step).to(
                    self._ema_cluster_size.device
                ),
            ],
        )
        self._ema_cluster_size.requires_grad = False

        # Set new embedding num
        # self.num_frozen_embeddings = self.num_embeddings
        self.num_embeddings = self.num_embeddings + self._num_embeddings_per_step

    def forward(self, inputs, return_distances=False):
        input_shape = inputs.shape
        """BTC shape"""

        # Flatten input
        flat_input = inputs.reshape(-1, self.embedding_dim)

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
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                    1 - self._decay
                ) * torch.sum(encodings, 0)

                # Laplace smoothing of the cluster size
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self.num_embeddings * self._epsilon)
                    * n
                )

                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

                new_embeddings = self._ema_w / self._ema_cluster_size.unsqueeze(1)
                self._embedding.weight.data[
                    self.num_frozen_embeddings :
                ] = new_embeddings[self.num_frozen_embeddings :]

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

        if return_distances:
            return (
                loss,
                quantized,
                perplexity,
                encoding_indices,
                avg_probs,
                distances,
            )
        else:
            return (
                loss,
                quantized,
                perplexity,
                encoding_indices,
                avg_probs,
            )
