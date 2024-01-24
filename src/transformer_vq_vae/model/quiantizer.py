import typing as t

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


def safe_autocast(func):
    """
    Forward operation for quantization requires an autocast to 32.
    However, this operation is not supported by mps and cpu devices.
    """

    def decorator(self, inputs):
        if inputs.device.type in ["cpu", "mps"]:
            return func(self, inputs)
        else:
            with torch.autocast(inputs.device.type, dtype=torch.float32):
                return func(self, inputs)

    return decorator


class SeparateQuantizerOutput(t.NamedTuple):
    loss: torch.Tensor
    quantized: torch.Tensor
    class_perplexity: torch.Tensor
    feature_perplexity: torch.Tensor
    class_avg_probs: torch.Tensor
    feature_avg_probs: torch.Tensor
    encoding_indices: torch.Tensor
    distances: torch.Tensor


class SeparateCodebooksFeatureQuantizerEMA(nn.Module):
    """
    This class combines two codebooks. One for clas token and another for the rest
    of the tokens.
    """

    def __init__(
        self,
        num_class_embeddings,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        top_k: int = 3,
        class_perplexity_threshold: float = 0,
        patches_perplexity_threshold: float = 0,
    ):
        super().__init__()

        self.feature_quantization = FeatureQuantizerEMA(
            num_embeddings=num_embeddings,
            num_embeddings_per_step=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
            top_k=top_k,
            perplexity_threshold=patches_perplexity_threshold,
        )
        self.class_quantization = FeatureQuantizerEMA(
            num_embeddings=num_embeddings,
            num_embeddings_per_step=num_class_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
            top_k=top_k,
            perplexity_threshold=class_perplexity_threshold,
        )

    def update_model_experience(self, experience_step: int) -> None:
        # skip extension if its first experience
        if experience_step == 0:
            return

        self.class_quantization.extend_codebook()
        self.feature_quantization.extend_codebook()

    @safe_autocast
    def forward(self, features) -> SeparateQuantizerOutput:
        (
            class_vq_loss,
            quantized_class_emb,
            class_perplexity,
            class_avg_probs,
            class_indices,
            class_distances,
        ) = self.class_quantization(features[0][None])
        (
            feature_vq_loss,
            quantized_features,
            feature_perplexity,
            feature_avg_probs,
            feature_indices,
            feature_distances,
        ) = self.feature_quantization(features[1:])
        """
        quantized_class_emb shape - 1 x B x top_k x emb_dim
        quantized_features shape  - T x B x top_k x emb_dim
        
        class_indices shape       - B x top_k
        feature_indices shape     - (BT) x top_k
        """

        num_class_embeddings = self.class_quantization.embedding.weight.shape[0]
        num_embeddings = self.feature_quantization.embedding.weight.shape[0]

        # Shift and concatenate indices
        class_indices = class_indices.reshape(quantized_class_emb.shape[:3])
        feature_indices = (feature_indices + num_class_embeddings).reshape(
            quantized_features.shape[:3]
        )
        """
        class_indices shape   - 1 x B x top_k
        feature_indices shape - T x B x top_k
        """

        # Concatenate distances
        class_distances = class_distances.reshape(*quantized_class_emb.shape[:2], -1)
        feature_distances = feature_distances.reshape(*quantized_features.shape[:2], -1)
        """
        class_distances shape   - 1 x B x num_class_emb
        feature_distances shape - T x B x num_feature_emb
        """

        class_distances = torch.cat(
            [
                class_distances,
                torch.full(
                    (
                        class_distances.shape[0],
                        class_distances.shape[1],
                        num_embeddings,
                    ),
                    torch.finfo(torch.float32).max,
                    device=class_distances.device,
                ),
            ],
            dim=-1,
        )
        feature_distances = torch.cat(
            [
                torch.full(
                    (
                        feature_distances.shape[0],
                        feature_distances.shape[1],
                        num_class_embeddings,
                    ),
                    torch.finfo(torch.float32).max,
                    device=class_distances.device,
                ),
                feature_distances,
            ],
            dim=-1,
        )
        """
        class_distances shape   - 1 x B x num_class_emb + num_feature_emb
        feature_distances shape - T x B x num_class_emb + num_feature_emb
        """

        try:
            distances = torch.cat([class_distances, feature_distances])
            """distances shape - (T+1) x B x num_class_emb + num_feature_emb"""
        except:
            raise Exception(f"{class_distances.shape},{feature_distances.shape}")

        encoding_indices = torch.cat([class_indices, feature_indices])
        quantized_features = torch.cat([quantized_class_emb, quantized_features])
        """
        encoding_indices shape   - (T+1) x B x top_k
        quantized_features shape - (T+1) x B x top_k x emb_dim
        """

        return SeparateQuantizerOutput(
            loss=feature_vq_loss + class_vq_loss,
            quantized=quantized_features,
            class_perplexity=class_perplexity,
            feature_perplexity=feature_perplexity,
            class_avg_probs=class_avg_probs,
            feature_avg_probs=feature_avg_probs,
            encoding_indices=encoding_indices,
            distances=distances,
        )


class QuantizerOutput(t.NamedTuple):
    loss: torch.Tensor
    quantized: torch.Tensor
    perplexity: torch.Tensor
    avg_probs: torch.Tensor
    encoding_indices: torch.Tensor
    distances: torch.Tensor


class FeatureQuantizerEMA(nn.Module):
    experience_step: int

    def __init__(
        self,
        num_embeddings,
        num_embeddings_per_step,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        top_k: int = 3,
        perplexity_threshold: float = 0,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._num_embeddings_per_step = num_embeddings_per_step

        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self._commitment_cost = commitment_cost
        self._perplexity_threshold = perplexity_threshold
        self._reset_counter = 200 * 400 - 100

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim),
            requires_grad=False,
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self._k = top_k

    def update_model_experience(self, experience_step: int) -> None:
        # skip extension if its first experience
        if experience_step == 0:
            return

        self.extend_codebook()

    def reset_codebook(self, inputs, encodings):
        # Find embeddings that wasn't used in this batch
        encodings_count = encodings.sum(dim=0)
        # avoid resetting old fixed embeddings
        encodings_count[: -self._num_embeddings] = 1

        num_embeddings_to_reset = (encodings_count == 0).float().sum().long()

        # Randomly select inputs
        flat_inputs = inputs.flatten(0, 2)
        random_idx = torch.randperm(flat_inputs.shape[0])[:num_embeddings_to_reset]

        # Replace embedding with random input from batch
        self.embedding.weight.data[encodings_count == 0] = flat_inputs[random_idx]

    def extend_codebook(self) -> None:

        # Extend embeddings
        embedding_copy = self.embedding
        num_embedding = embedding_copy.weight.shape[0]

        self.embedding = nn.Embedding(
            num_embedding + self._num_embeddings_per_step,
            self._embedding_dim,
        )
        self.embedding.weight.data[:num_embedding] = embedding_copy.weight.data
        self.embedding.weight.requires_grad = False

        # Extend ema vector
        ema_w_copy = self._ema_w
        self._ema_w = nn.Parameter(
            torch.Tensor(
                num_embedding + self._num_embeddings_per_step, self._embedding_dim
            )
        )
        self._ema_w.data.normal_()
        self._ema_w.data[:num_embedding] = ema_w_copy.data

        # Extend ema cluster size
        self._ema_cluster_size = torch.cat(
            [
                self._ema_cluster_size,
                torch.zeros(self._num_embeddings_per_step).to(
                    self._ema_cluster_size.device
                ),
            ]
        )

    @safe_autocast
    def forward(self, inputs) -> QuantizerOutput:
        input_shape = inputs.shape
        num_embeddings = self.embedding.weight.shape[0]
        """T B emb_dim"""

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        """(flat_input - self._embedding.weight)^2"""

        # Encoding
        encoding_indices = torch.topk(
            distances, k=self._k, dim=1, largest=False
        ).indices
        encodings = torch.zeros(
            encoding_indices.shape[0], num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)
        """(T B) x num_embeddings"""

        # Quantize and unflatten
        quantized = self.embedding(encoding_indices)
        """
        Since we used top_k closest elements, dimensionality of 
        quantized is - (B T) x top_k x emb_dim
        """

        # Reshape input and quantized
        quantized = rearrange(quantized, "(b t) k c -> b t k c", b=input_shape[0])
        inputs = repeat(inputs, "b t c -> b t k c", k=self._k)
        """
        quantized shape - B x T x top_k x emb_dim
        inputs shape    - B x T x top_k x emb_dim
        """

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)
            # {num_embeddings} * decay + (1- decay) * num_embeddings

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            updated_embeddings = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            self.embedding.weight.data[-self._num_embeddings :] = updated_embeddings[
                -self._num_embeddings :
            ]

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)  # num_embeddings
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape
        encoding_indices = encoding_indices.reshape(quantized.shape[:3])
        distances = distances.reshape(*quantized.shape[:2], -1)

        if perplexity < self._perplexity_threshold and self._reset_counter >= 200 * 400:
            self.reset_codebook(inputs, encodings)
            self._reset_counter = 0
        self._reset_counter += 1

        return QuantizerOutput(
            loss=loss,
            quantized=quantized,
            perplexity=perplexity,
            avg_probs=avg_probs,
            encoding_indices=encoding_indices,
            distances=distances,
        )
