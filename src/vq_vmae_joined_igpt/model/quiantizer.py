import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class FeatureQuantizer(nn.Module):
    def __init__(
        self,
        num_class_embeddings,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        top_k: int = 3,
        separate_codebooks: bool = True,
    ):
        super().__init__()

        self._num_embeddings = num_embeddings
        self._num_class_embeddings = num_class_embeddings

        self.feature_quantization = VectorQuantizerEMA(
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay,
            epsilon,
            top_k=top_k,
        )
        if separate_codebooks:
            self.class_quantization = VectorQuantizerEMA(
                num_class_embeddings,
                embedding_dim,
                commitment_cost,
                decay,
                epsilon,
                top_k=top_k,
            )
        else:
            self.class_quantization = self.feature_quantization

    def forward(self, features, return_distances=False):
        (
            class_vq_loss,
            quantized_class_emb,
            class_perplexity,
            class_indices,
            class_distances,
        ) = self.class_quantization(features[0][None])
        (
            feature_vq_loss,
            quantized_features,
            feature_perplexity,
            feature_indices,
            feature_distances,
        ) = self.feature_quantization(features[1:])
        """
        quantized_class_emb shape - 1 x B x top_k x emb_dim
        quantized_features shape  - T x B x top_k x emb_dim
        
        class_indices shape       - B x top_k
        feature_indices shape     - (BT) x top_k
        """

        # Shift and concatenate indices
        class_indices = class_indices.reshape(quantized_class_emb.shape[:3])
        feature_indices = (feature_indices + self._num_class_embeddings).reshape(
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
                        self._num_embeddings,
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
                        self._num_class_embeddings,
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

        distances = torch.cat([class_distances, feature_distances])
        """distances shape - (T+1) x B x num_class_emb + num_feature_emb"""

        encoding_indices = torch.cat([class_indices, feature_indices])
        quantized_features = torch.cat([quantized_class_emb, quantized_features])
        """
        encoding_indices shape   - (T+1) x B x top_k
        quantized_features shape - (T+1) x B x top_k x emb_dim
        """

        if return_distances:
            return (
                feature_vq_loss + class_vq_loss,
                quantized_features,
                feature_perplexity,
                class_perplexity,
                encoding_indices,
                distances,
            )
        else:
            return (
                feature_vq_loss + class_vq_loss,
                quantized_features,
                feature_perplexity,
                class_perplexity,
                encoding_indices,
            )


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        embedding,
        epsilon=1e-5,
        top_k: int = 3,
    ):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self._embedding = embedding

        self._decay = decay
        self._epsilon = epsilon

        self._k = top_k

    def forward(self, inputs, return_distances=False):
        input_shape = inputs.shape
        """B T emb_dim"""

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        embedding_weight = self._embedding.weight[: self._num_embeddings]
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2 * torch.matmul(flat_input, embedding_weight.t())
        )
        """(flat_input - self._embedding.weight)^2"""

        # Encoding
        encoding_indices = torch.topk(
            distances, k=self._k, dim=1, largest=False
        ).indices
        """
        encoding_indices shape - T x B x top_k
        """

        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = self._embedding(encoding_indices)
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

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape
        encoding_indices = encoding_indices.reshape(quantized.shape[:3])
        distances = distances.reshape(*quantized.shape[:2], -1)

        if return_distances:
            return loss, quantized, perplexity, encoding_indices, distances
        else:
            return loss, quantized, perplexity, encoding_indices
