import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        codebook_size,
        embedding_dim,
        block_size,
        n_layers,
        num_heads,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = num_heads
        self.codebook_size = codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.embedding_dim))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        # transformer
        self.blocks = nn.Sequential(
            *[Block(self.embedding_dim, self.n_heads) for _ in range(self.n_layers)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.head = nn.Linear(self.embedding_dim, self.codebook_size, bias=False)

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        t = token_embeddings.shape[1]
        assert (
            t <= self.block_size
        ), f"Cannot forward, model block size is exhausted., {t} vs {self.block_size}"
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.head(x)

        return logits
