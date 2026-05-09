# Copyright (c) OpenMMLab. Semantic cross-attention for SA-DINO encoder.
"""Inject tooth prior features into DINO's Deformable Encoder via
cross-attention inserted after each encoder layer's self-attention."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SemanticCrossAttention(nn.Module):
    """Lightweight multi-head cross-attention: encoder tokens attend to
    flattened tooth-prior feature tokens.

    query  = encoder memory tokens  [B, N_enc, C]
    key/value = tooth prior tokens  [B, N_prior, C]

    Uses standard (non-deformable) attention since prior maps are small.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 prior_channels: int = 64,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.prior_proj = nn.Linear(prior_channels, embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        self.norm = nn.LayerNorm(embed_dims)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,
                query: Tensor,
                prior_tokens: Tensor) -> Tensor:
        """
        Args:
            query: [B, N, C] encoder memory tokens.
            prior_tokens: [B, M, prior_channels] flattened prior features.
        Returns:
            [B, N, C] modulated encoder tokens.
        """
        kv = self.prior_proj(prior_tokens)
        attn_out, _ = self.attn(query, kv, kv, need_weights=False)
        return query + self.gamma * self.norm(attn_out)


def flatten_multi_scale_priors(
        priors: Tuple[Tensor, ...]) -> Tensor:
    """Flatten multi-scale prior feature maps into a single token sequence.

    Args:
        priors: tuple of [B, C, H_l, W_l] tensors from ToothEmbeddingEncoder.
    Returns:
        [B, sum(H_l*W_l), C] flattened tokens.
    """
    tokens = []
    for p in priors:
        b, c, h, w = p.shape
        tokens.append(p.reshape(b, c, h * w).permute(0, 2, 1))
    return torch.cat(tokens, dim=1)
