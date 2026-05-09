# Copyright (c) OpenMMLab. Tooth-local dense attention for SA-DINO v3b.
"""Post-decoder tooth-local dense cross-attention.

DINO's decoder uses deformable attention sampling only K=4 points per level.
This module provides complementary dense attention within each query's
associated tooth region, giving richer local spatial context for small
lesion localization.

Design:
  1. Map each query to a tooth ID via reference point + tooth mask
  2. Build per-tooth spatial masks over flattened encoder memory
  3. Dense multi-head cross-attention with tooth-based masking
  4. Apply as a post-decoder refinement to the last decoder layer output
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToothLocalAttention(nn.Module):
    """Dense cross-attention between decoder queries and local tooth regions.

    For each query, the tooth region is determined by looking up the tooth ID
    at the query's reference point in the mask. Standard MHA is applied with
    a mask that blocks out-of-tooth memory positions.

    Args:
        embed_dims: Feature embedding dimension (must match encoder/decoder).
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        ffn_dim: Hidden dim of the post-attention FFN (0 = no FFN).
        num_layers: Number of stacked local attention + FFN blocks.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        ffn_dim: int = 1024,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dims,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True))
            self.attn_norms.append(nn.LayerNorm(embed_dims))

            if ffn_dim > 0:
                self.ffn_layers.append(nn.Sequential(
                    nn.Linear(embed_dims, ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dims),
                    nn.Dropout(dropout),
                ))
                self.ffn_norms.append(nn.LayerNorm(embed_dims))

        self.gamma = nn.Parameter(torch.zeros(1))

    def _build_tooth_attn_mask(
        self,
        ref_points: Tensor,
        tooth_mask: Tensor,
        spatial_shapes: Tensor,
        num_tooth_classes: int = 34,
    ) -> Tensor:
        """Build [B, N, S] boolean attention mask based on tooth membership.

        True = masked (cannot attend), False = can attend.

        For each query, find its tooth ID from the reference point, then mark
        all memory positions belonging to the same tooth as attendable.
        Background (tooth_id=0) queries can attend to ALL positions.

        Args:
            ref_points: [B, N, 4] normalized (cx, cy, w, h) reference points.
            tooth_mask: [B, 1, H_img, W_img] tooth ID mask.
            spatial_shapes: [num_levels, 2] (h, w) of each level.
            num_tooth_classes: Total tooth classes.

        Returns:
            attn_mask: [B, N, S] boolean, True = masked.
        """
        B, N, _ = ref_points.shape
        device = ref_points.device
        img_h, img_w = tooth_mask.shape[2], tooth_mask.shape[3]
        mask_hw = tooth_mask[:, 0].long().clamp(0, num_tooth_classes - 1)

        query_cx = (ref_points[:, :, 0] * img_w).long().clamp(0, img_w - 1)
        query_cy = (ref_points[:, :, 1] * img_h).long().clamp(0, img_h - 1)

        query_tooth_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        for b in range(B):
            query_tooth_ids[b] = mask_hw[b, query_cy[b], query_cx[b]]

        token_tooth_ids_list = []
        for lvl, (h_l, w_l) in enumerate(spatial_shapes):
            h_l, w_l = int(h_l), int(w_l)
            lvl_mask = F.interpolate(
                mask_hw.unsqueeze(1).float(),
                size=(h_l, w_l),
                mode='nearest').long().squeeze(1)
            token_tooth_ids_list.append(lvl_mask.reshape(B, -1))
        token_tooth_ids = torch.cat(token_tooth_ids_list, dim=1)
        S = token_tooth_ids.shape[1]

        is_bg_query = (query_tooth_ids == 0)

        attn_mask = (query_tooth_ids.unsqueeze(2) !=
                     token_tooth_ids.unsqueeze(1))

        attn_mask[is_bg_query] = False

        all_masked = attn_mask.all(dim=2)
        attn_mask[all_masked] = False

        return attn_mask

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        ref_points: Tensor,
        tooth_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        num_tooth_classes: int = 34,
    ) -> Tensor:
        """Apply tooth-local dense attention.

        Args:
            query: [B, N, D] decoder output queries.
            memory: [B, S, D] encoder output memory.
            ref_points: [B, N, 4] normalized reference points (cx,cy,w,h).
            tooth_mask: [B, 1, H, W] tooth ID mask. If None, skip.
            spatial_shapes: [num_levels, 2] (h, w) per level.
            num_tooth_classes: Number of tooth classes.

        Returns:
            Refined query: [B, N, D].
        """
        if tooth_mask is None:
            return query

        attn_mask = self._build_tooth_attn_mask(
            ref_points, tooth_mask, spatial_shapes, num_tooth_classes)

        B, N, D = query.shape
        S = memory.shape[1]

        residual = query
        x = query

        for i in range(self.num_layers):
            attn_mask_expanded = attn_mask.unsqueeze(1).expand(
                B, self.attn_layers[i].num_heads, N, S).reshape(
                    B * self.attn_layers[i].num_heads, N, S)

            attn_out, _ = self.attn_layers[i](
                x, memory, memory,
                attn_mask=attn_mask_expanded,
                need_weights=False)
            x = self.attn_norms[i](x + attn_out)

            if i < len(self.ffn_layers):
                ffn_out = self.ffn_layers[i](x)
                x = self.ffn_norms[i](x + ffn_out)

        return residual + self.gamma * x
