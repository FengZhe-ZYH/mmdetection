# Copyright (c) OpenMMLab. Tooth-guided query initialization for SA-DINO.
"""Generate decoder query reference points and content embeddings from the
tooth semantic mask.  Unlike the original tooth_query_generator that only
uses connected-component bounding boxes, this module:
1. Uses per-tooth centroid (more stable than CC grid sampling)
2. Adds learnable offset MLP (since lesions are at root apex, not tooth center)
3. Injects tooth-ID embedding into query content (not just position)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ToothGuidedQueryInit(nn.Module):
    """Generate tooth-guided queries from semantic mask.

    For each image in the batch, extract up to ``max_tooth_queries`` tooth
    regions from the mask, produce normalized reference points with a
    learnable offset, and generate content embeddings from tooth-ID.
    """

    def __init__(self,
                 max_tooth_queries: int = 32,
                 num_tooth_classes: int = 34,
                 embed_dims: int = 256,
                 offset_hidden: int = 64) -> None:
        super().__init__()
        self.max_tooth_queries = max_tooth_queries
        self.num_tooth_classes = num_tooth_classes
        self.embed_dims = embed_dims

        self.tooth_content_embed = nn.Embedding(num_tooth_classes, embed_dims)

        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dims, offset_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(offset_hidden, 4),
        )
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def _extract_tooth_refs_single(
            self, mask_hw: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract centroids and bboxes for each tooth ID in a single mask.

        Args:
            mask_hw: [H, W] integer tensor of tooth IDs.
        Returns:
            refs: [N, 4] normalized cxcywh
            ids:  [N] tooth IDs (long)
        """
        h, w = mask_hw.shape
        unique_ids = mask_hw.unique()
        unique_ids = unique_ids[unique_ids > 0]

        refs_list: List[Tensor] = []
        ids_list: List[int] = []

        for tid in unique_ids:
            tid_val = int(tid.item())
            region = (mask_hw == tid)
            ys, xs = torch.where(region)
            if len(ys) < 4:
                continue
            cx = xs.float().mean() / w
            cy = ys.float().mean() / h
            x1 = xs.float().min() / w
            y1 = ys.float().min() / h
            x2 = xs.float().max() / w
            y2 = ys.float().max() / h
            bw = (x2 - x1).clamp(min=1e-4)
            bh = (y2 - y1).clamp(min=1e-4)
            refs_list.append(torch.tensor(
                [cx, cy, bw, bh], device=mask_hw.device, dtype=torch.float32))
            ids_list.append(tid_val)

        if not refs_list:
            dummy_ref = torch.tensor(
                [0.5, 0.5, 0.1, 0.1], device=mask_hw.device)
            return dummy_ref.unsqueeze(0), torch.zeros(
                1, device=mask_hw.device, dtype=torch.long)

        refs = torch.stack(refs_list, dim=0)
        ids = torch.tensor(ids_list, device=mask_hw.device, dtype=torch.long)
        return refs, ids

    def forward(
            self, tooth_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            tooth_mask: [B, 1, H, W] float tooth-ID mask.
        Returns:
            ref_points: [B, max_tooth_queries, 4] normalized cxcywh in (0,1).
            query_embed: [B, max_tooth_queries, embed_dims] content embeddings.
        """
        B = tooth_mask.shape[0]
        device = tooth_mask.device
        k = self.max_tooth_queries

        all_refs = []
        all_embeds = []

        for b in range(B):
            mask_hw = tooth_mask[b, 0].long().clamp(
                0, self.num_tooth_classes - 1)
            refs, ids = self._extract_tooth_refs_single(mask_hw)

            content = self.tooth_content_embed(ids)
            offsets = self.offset_mlp(content)
            refs_with_offset = refs + 0.1 * offsets

            n = refs_with_offset.shape[0]
            if n >= k:
                refs_with_offset = refs_with_offset[:k]
                content = content[:k]
            else:
                pad_n = k - n
                refs_with_offset = torch.cat([
                    refs_with_offset,
                    refs_with_offset[-1:].expand(pad_n, -1)
                ], dim=0)
                content = torch.cat([
                    content,
                    content[-1:].expand(pad_n, -1)
                ], dim=0)

            all_refs.append(refs_with_offset)
            all_embeds.append(content)

        ref_points = torch.stack(all_refs, dim=0).sigmoid()
        query_embed = torch.stack(all_embeds, dim=0)

        return ref_points, query_embed


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))
