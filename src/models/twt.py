from typing import Tuple

import torch
import torch.nn as nn

from .cdcn_pp import CDCNpp
from .cct import SpatialCCT


class TwT(nn.Module):
    """Texture Weighted Transformer (TwT) combining CDCN++ and CCT branches."""

    def __init__(self, num_classes: int = 2, theta: float = 0.7,
                 embed_dim: int = 256):
        super().__init__()
        self.texture_branch = CDCNpp(num_classes=num_classes, theta=theta)
        self.spatial_branch = SpatialCCT(num_classes=num_classes, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits from both branches during training.

        Returns:
            logits_texture: CDCN++ branch logits (B, num_classes)
            logits_spatial: CCT branch logits (B, num_classes)
        """
        texture_weights, logits_texture = self.texture_branch(x)
        logits_spatial = self.spatial_branch(x, texture_weights)
        return logits_texture, logits_spatial

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference using only the spatial branch, as per paper."""
        # We still need texture weights for modulation, so compute them first.
        texture_weights, _ = self.texture_branch(x)
        logits = self.spatial_branch(x, texture_weights)
        return logits
