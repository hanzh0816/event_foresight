"""Frozen DINOv2 patch features shared by predictor and detector training."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ..vendor.dinov2.models.vision_transformer import vit_base


BACKBONE = dict(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
                block_chunks=0, num_register_tokens=4,
                interpolate_antialias=True, interpolate_offset=0.0)


class DINOFeatureEncoder(nn.Module):
    """Load the official frozen ViT-B/14-reg4 and return its last-layer tokens."""

    feature_space = "dinov2-vitb14-reg-last-layer"
    feature_dim = 768
    patch_size = 14

    def __init__(self, checkpoint, image_size=(448, 896), chunk_size=1):
        super().__init__()
        if len(image_size) != 2 or any(type(x) is not int or x <= 0 or x % 14 for x in image_size):
            raise ValueError("image_size must contain two positive multiples of 14")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        self.image_size = tuple(image_size)
        self.grid_size = tuple(x // self.patch_size for x in image_size)
        self.chunk_size = chunk_size
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 encoder weights not found: {path}. See README.md for manual download.")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            # The official pretrained asset may use PyTorch's legacy serialization.
            if "mmap" not in str(error):
                raise
            state = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(state, Mapping) or not state
                or any(not isinstance(v, torch.Tensor) for v in state.values())):
            raise ValueError("Expected the official raw DINOv2 ViT-B/14-reg4 tensor state_dict")
        with torch.device("meta"):
            self.backbone = vit_base(**BACKBONE)
        self.backbone.load_state_dict(state, strict=True, assign=True)
        if any(v.is_meta for v in self.backbone.state_dict().values()):
            raise RuntimeError("Encoder load did not materialize every tensor")
        if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
               for v in self.backbone.state_dict().values()):
            raise ValueError("Encoder weights must be finite FP32 tensors")
        self.architecture = dict(encoder="dinov2_vitb14_reg", backbone=deepcopy(BACKBONE),
                                 image_size=list(self.image_size), feature_space=self.feature_space,
                                 feature_dim=self.feature_dim)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        # A trainable caller must never enable encoder dropout or stochastic depth.
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb):
        """Patch tokens after the final LN; no CLS/registers, L2 norm or PCA."""
        pieces = []
        for batch in rgb.split(self.chunk_size):
            features, = self.backbone.get_intermediate_layers(batch, n=[11], reshape=False)
            pieces.append(features)
        return torch.cat(pieces)
