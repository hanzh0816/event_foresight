"""Offline DINO-Foresight with independently trainable model components.

Only normalized historical RGB is accepted. No weight downloads, torch.hub,
global source-path changes, or hidden freezing are performed here.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..vendor.dinov2.models.vision_transformer import vit_base
from ..vendor.dinof.attention_masked import MaskTransformer


DINO_FORMAT = "event-foresight-dinof-v1"
PCA_KEYS = ("mean", "std", "pca_mean", "pca_components")
DINO_ARCHITECTURE = {
    "image_size": [448, 896], "history_length": 4, "patch_size": 14,
    "d_layers": [2, 5, 8, 11], "feature_dim": 3072, "pca_dim": 1152,
    "hidden_dim": 1152, "depth": 12, "heads": 8, "dropout": 0.1,
    "backbone": {"name": "vit_base", "img_size": 518, "patch_size": 14,
                 "init_values": 1.0, "ffn_layer": "mlp", "block_chunks": 0,
                 "num_register_tokens": 4, "interpolate_antialias": True,
                 "interpolate_offset": 0.0},
}


def read_weights(checkpoint, expected_format):
    """New exports contain tensors and primitive metadata only."""
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, Mapping) or saved.get("format") != expected_format:
        raise ValueError(f"Expected {expected_format} model weights")
    state = saved.get("state_dict")
    if not isinstance(state, Mapping) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Model weights must contain a nonempty tensor state_dict")
    return saved


def internal_state_dict(official):
    """Preserve every official tensor while assigning component namespaces."""
    result = {}
    for key, value in official.items():
        if key.startswith("dino_v2."):
            name = "encoder." + key.removeprefix("dino_v2.")
        elif key in PCA_KEYS:
            name = "pca." + key
        elif key.startswith(("embed.", "maskvit.")) or key == "replace_vector":
            name = "predictor." + key
        else:
            raise ValueError(f"Unexpected official model tensor: {key}")
        result[name] = value
    return result


class PCAProjection(nn.Module):
    """Official fixed mean/std and PCA basis; projection stays differentiable."""

    def __init__(self, state, feature_dim=3072, pca_dim=1152):
        super().__init__()
        self.feature_dim, self.pca_dim = int(feature_dim), int(pca_dim)
        for key in PCA_KEYS:
            if key not in state:
                raise ValueError(f"Missing official PCA buffer: {key}")
            value = state[key]
            shape = (pca_dim, feature_dim) if key == "pca_components" else (feature_dim,)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32:
                raise ValueError(f"PCA {key} must be FP32 with shape {shape}")
            if not torch.isfinite(value).all():
                raise ValueError(f"PCA {key} contains nonfinite values")
            self.register_buffer(key, value.detach().clone())
        if (self.std <= 0).any():
            raise ValueError("PCA std must be strictly positive")

    def project(self, features):
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Dense feature dimension differs from the PCA basis")
        return (((features - self.mean) / self.std) - self.pca_mean) @ self.pca_components.T

    def inverse(self, latents):
        if latents.shape[-1] != self.pca_dim:
            raise ValueError("Latent dimension differs from the PCA basis")
        return (latents @ self.pca_components + self.pca_mean) * self.std + self.mean

    def forward(self, features):
        return self.project(features)


class DINOFeaturePredictor(nn.Module):
    """Published separable space/time Transformer, with a trainable forward."""

    def __init__(self, grid_size=(32, 64), history_length=4, pca_dim=1152,
                 hidden_dim=1152, depth=12, heads=8, dropout=0.1):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.history_length, self.pca_dim = history_length, pca_dim
        self.activation_checkpointing = False
        self.maskvit = MaskTransformer(
            shape=(history_length + 1, *self.grid_size), embedding_dim=pca_dim,
            hidden_dim=hidden_dim, depth=depth, heads=heads, mlp_dim=4 * hidden_dim,
            dropout=dropout, use_fc_bias=True, seperable_attention=True,
            seperable_window_size=1, use_first_last=False,
        )
        self.maskvit.fc_in = nn.Identity()
        self.embed = nn.Linear(pca_dim, hidden_dim)
        self.replace_vector = nn.Parameter(torch.empty(1, 1, 1, 1, hidden_dim))
        nn.init.normal_(self.replace_vector, std=0.02)

    @staticmethod
    def _block_forward(x, layer, shape):
        """Published window-size1 separable block, without attention export."""
        batch, frames, height, width, dim = shape
        temporal, spatial, feedforward = layer
        x = x.reshape(batch, frames, height * width, dim).permute(0, 2, 1, 3).reshape(batch * height * width, frames, dim)
        value, _ = temporal(x)
        x = x + value
        x = x.reshape(batch, height * width, frames, dim).permute(0, 2, 1, 3).reshape(batch * frames, height * width, dim)
        value, _ = spatial(x)
        x = x + value
        x = x.reshape(batch, frames * height * width, dim)
        return x + feedforward(x)

    def forward(self, history):
        expected = (self.history_length, self.grid_size[0] * self.grid_size[1], self.pca_dim)
        if history.ndim != 4 or tuple(history.shape[1:]) != expected or not history.is_floating_point():
            raise ValueError(f"Expected floating PCA history [B,{expected}], got {tuple(history.shape)}")
        batch = history.shape[0]
        history = history.reshape(batch, self.history_length, *self.grid_size, self.pca_dim)
        embedded = self.embed(history)
        # Keep this FP32 parameter. Casting it to embedded.dtype changes the
        # published BF16 path before positional embeddings and layer norms.
        future = self.replace_vector.expand(batch, 1, *self.grid_size, -1)
        tokens = torch.cat((embedded, future), dim=1)
        if self.activation_checkpointing and torch.is_grad_enabled():
            x = self.maskvit.pos_embd(self.maskvit.fc_in(tokens))
            shape = tuple(x.shape)
            x = self.maskvit.first_layer(x.flatten(1, 3))
            for layer in self.maskvit.transformer.layers:
                # Bind the current layer for its later backward recomputation.
                def block(value, current_layer=layer):
                    return self._block_forward(value, current_layer, shape)
                x = checkpoint(block, x, use_reentrant=False)
            prediction = self.maskvit.fc_out(self.maskvit.last_layer(x))
        else:
            prediction, = self.maskvit(tokens)
        return prediction.reshape(batch, self.history_length + 1, -1, self.pca_dim)[:, -1]


class DINOForecaster(nn.Module):
    """Four RGB frames -> future1152 PCA tokens or future3072 dense features.

    Caller owns eval/train mode, requires_grad and autocast precision. The
    registered baseline uses eval(), BF16 autocast, high matmul precision and
    encoder_chunk_size=1. Module construction never initializes CUDA.
    """

    def __init__(self, checkpoint, encoder_chunk_size=1):
        super().__init__()
        if type(encoder_chunk_size) is not int or encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        saved = read_weights(checkpoint, DINO_FORMAT)
        if saved.get("architecture") != DINO_ARCHITECTURE:
            raise ValueError("Weights do not match the fixed high-resolution ViT-B/14-reg architecture")
        self.architecture = deepcopy(saved["architecture"])
        self.image_size, self.grid_size = (448, 896), (32, 64)
        self.grid_hw = self.grid_size
        self.history_length, self.patch_size = 4, 14
        self.feature_dim, self.pca_dim = 3072, 1152
        self.d_layers = [2, 5, 8, 11]
        self.encoder_chunk_size = encoder_chunk_size
        self.pca = PCAProjection(saved["state_dict"])
        # Allocate metadata first; assign=True loads the exact mapped FP32
        # tensors without random initialization or a second giant weight copy.
        backbone = {k: v for k, v in self.architecture["backbone"].items() if k != "name"}
        with torch.device("meta"):
            self.encoder = vit_base(**backbone)
            self.predictor = DINOFeaturePredictor()
        self.load_state_dict(internal_state_dict(saved["state_dict"]), strict=True, assign=True)
        if any(v.is_meta for v in self.state_dict().values()):
            raise RuntimeError("Model weights did not materialize every tensor")
        self.provenance = deepcopy(saved.get("provenance", {}))

    def _validate_rgb(self, rgb, ndim):
        if rgb.ndim != ndim or rgb.shape[-3] != 3 or tuple(rgb.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected {ndim}-D RGB ending [3,{self.image_size}], got {tuple(rgb.shape)}")
        if not rgb.is_floating_point():
            raise TypeError("RGB must be floating point and ImageNet-normalized")
        if len(rgb) == 0:
            raise ValueError("RGB batch must not be empty")

    def encode_dense(self, rgb):
        self._validate_rgb(rgb, 4)
        pieces = []
        for batch in rgb.split(self.encoder_chunk_size):
            layers = self.encoder.get_intermediate_layers(batch, n=self.d_layers, reshape=False)
            features = torch.cat(layers, dim=-1)
            expected = (len(batch), self.grid_size[0] * self.grid_size[1], self.feature_dim)
            if tuple(features.shape) != expected:
                raise ValueError(f"Encoder returned {tuple(features.shape)}, expected {expected}")
            pieces.append(features)
        return torch.cat(pieces)

    def encode_latents(self, rgb):
        self._validate_rgb(rgb, 4)
        # Keep PCA matmuls in the same per-frame chunks as the canonical
        # BF16/high/chunk1 baseline, rather than projecting a concatenated batch.
        return torch.cat([self.pca.project(self.encode_dense(batch))
                          for batch in rgb.split(self.encoder_chunk_size)])

    def forecast_latents(self, history):
        """Normalized historical RGB [B,4,3,H,W] -> future PCA [B,N,1152]."""
        self._validate_rgb(history, 5)
        if history.shape[1] != self.history_length:
            raise ValueError("Exactly four historical RGB frames are required")
        batch = history.shape[0]
        latent = self.encode_latents(history.flatten(0, 1))
        return self.predictor(latent.reshape(batch, self.history_length, -1, self.pca_dim))

    def forecast(self, history):
        return self.pca.inverse(self.forecast_latents(history))

    def forward(self, history):
        return self.forecast(history)
