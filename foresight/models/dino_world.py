"""Paper-based DINO-world (arXiv:2507.19468), with offline DINOv2 weights.

The predictor is implemented here from the paper, not imported from an official
release. See README.md for the explicit choices where the paper is underspecified.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..vendor.dinov2.models.vision_transformer import vit_base


BACKBONE = dict(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp",
                block_chunks=0, num_register_tokens=4,
                interpolate_antialias=True, interpolate_offset=0.0)
PREDICTOR_DEFAULTS = dict(hidden_dim=768, depth=12, heads=12, mlp_ratio=4.0,
                          dropout=0.0, query_chunk_size=1024)


def apply_rope_3d(tokens, coordinates):
    """[B,heads,N,64]: rotate 20 channels per (seconds, y, x), retain four.

    Eq. 5 gives angular *periods*. We use angle=2*pi*coordinate/period and
    adjacent real/imaginary pairs. Trigonometry stays FP32 under autocast.
    """
    if tokens.ndim != 4 or tokens.shape[-1] != 64:
        raise ValueError("3D RoPE requires [B,heads,N,64]")
    if tuple(coordinates.shape) != (tokens.shape[-2], 3):
        raise ValueError("RoPE coordinates must have shape [N,3]")
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        periods = torch.logspace(-2, 2, 10, device=tokens.device, dtype=torch.float32)
        angles = coordinates.float()[..., None] * (2 * math.pi / periods)
        cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
        pairs = tokens[..., :60].float().reshape(*tokens.shape[:-1], 3, 10, 2)
        real, imaginary = pairs.unbind(-1)
        rotated = torch.stack((real * cosine - imaginary * sine,
                               real * sine + imaginary * cosine), dim=-1)
    return torch.cat((rotated.flatten(-3).to(tokens.dtype), tokens[..., 60:]), dim=-1)


class CrossAttentionBlock(nn.Module):
    """Queries update; projected context remains fixed across all blocks."""

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
                                 nn.Dropout(dropout))

    def forward(self, query, context, query_coordinates, context_coordinates,
                query_times, context_times, query_chunk_size):
        batch, count, dim = query.shape
        q = self.q(self.norm1(query)).reshape(batch, count, self.heads, 64).transpose(1, 2)
        k, v = self.kv(context).reshape(batch, -1, 2, self.heads, 64).unbind(2)
        q = apply_rope_3d(q, query_coordinates)
        k = apply_rope_3d(k.transpose(1, 2), context_coordinates)
        # Project K/V once per layer, then bound attention workspace by splitting
        # independent queries. Every chunk still reads the entire context.
        pieces = []
        for start in range(0, count, query_chunk_size or count):
            end = min(start + (query_chunk_size or count), count)
            allowed = query_times[start:end, None] > context_times[None]
            pieces.append(F.scaled_dot_product_attention(
                q[:, :, start:end], k, v.transpose(1, 2), attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0))
        value = torch.cat(pieces, dim=2)
        query = query + F.dropout(self.proj(value.transpose(1, 2).reshape(batch, count, dim)),
                                  p=self.dropout, training=self.training)
        return query + self.mlp(self.norm2(query))


class DINOWorldPredictor(nn.Module):
    """Shared learned queries cross-attend only to strictly earlier frame tokens.

    ``predict_sequence`` provides all next-frame teacher-forcing predictions.
    ``forward`` produces the one future map consumed by the existing detector.
    Time coordinates are seconds, shared across a batch; DSEC uses 0.05 s steps.
    """

    def __init__(self, grid_size=(32, 64), history_length=4, feature_dim=768,
                 hidden_dim=768, depth=12, heads=12, mlp_ratio=4.0, dropout=0.0,
                 frame_interval_s=0.05, query_chunk_size=1024):
        super().__init__()
        if len(grid_size) != 2 or any(type(x) is not int or x < 1 for x in grid_size):
            raise ValueError("grid_size must contain two positive integers")
        if any(type(x) is not int or x < 1 for x in (history_length, feature_dim, hidden_dim, depth, heads)):
            raise ValueError("Predictor dimensions and history_length must be positive integers")
        if hidden_dim != heads * 64:
            raise ValueError("DINO-world 3D RoPE requires exactly 64 channels per attention head")
        if not math.isfinite(frame_interval_s) or frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be finite and positive")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and positive")
        if not 0 <= dropout < 1 or type(query_chunk_size) is not int or query_chunk_size < 0:
            raise ValueError("dropout must lie in [0,1); query_chunk_size must be a nonnegative integer")
        self.grid_size = tuple(grid_size)
        self.history_length, self.feature_dim = history_length, feature_dim
        self.frame_interval_s, self.query_chunk_size = float(frame_interval_s), query_chunk_size
        self.activation_checkpointing = False
        # Chunk size is an execution choice, not part of model/weight identity.
        self.architecture = dict(grid_size=list(grid_size), history_length=history_length,
                                 feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth,
                                 heads=heads, mlp_ratio=float(mlp_ratio), dropout=float(dropout),
                                 frame_interval_s=self.frame_interval_s,
                                 rope="periods-1e-2-to-1e2-2pi-pairs-tyx-grid-endpoints-v1")
        self.context_proj = nn.Linear(feature_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, heads, mlp_ratio, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def _validate_history(self, history):
        if (history.ndim != 4 or history.shape[0] < 1 or history.shape[1] < 1
                or tuple(history.shape[2:]) != (math.prod(self.grid_size), self.feature_dim)
                or not history.is_floating_point()):
            raise ValueError("history must be floating [B,T,H*W,feature_dim] with nonempty B,T")

    @staticmethod
    def _times(values, count, device, name):
        # Shift timestamps to a local origin before FP32 conversion below.
        values = torch.as_tensor(values, device=device, dtype=torch.float64)
        if (values.ndim != 1 or len(values) != count or count < 1
                or not torch.isfinite(values).all()
                or (values[1:] <= values[:-1]).any()):
            raise ValueError(f"{name} must be finite, strictly increasing, and have length {count}")
        return values

    def _coordinates(self, times):
        height, width = self.grid_size
        # A singleton dimension sits at the center; otherwise use [-1,+1].
        y = torch.linspace(-1, 1, height, device=times.device) if height > 1 else times.new_zeros(1)
        x = torch.linspace(-1, 1, width, device=times.device) if width > 1 else times.new_zeros(1)
        t, y, x = torch.meshgrid(times.float(), y.float(), x.float(), indexing="ij")
        return torch.stack((t, y, x), dim=-1).reshape(-1, 3)

    def predict_at(self, history, context_times, query_times):
        self._validate_history(history)
        context_times = self._times(context_times, history.shape[1], history.device, "context_times")
        query_times = torch.as_tensor(query_times, device=history.device, dtype=torch.float64)
        query_times = self._times(query_times, query_times.numel(), history.device, "query_times")
        if (query_times <= context_times[0]).any():
            raise ValueError("Every query must have at least one strictly earlier context frame")
        spatial = math.prod(self.grid_size)
        origin = context_times[0]
        context_coordinates = self._coordinates(context_times - origin)
        query_coordinates = self._coordinates(query_times - origin)
        # Comparison in FP64 protects the strict boundary from rounding of seconds.
        context_time_tokens = context_times.repeat_interleave(spatial)
        query_time_tokens = query_times.repeat_interleave(spatial)
        context = self.context_proj(history.flatten(1, 2))
        total_queries = len(query_coordinates)
        query = self.query_token.expand(history.shape[0], total_queries, -1)
        for block in self.blocks:
            args = (query, context, query_coordinates, context_coordinates,
                    query_time_tokens, context_time_tokens, self.query_chunk_size)
            if self.activation_checkpointing and torch.is_grad_enabled():
                # Explicit arguments and module binding preserve the positions,
                # strict mask and block identity during backward recomputation.
                query = checkpoint(block, *args, use_reentrant=False)
            else:
                query = block(*args)
        return self.output_proj(self.norm(query)).reshape(
            history.shape[0], len(query_times), spatial, self.feature_dim)

    def _default_times(self, history):
        return torch.arange(history.shape[1], device=history.device, dtype=torch.float64) * self.frame_interval_s

    def predict_sequence(self, history):
        """Predictions for t=1..T, with context t=0..T-1 and a strict causal mask."""
        self._validate_history(history)
        context_times = self._default_times(history)
        # Reuse exact context timestamps for overlapping targets, avoiding
        # floating-point (t+dt)>next_t at a nominally equal time boundary.
        queries = torch.cat((context_times[1:], context_times[-1:] + self.frame_interval_s))
        return self.predict_at(history, context_times, queries)

    def forward(self, history, context_times=None, query_times=None):
        self._validate_history(history)
        times = self._default_times(history) if context_times is None else context_times
        times = self._times(times, history.shape[1], history.device, "context_times")
        queries = times[-1:] + self.frame_interval_s if query_times is None else query_times
        queries = torch.as_tensor(queries, device=history.device, dtype=torch.float64).reshape(-1)
        if len(queries) != 1:
            raise ValueError("forward requires one query timestamp; use predict_at for multiple frames")
        return self.predict_at(history, times, queries)[:, 0]

    def rollout(self, history, steps):
        """Autoregressive feature feedback with a bounded sliding history window."""
        self._validate_history(history)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        context, predictions = history[:, -self.history_length:], []
        for _ in range(steps):
            prediction = self(context)
            predictions.append(prediction)
            context = torch.cat((context, prediction[:, None]), dim=1)[:, -self.history_length:]
        return torch.stack(predictions, dim=1)


class DINOWorldForecaster(nn.Module):
    """Frozen last-layer DINOv2-reg features and a randomly initialized predictor."""

    feature_space = "dinov2-vitb14-reg-last-layer"
    feature_dim = latent_dim = 768
    patch_size = 14

    def __init__(self, encoder_checkpoint, image_size=(448, 896), history_length=4,
                 frame_interval_s=0.05, encoder_chunk_size=1, predictor_options=None):
        super().__init__()
        if len(image_size) != 2 or any(type(x) is not int or x <= 0 or x % 14 for x in image_size):
            raise ValueError("image_size must contain two positive multiples of 14")
        if type(encoder_chunk_size) is not int or encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        self.image_size = tuple(image_size)
        self.grid_size = self.grid_hw = tuple(x // 14 for x in image_size)
        self.history_length, self.encoder_chunk_size = history_length, encoder_chunk_size
        options = dict(PREDICTOR_DEFAULTS)
        supplied = dict(predictor_options or {})
        if set(supplied) - set(options):
            raise ValueError(f"Unknown predictor options: {sorted(set(supplied) - set(options))}")
        options.update(supplied)
        path = Path(encoder_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 encoder weights not found: {path}. See README.md for manual download.")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError as error:
            if "mmap" not in str(error):
                raise
            state = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(state, Mapping) or not state
                or any(not isinstance(v, torch.Tensor) for v in state.values())):
            raise ValueError("Expected the official raw DINOv2 ViT-B/14-reg4 tensor state_dict")
        with torch.device("meta"):
            self.encoder = vit_base(**BACKBONE)
        self.encoder.load_state_dict(state, strict=True, assign=True)
        if any(v.is_meta for v in self.encoder.state_dict().values()):
            raise RuntimeError("Encoder load did not materialize every tensor")
        if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
               for v in self.encoder.state_dict().values()):
            raise ValueError("Encoder weights must be finite FP32 tensors")
        self.encoder.requires_grad_(False).eval()
        self.predictor = DINOWorldPredictor(grid_size=self.grid_size, history_length=history_length,
                                            feature_dim=self.feature_dim, frame_interval_s=frame_interval_s,
                                            **options)
        self.architecture = dict(model="dino-world-paper-v1", encoder="dinov2_vitb14_reg",
                                 backbone=deepcopy(BACKBONE), image_size=list(image_size),
                                 feature_space=self.feature_space, feature_dim=self.feature_dim,
                                 predictor=deepcopy(self.predictor.architecture))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _validate_rgb(self, rgb, ndim):
        if (rgb.ndim != ndim or rgb.shape[0] < 1 or rgb.shape[-3] != 3
                or tuple(rgb.shape[-2:]) != self.image_size or not rgb.is_floating_point()):
            raise ValueError(f"Expected floating {ndim}-D RGB ending [3,{self.image_size}]")

    @torch.no_grad()
    def encode_latents(self, rgb):
        """Patch tokens after encoder's final LN; no CLS/registers, L2 norm or PCA."""
        self._validate_rgb(rgb, 4)
        pieces = []
        for batch in rgb.split(self.encoder_chunk_size):
            features, = self.encoder.get_intermediate_layers(batch, n=[11], reshape=False)
            if tuple(features.shape) != (len(batch), math.prod(self.grid_size), self.feature_dim):
                raise ValueError("DINOv2 returned an incompatible last-layer patch feature shape")
            pieces.append(features)
        return torch.cat(pieces)

    encode_dense = encode_latents

    def forecast_latents(self, history):
        self._validate_rgb(history, 5)
        if history.shape[1] != self.history_length:
            raise ValueError(f"Expected {self.history_length} historical RGB frames")
        encoded = self.encode_latents(history.flatten(0, 1))
        return self.predictor(encoded.reshape(history.shape[0], self.history_length,
                                              math.prod(self.grid_size), self.feature_dim))

    forecast = forecast_latents
    forward = forecast_latents
