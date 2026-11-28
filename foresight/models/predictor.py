"""Causal latent prediction and rollout with RGB/Event context."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import CrossAttentionBlock, rope_3d_tables


PREDICTOR_DEFAULTS = dict(hidden_dim=768, depth=8, heads=12, mlp_ratio=4.0,
                          dropout=0.0, query_chunk_size=1024, drop_path=0.0,
                          event_enabled=False, event_layers=None, gate_init=0.1)


class DINOWorldPredictor(nn.Module):
    """Predict future patch features using causal RGB and Event KV prefixes.

    ``predict_sequence`` runs parallel next-frame teacher forcing; ``forward``
    returns a single future frame. Every target has an observation cutoff that
    is distinct from its future timestamp. All times are seconds and shared
    across the batch. Callers supply prepared CPU time tensors; configuration
    and external input checks live in foresight.timing. Event tensors contain
    only encoded observed windows.
    """

    def __init__(self, grid_size=(32, 64), history_length=4, feature_dim=768,
                 hidden_dim=768, depth=8, heads=12, mlp_ratio=4.0, dropout=0.0,
                 query_chunk_size=1024, drop_path=0.0,
                 event_enabled=False, event_layers=None, gate_init=0.1):
        super().__init__()
        if len(grid_size) != 2 or any(type(x) is not int or x < 1 for x in grid_size):
            raise ValueError("grid_size must contain two positive integers")
        if any(type(x) is not int or x < 1 for x in (history_length, feature_dim, hidden_dim, depth, heads)):
            raise ValueError("Predictor dimensions and history_length must be positive integers")
        if hidden_dim != heads * 64:
            raise ValueError("DINO-world 3D RoPE requires exactly 64 channels per attention head")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and positive")
        if not 0 <= dropout < 1 or not 0 <= drop_path < 1:
            raise ValueError("dropout and drop_path must lie in [0,1)")
        if type(query_chunk_size) is not int or query_chunk_size < 0:
            raise ValueError("query_chunk_size must be a nonnegative integer")
        if type(event_enabled) is not bool or not math.isfinite(gate_init):
            raise ValueError("event_enabled must be bool and gate_init must be finite")
        if event_layers is None:
            event_layers = list(range(1, depth + 1)) if event_enabled else []
        else:
            event_layers = list(event_layers)
            if (any(type(x) is not int or not 1 <= x <= depth for x in event_layers)
                    or len(set(event_layers)) != len(event_layers)
                    or bool(event_layers) != event_enabled):
                raise ValueError("event_layers must be unique 1-based layer numbers for enabled Event attention")
        self.grid_size = tuple(grid_size)
        self.history_length, self.feature_dim = history_length, feature_dim
        self.query_chunk_size = query_chunk_size
        self.event_enabled = event_enabled
        self.activation_checkpointing = False
        # Chunk sizes/checkpointing change execution, not the model's weights.
        self.architecture = dict(grid_size=list(grid_size), history_length=history_length,
                                 feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth,
                                 heads=heads, mlp_ratio=float(mlp_ratio), dropout=float(dropout),
                                 drop_path=float(drop_path),
                                 event_enabled=event_enabled, event_layers=sorted(event_layers),
                                 rope="periods-1e-2-to-1e2-2pi-tyx-16-24-24-patch-centers-v31")
        self.context_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.event_proj = (nn.Linear(feature_dim, hidden_dim) if feature_dim != hidden_dim
                           else nn.Identity()) if event_enabled else None
        self.query_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.query_token, std=0.02)
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, heads, mlp_ratio, dropout,
                                index + 1 in event_layers, gate_init, rates[index])
            for index in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def _coordinates(self, times, grid_size=None):
        height, width = grid_size or self.grid_size
        # Compact Event KV uses cell centers in the RGB patch coordinate system.
        y = (torch.arange(height, device=times.device).float() + .5) * self.grid_size[0] / height - .5
        x = (torch.arange(width, device=times.device).float() + .5) * self.grid_size[1] / width - .5
        t, y, x = torch.meshgrid(times.float(), y, x, indexing="ij")
        return torch.stack((t, y, x), dim=-1).reshape(-1, 3)

    def _event_inputs(self, event_context, event_times, event_grid_size, context_times, event_off):
        if event_off:
            return None, None, None
        if not self.event_enabled:
            if event_context is not None:
                raise ValueError("event_context requires event_enabled=True")
            return None, None, None
        if event_context is None:
            raise ValueError("Event predictor requires event_context or event_off=True")
        grid = tuple(event_grid_size or self.grid_size)
        if event_times is None:
            event_times = context_times
        return event_context, event_times, grid

    @staticmethod
    def _prefixes(context_times, observation_times, query_spatial, context_spatial):
        """Plan contiguous query runs without assuming cutoffs are increasing."""
        counts = torch.searchsorted(context_times.detach().contiguous(),
                                    observation_times.detach().contiguous(), right=True).tolist()
        runs = []
        for frame, visible in enumerate(counts):
            start, end = frame * query_spatial, (frame + 1) * query_spatial
            visible *= context_spatial
            if runs and runs[-1][2] == visible:
                runs[-1] = (runs[-1][0], end, visible)
            else:
                runs.append((start, end, visible))
        return tuple(runs)

    def _predict(self, history, context_times, query_times, observation_times,
                 event_context, event_times, event_grid):
        """Execute one prediction from caller-prepared CPU times."""
        spatial = math.prod(self.grid_size)
        rgb_prefixes = self._prefixes(context_times, observation_times, spatial, spatial)
        event_prefixes = (self._prefixes(event_times, observation_times, spatial, math.prod(event_grid))
                          if event_context is not None else None)
        origin = context_times[0]
        # Shared positional tables contain no Q/K activations. They are built
        # before checkpointed blocks and reused by every layer and spatial Q/K.
        context_rope = rope_3d_tables(self._coordinates((context_times - origin).to(history.device)))
        query_rope = rope_3d_tables(self._coordinates((query_times - origin).to(history.device)))
        spatial_rope = rope_3d_tables(self._coordinates(history.new_zeros(1, dtype=torch.float64)))
        event_rope = (rope_3d_tables(self._coordinates((event_times - origin).to(history.device), event_grid))
                      if event_context is not None else None)
        events = (self.event_proj(event_context.reshape(
            history.shape[0], len(event_times) * math.prod(event_grid), self.feature_dim))
            if event_context is not None else None)
        context = self.context_proj(history.flatten(1, 2))
        query = self.query_token.expand(history.shape[0], len(query_times) * spatial, -1)
        for block in self.blocks:
            args = (query, context, query_rope, context_rope, rgb_prefixes,
                    spatial_rope, self.query_chunk_size, events, event_rope, event_prefixes)
            if self.activation_checkpointing and torch.is_grad_enabled():
                query = checkpoint(block, *args, use_reentrant=False)
            else:
                query = block(*args)
        return self.output_proj(self.norm(query)).reshape(
            history.shape[0], len(query_times), spatial, self.feature_dim)

    def predict_at(self, history, context_times, query_times, *, observation_times=None,
                   event_context=None, event_times=None, event_grid_size=None, event_off=False):
        if observation_times is None:
            indices = torch.searchsorted(context_times, query_times, right=False) - 1
            observation_times = context_times[indices]
        events, event_times, event_grid = self._event_inputs(
            event_context, event_times, event_grid_size, context_times, event_off)
        return self._predict(history, context_times, query_times, observation_times,
                             events, event_times, event_grid)

    def predict_sequence(self, history, context_times, query_times, *, event_context=None,
                         event_times=None, event_grid_size=None, event_off=False):
        """Predict each context's next frame, with its own observed-input cutoff."""
        events, event_times, event_grid = self._event_inputs(
            event_context, event_times, event_grid_size, context_times, event_off)
        return self._predict(history, context_times, query_times, context_times, events, event_times, event_grid)

    def forward(self, history, context_times, query_times, *, observation_times=None,
                event_context=None, event_times=None, event_grid_size=None, event_off=False):
        """Predict one explicitly supplied future timestamp."""
        prediction = self.predict_at(history, context_times, query_times, observation_times=observation_times,
                                     event_context=event_context, event_times=event_times,
                                     event_grid_size=event_grid_size, event_off=event_off)
        return prediction.reshape(history.shape[0], math.prod(self.grid_size), self.feature_dim)

    def rollout(self, history, context_times, query_times, *, event_context=None,
                event_times=None, event_grid_size=None, event_off=False):
        """Feed predictions back; observed Event context stays fixed at initial cutoff."""
        events, event_times, event_grid = self._event_inputs(
            event_context, event_times, event_grid_size, context_times, event_off)
        context = history[:, -self.history_length:]
        times, predictions = context_times[-self.history_length:], []
        for target in query_times.split(1):
            prediction = self._predict(context, times, target, times[-1:],
                                       events, event_times, event_grid)[:, 0]
            predictions.append(prediction)
            context = torch.cat((context, prediction[:, None]), dim=1)[:, -self.history_length:]
            times = torch.cat((times, target))[-self.history_length:]
        return torch.stack(predictions, dim=1)
