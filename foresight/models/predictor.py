"""Causal latent prediction and rollout with RGB/Event context."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import CrossAttentionBlock


PREDICTOR_DEFAULTS = dict(hidden_dim=768, depth=8, heads=12, mlp_ratio=4.0,
                          dropout=0.0, query_chunk_size=1024, drop_path=0.0,
                          event_enabled=False, event_layers=None, gate_init=0.1)


class DINOWorldPredictor(nn.Module):
    """Predict future patch features using independently masked RGB and Event KV.

    ``predict_sequence`` runs parallel next-frame teacher forcing; ``forward``
    returns a single future frame. Every target has an observation cutoff that
    is distinct from its future timestamp. All times are seconds and shared
    across the batch. Event tensors contain only encoded observed windows.
    """

    def __init__(self, grid_size=(32, 64), history_length=4, feature_dim=768,
                 hidden_dim=768, depth=8, heads=12, mlp_ratio=4.0, dropout=0.0,
                 frame_interval_s=0.05, query_chunk_size=1024, drop_path=0.0,
                 event_enabled=False, event_layers=None, gate_init=0.1):
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
        self.frame_interval_s, self.query_chunk_size = float(frame_interval_s), query_chunk_size
        self.event_enabled = event_enabled
        self.activation_checkpointing = False
        # Chunk sizes/checkpointing change execution, not the model's weights.
        self.architecture = dict(grid_size=list(grid_size), history_length=history_length,
                                 feature_dim=feature_dim, hidden_dim=hidden_dim, depth=depth,
                                 heads=heads, mlp_ratio=float(mlp_ratio), dropout=float(dropout),
                                 frame_interval_s=self.frame_interval_s, drop_path=float(drop_path),
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

    def _validate_history(self, history):
        if (history.ndim != 4 or history.shape[0] < 1 or history.shape[1] < 1
                or tuple(history.shape[2:]) != (math.prod(self.grid_size), self.feature_dim)
                or not history.is_floating_point()):
            raise ValueError("history must be floating [B,T,H*W,feature_dim] with nonempty B,T")

    @staticmethod
    def _times(values, count, device, name):
        values = torch.as_tensor(values, device=device, dtype=torch.float64)
        if (values.ndim != 1 or len(values) != count or count < 1
                or not torch.isfinite(values).all()
                or (values[1:] <= values[:-1]).any()):
            raise ValueError(f"{name} must be finite, strictly increasing, and have length {count}")
        return values

    def _coordinates(self, times, grid_size=None):
        height, width = grid_size or self.grid_size
        # Compact Event KV uses cell centers in the RGB patch coordinate system.
        y = (torch.arange(height, device=times.device).float() + .5) * self.grid_size[0] / height - .5
        x = (torch.arange(width, device=times.device).float() + .5) * self.grid_size[1] / width - .5
        t, y, x = torch.meshgrid(times.float(), y, x, indexing="ij")
        return torch.stack((t, y, x), dim=-1).reshape(-1, 3)

    def _observation_times(self, context_times, query_times, observation_times):
        if observation_times is None:
            indices = torch.searchsorted(context_times, query_times, right=False) - 1
            if (indices < 0).any():
                raise ValueError("Every query must have at least one strictly earlier context frame")
            observation_times = context_times[indices]
        else:
            observation_times = torch.as_tensor(observation_times, device=context_times.device,
                                               dtype=torch.float64)
        if (observation_times.shape != query_times.shape
                or not torch.isfinite(observation_times).all()
                or (observation_times < context_times[0]).any()
                or (observation_times >= query_times).any()):
            raise ValueError("Each observation_time must precede its query and allow at least one RGB frame")
        return observation_times

    def _event_inputs(self, event_context, event_times, event_grid_size, history, context_times, event_off):
        if event_off:
            return None, None, None
        if not self.event_enabled:
            if event_context is not None:
                raise ValueError("event_context requires event_enabled=True")
            return None, None, None
        if event_context is None:
            raise ValueError("Event predictor requires event_context or event_off=True")
        grid = tuple(event_grid_size or self.grid_size)
        if (len(grid) != 2 or any(type(x) is not int or x < 1 for x in grid)
                or event_context.ndim != 4 or event_context.shape[0] != history.shape[0]
                or event_context.shape[1] < 1
                or tuple(event_context.shape[2:]) != (math.prod(grid), self.feature_dim)
                or not event_context.is_floating_point() or event_context.device != history.device):
            raise ValueError("event_context must be floating [B,Te,He*We,feature_dim] on the history device")
        if event_times is None:
            if event_context.shape[1] != len(context_times):
                raise ValueError("event_times is required when Event and RGB context lengths differ")
            event_times = context_times
        event_times = self._times(event_times, event_context.shape[1], history.device, "event_times")
        return self.event_proj(event_context.flatten(1, 2)), event_times, grid

    def predict_at(self, history, context_times, query_times, *, observation_times=None,
                   event_context=None, event_times=None, event_grid_size=None, event_off=False):
        self._validate_history(history)
        context_times = self._times(context_times, history.shape[1], history.device, "context_times")
        query_times = torch.as_tensor(query_times, device=history.device, dtype=torch.float64)
        query_times = self._times(query_times, query_times.numel(), history.device, "query_times")
        observation_times = self._observation_times(context_times, query_times, observation_times)
        events, event_times, event_grid = self._event_inputs(
            event_context, event_times, event_grid_size, history, context_times, event_off)
        spatial = math.prod(self.grid_size)
        origin = context_times[0]
        context_coordinates = self._coordinates(context_times - origin)
        query_coordinates = self._coordinates(query_times - origin)
        spatial_coordinates = self._coordinates(context_times.new_zeros(1))
        context_time_tokens = context_times.repeat_interleave(spatial)
        observation_tokens = observation_times.repeat_interleave(spatial)
        event_coordinates = (self._coordinates(event_times - origin, event_grid)
                             if events is not None else None)
        event_time_tokens = (event_times.repeat_interleave(math.prod(event_grid))
                             if events is not None else None)
        context = self.context_proj(history.flatten(1, 2))
        query = self.query_token.expand(history.shape[0], len(query_coordinates), -1)
        for block in self.blocks:
            args = (query, context, query_coordinates, context_coordinates,
                    observation_tokens, context_time_tokens, spatial_coordinates,
                    self.query_chunk_size, events, event_coordinates, event_time_tokens)
            if self.activation_checkpointing and torch.is_grad_enabled():
                query = checkpoint(block, *args, use_reentrant=False)
            else:
                query = block(*args)
        return self.output_proj(self.norm(query)).reshape(
            history.shape[0], len(query_times), spatial, self.feature_dim)

    def _default_times(self, history):
        return torch.arange(history.shape[1], device=history.device, dtype=torch.float64) * self.frame_interval_s

    def predict_sequence(self, history, *, context_times=None, event_context=None,
                         event_times=None, event_grid_size=None, event_off=False):
        """Predict each context's next frame, with its own observed-input cutoff."""
        self._validate_history(history)
        context_times = self._default_times(history) if context_times is None else context_times
        context_times = self._times(context_times, history.shape[1], history.device, "context_times")
        queries = torch.cat((context_times[1:], context_times[-1:] + self.frame_interval_s))
        return self.predict_at(history, context_times, queries, observation_times=context_times,
                               event_context=event_context, event_times=event_times,
                               event_grid_size=event_grid_size, event_off=event_off)

    def forward(self, history, context_times=None, query_times=None, *, observation_times=None,
                event_context=None, event_times=None, event_grid_size=None, event_off=False):
        self._validate_history(history)
        times = self._default_times(history) if context_times is None else context_times
        times = self._times(times, history.shape[1], history.device, "context_times")
        queries = times[-1:] + self.frame_interval_s if query_times is None else query_times
        queries = torch.as_tensor(queries, device=history.device, dtype=torch.float64).reshape(-1)
        if len(queries) != 1:
            raise ValueError("forward requires one query timestamp; use predict_at for multiple frames")
        return self.predict_at(history, times, queries, observation_times=observation_times,
                               event_context=event_context, event_times=event_times,
                               event_grid_size=event_grid_size, event_off=event_off)[:, 0]

    def rollout(self, history, steps, *, context_times=None, event_context=None,
                event_times=None, event_grid_size=None, event_off=False):
        """Feed predictions back; observed Event context stays fixed at initial cutoff."""
        self._validate_history(history)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        times = self._default_times(history) if context_times is None else context_times
        times = self._times(times, history.shape[1], history.device, "context_times")
        if event_context is not None and not event_off:
            if event_times is None:
                if event_context.shape[1] != len(times):
                    raise ValueError("event_times is required when Event and RGB context lengths differ")
                event_times = times.clone()
            event_times = self._times(event_times, event_context.shape[1], history.device, "event_times")
            if (event_times > times[-1]).any():
                raise ValueError("rollout cannot receive events after the initial observation cutoff")
        context, times, predictions = history[:, -self.history_length:], times[-self.history_length:], []
        for _ in range(steps):
            target = times[-1:] + self.frame_interval_s
            prediction = self(context, context_times=times, query_times=target,
                              event_context=event_context, event_times=event_times,
                              event_grid_size=event_grid_size, event_off=event_off)
            predictions.append(prediction)
            context = torch.cat((context, prediction[:, None]), dim=1)[:, -self.history_length:]
            times = torch.cat((times, target))[-self.history_length:]
        return torch.stack(predictions, dim=1)
