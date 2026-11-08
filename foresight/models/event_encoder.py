"""RGB-initialized Event Encoder; no predictor or detector parameters."""

from copy import deepcopy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class EventEncoder(nn.Module):
    """Full DINOv2 fine-tuning on ten event count channels (the E2 baseline).

    Fixed PCA buffers keep the RGB predictor coordinate system. Silent pixels
    map to -2 through the original (counts - 0.5) / 0.25 normalization.
    """

    def __init__(self, forecaster, gradient_checkpointing=False):
        super().__init__()
        self.in_channels = 10
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.image_size = tuple(forecaster.image_size)
        self.grid_size = tuple(forecaster.grid_size)
        self.d_layers = list(forecaster.d_layers)
        self.feature_dim = int(forecaster.feature_dim)
        self.pca_dim = int(forecaster.pca_dim)
        self.backbone = deepcopy(forecaster.encoder)
        old = self.backbone.patch_embed.proj
        if not isinstance(old, nn.Conv2d) or old.in_channels != 3 or old.groups != 1:
            raise ValueError("Expected the original DINOv2 three-channel Conv2d patch projection")
        # Keep deterministic RGB initialization without advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            new = nn.Conv2d(
                self.in_channels, old.out_channels, old.kernel_size,
                stride=old.stride, padding=old.padding, dilation=old.dilation,
                bias=old.bias is not None, padding_mode=old.padding_mode,
            )
        new = new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            count_channels = self.in_channels
            new.weight.zero_()
            new.weight[:, :count_channels].copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, count_channels, 1, 1)
                                                * (3.0 / count_channels))
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.backbone.patch_embed.proj = new
        self.backbone.patch_embed.in_chans = self.in_channels
        self.backbone.requires_grad_(True)
        # No masks are passed to get_intermediate_layers. This parameter does
        # not participate in the graph, so excluding it also keeps DDP valid.
        if getattr(self.backbone, "mask_token", None) is not None:
            self.backbone.mask_token.requires_grad_(False)
        for name in ("mean", "std", "pca_mean", "pca_components"):
            self.register_buffer(name, getattr(forecaster.pca, name).detach().clone())
        self.register_buffer("input_mean", torch.tensor(0.5))
        self.register_buffer("input_std", torch.tensor(0.25))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Deterministic feature extraction; eval() does not disable autograd.
        self.backbone.eval()
        return self

    def forward(self, events):
        """Return the original RGB Predictor's PCA token coordinates."""
        expected = (self.in_channels, *self.image_size)
        if events.ndim != 4 or tuple(events.shape[1:]) != expected:
            raise ValueError(f"Expected event input shape [B,{','.join(map(str, expected))}], got {tuple(events.shape)}")
        if not events.is_floating_point():
            raise TypeError("Event inputs must be floating point normalized log counts")
        inputs = (events - self.input_mean) / self.input_std

        def encode(x):
            return self.backbone.get_intermediate_layers(x, n=self.d_layers, reshape=False)

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            layers = checkpoint(encode, inputs, use_reentrant=False)
        else:
            layers = encode(inputs)
        features = torch.cat(layers, dim=-1)
        shape = (len(events), self.grid_size[0] * self.grid_size[1], self.feature_dim)
        if tuple(features.shape) != shape:
            raise ValueError(f"Backbone returned {tuple(features.shape)}, expected {shape}")
        latent = (((features - self.mean) / self.std) - self.pca_mean) @ self.pca_components.T
        return latent
