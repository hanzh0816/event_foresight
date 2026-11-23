"""Regression coverage for the independent-site CUDA attention batch limit."""

from contextlib import nullcontext
from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch import nn

from foresight.research.common import EventMotionEncoder, HistoryFeatures, TemporalMixer


def legacy_forward(module, tokens):
    """The original, unchunked temporal equation, independent of the new helper."""
    x = tokens + module.position
    q = module.norm(x)
    x = x + module.attention(q, q, q, need_weights=False)[0]
    return x + module.ffn(x)


class LegacyTemporalMixer(nn.Module):
    """Pre-fix module layout: catches accidental checkpoint schema changes."""

    def __init__(self, width, length):
        super().__init__()
        self.length = length
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        heads = next(h for h in (4, 2, 1) if width % h == 0)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                 nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, tokens):
        return legacy_forward(self, tokens)


def assert_parameter_gradients_close(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in actual_parameters.items():
        expected_grad = expected_parameters[name].grad
        assert parameter.grad is not None, name
        assert expected_grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_grad, atol=1e-10, rtol=1e-10,
                                   msg=lambda message, name=name: f"{name}: {message}")


@pytest.mark.parametrize("training", [True, False])
def test_chunked_temporal_equation_preserves_outputs_and_all_gradients(training):
    torch.manual_seed(401)
    reference = LegacyTemporalMixer(8, 3).double().train(training)
    model = TemporalMixer(8, 3).double().train(training)
    model.load_state_dict(reference.state_dict(), strict=True)
    model.MAX_BATCH_SITES = 5
    tokens = torch.randn(13, 3, 8, dtype=torch.float64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    probe = torch.randn_like(tokens)

    actual = model(tokens)
    expected = reference(reference_tokens)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad, atol=1e-10, rtol=1e-10)
    assert_parameter_gradients_close(model, reference)


@pytest.mark.parametrize("sites", [16383, 16384, 16385, 65536, 65537])
def test_actual_batch_boundaries_preserve_site_order_and_tail(sites):
    torch.manual_seed(402)
    model = TemporalMixer(4, 2).eval()
    assert model.MAX_BATCH_SITES == 16384
    reference = LegacyTemporalMixer(4, 2).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randn(sites, 2, 4)
    # Distinct per-site values make dropped, duplicated, or reordered chunks observable.
    tokens = tokens + torch.arange(sites).reshape(-1, 1, 1) / sites
    attention_batches = []
    handle = model.attention.register_forward_pre_hook(
        lambda module, inputs: attention_batches.append(inputs[0].shape[0]))
    try:
        with torch.no_grad():
            actual = model(tokens)
            expected = reference(tokens)
    finally:
        handle.remove()
    assert sum(attention_batches) == sites
    assert max(attention_batches) <= model.MAX_BATCH_SITES
    assert len(attention_batches) == (sites + model.MAX_BATCH_SITES - 1) // model.MAX_BATCH_SITES
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_cpu_bfloat16_autocast_preserves_chunked_outputs():
    torch.manual_seed(403)
    model = TemporalMixer(8, 3).train()
    model.MAX_BATCH_SITES = 5
    reference = LegacyTemporalMixer(8, 3).train()
    reference.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.randn(13, 3, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(tokens)
        expected = reference(tokens)
        loss = actual.square().mean()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    loss.backward()
    assert torch.isfinite(tokens.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("caller", ["event_ordered", "event_pooled", "history"])
def test_temporal_callers_preserve_outputs_and_gradients(caller):
    torch.manual_seed(404)
    if caller.startswith("event_"):
        model = EventMotionEncoder((2, 3), 8, 8, 3, caller.removeprefix("event_")).double()
        inputs = [torch.randn(2, 2, 3, 2, 16, 24, dtype=torch.float64, requires_grad=True)]
    else:
        model = HistoryFeatures(12, 8, 8, 3).double()
        inputs = [torch.randn(2, 3, 6, 12, dtype=torch.float64, requires_grad=True),
                  torch.randn(2, 2, 6, 8, dtype=torch.float64, requires_grad=True)]
    reference = deepcopy(model)
    reference.temporal.forward = MethodType(legacy_forward, reference.temporal)
    model.temporal.MAX_BATCH_SITES = 5
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    attention_batches = []
    handle = model.temporal.attention.register_forward_pre_hook(
        lambda module, values: attention_batches.append(values[0].shape[0]))
    try:
        actual = model(*inputs)
        expected = reference(*reference_inputs)
    finally:
        handle.remove()
    assert len(attention_batches) > 1
    assert max(attention_batches) <= 5
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for value, reference_value in zip(inputs, reference_inputs):
        torch.testing.assert_close(value.grad, reference_value.grad, atol=1e-10, rtol=1e-10)
    assert_parameter_gradients_close(model, reference)


def test_pre_fix_state_dict_restores_strictly_without_new_entries():
    torch.manual_seed(405)
    old_model = LegacyTemporalMixer(8, 3).eval()
    model = TemporalMixer(8, 3).eval()
    old_state = old_model.state_dict()
    result = model.load_state_dict(old_state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert model.state_dict().keys() == old_state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, old_state[name], atol=0, rtol=0)
    with torch.no_grad():
        tokens = torch.randn(3, 3, 8)
        torch.testing.assert_close(model(tokens), old_model(tokens), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: real attention kernel regression")
@pytest.mark.parametrize("mixed_bfloat16", [False, True], ids=["float32", "bf16-mixed"])
def test_cuda_real_65536_site_shape_forward_backward_and_eval(mixed_bfloat16):
    if mixed_bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support bfloat16")
    torch.manual_seed(406)
    model = TemporalMixer(128, 5).cuda().train()
    tokens = torch.randn(65536, 5, 128, device="cuda", requires_grad=True)
    batches = []
    handle = model.attention.register_forward_pre_hook(
        lambda module, values: batches.append(values[0].shape[0]))

    def precision_context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if mixed_bfloat16 else nullcontext()

    try:
        with precision_context():
            output = model(tokens)
            loss = output.square().mean()
        assert torch.isfinite(output).all()
        loss.backward()
        assert torch.isfinite(tokens.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        del output, loss
        model.zero_grad(set_to_none=True)
        tokens.grad = None
        model.eval()
        with torch.no_grad(), precision_context():
            output = model(tokens)
        assert output.shape == tokens.shape
        assert torch.isfinite(output).all()
        torch.cuda.synchronize()
    finally:
        handle.remove()
    assert batches and max(batches) <= model.MAX_BATCH_SITES
    assert sum(batches) == 2 * tokens.shape[0]
