"""RVT spatial grouping, explicit recurrent state, and training contracts."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from foresight.models.rvt import PartitionAttention, PixelLSTM, RVTEventEncoder, partition_tokens, restore_tokens


def tiny_encoder(**kwargs):
    options = dict(
        in_channels=4, feature_dim=24, input_size=(32, 64), channels=(8, 16, 24, 32),
        fpn_channels=16, partition_size=(2, 2), attention_batch_chunk_size=3,
    )
    options.update(kwargs)
    return RVTEventEncoder(**options)


@pytest.mark.parametrize("grid", [False, True])
def test_partitions_are_invertible_and_do_not_mix_samples(grid):
    x = torch.arange(2 * 4 * 8).reshape(2, 4, 8, 1).float()
    grouped = partition_tokens(x, (2, 2), grid)
    assert torch.equal(restore_tokens(grouped, (2, 2), (4, 8), grid), x)
    expected = torch.tensor([0, 4, 16, 20] if grid else [0, 1, 8, 9])
    assert torch.equal(grouped[0, :, 0], expected)
    assert grouped[:8].max() < 32
    assert grouped[8:].min() >= 32


def test_chunking_preserves_outputs_and_parameter_gradients():
    torch.manual_seed(3)
    unchunked = PartitionAttention(8, 2, (2, 2), grid=True, layer_scale=0.2, attention_batch_chunk_size=100)
    chunked = deepcopy(unchunked)
    chunked.attention_batch_chunk_size = 3
    x = torch.randn(3, 4, 8, 8)
    a, b = unchunked(x), chunked(x)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, lhs), (_, rhs) in zip(unchunked.named_parameters(), chunked.named_parameters()):
        torch.testing.assert_close(lhs.grad, rhs.grad, atol=1e-7, rtol=1e-4, msg=name)


def test_attention_bounds_independent_window_batch(monkeypatch):
    sizes = []
    real_sdpa = F.scaled_dot_product_attention
    def wrapped(q, k, v, **kwargs):
        sizes.append(q.shape[0])
        return real_sdpa(q, k, v, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", wrapped)
    block = PartitionAttention(4, 1, (1, 1), grid=False, attention_batch_chunk_size=16384)
    with torch.no_grad():
        output = block(torch.randn(3, 128, 256, 4))
    assert output.shape == (3, 128, 256, 4)
    assert len(sizes) > 1 and max(sizes) <= 16384 and sum(sizes) == 3 * 128 * 256


def test_pixel_lstm_does_not_move_memory_between_locations():
    cell = PixelLSTM(2)
    with torch.no_grad():
        cell.gates.weight.fill_(0.3)
        cell.gates.bias.zero_()
    x = torch.zeros(1, 2, 4, 4)
    hidden = torch.zeros_like(x)
    hidden[:, :, 1, 2] = 1
    output, _ = cell(x, (hidden, torch.zeros_like(x)))
    assert output[:, :, 1, 2].abs().sum() > 0
    output[:, :, 1, 2] = 0
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("stride, tokens", [(16, 8), (32, 2)])
def test_shapes_gradients_and_explicit_state(stride, tokens):
    torch.manual_seed(5)
    model = tiny_encoder(output_stride=stride)
    events = torch.randn(2, 3, 4, 32, 64, requires_grad=True)
    features, state = model(events, return_state=True)
    assert features.shape == (2, 3, tokens, 24)
    assert len(state) == 4
    for stage, (hidden, cell) in enumerate(state):
        factor = 4 * 2 ** stage
        assert hidden.shape == cell.shape == (2, model.channels[stage], 32 // factor, 64 // factor)
    features.square().mean().backward()
    assert events.grad.isfinite().all() and events.grad.abs().sum() > 0
    for stage in model.stages:
        assert stage.lstm.gates.weight.grad.abs().sum() > 0
        assert stage.blocks[0][0].qkv.weight.grad.abs().sum() > 0
        assert stage.blocks[0][1].qkv.weight.grad.abs().sum() > 0


def test_streaming_equals_clip_and_calls_reset_state():
    torch.manual_seed(7)
    model = tiny_encoder().eval()
    events = torch.randn(3, 3, 4, 32, 64)
    with torch.no_grad():
        together = model(events)
        first, state = model(events[:, :1], return_state=True)
        remaining = model(events[:, 1:], state)
        again = model(events)
        independent = torch.cat([model(sample[None]) for sample in events], dim=0)
    torch.testing.assert_close(torch.cat((first, remaining), dim=1), together)
    torch.testing.assert_close(again, together, atol=0, rtol=0)
    torch.testing.assert_close(independent, together, atol=2e-5, rtol=1e-4)
    assert not torch.allclose(model(events[:, 1:]), remaining)


def test_future_events_cannot_change_earlier_features():
    torch.manual_seed(9)
    model = tiny_encoder().eval()
    events = torch.randn(2, 3, 4, 32, 64)
    altered = events.clone()
    altered[:, 2] += 10
    with torch.no_grad():
        a, b = model(events), model(altered)
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=0, rtol=0)
    assert not torch.allclose(a[:, 2], b[:, 2])


def test_bfloat16_autocast_backward_and_state_memory():
    model = tiny_encoder()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, state = model(torch.randn(2, 2, 4, 32, 64), return_state=True)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(cell.dtype == torch.float32 for _, cell in state)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_wrong_state_batch_is_rejected():
    model = tiny_encoder()
    _, state = model(torch.randn(2, 1, 4, 32, 64), return_state=True)
    with pytest.raises(ValueError, match="batch and stage grid"):
        model(torch.randn(3, 1, 4, 32, 64), state)


def test_default_architecture_has_document_grid_and_rvt_s_settings():
    model = RVTEventEncoder()
    assert model.output_grid == (32, 64)
    assert model.architecture["heads"] == (2, 4, 8, 16)
    assert model.architecture["channels"] == (48, 96, 192, 384)
    assert model.architecture["num_blocks"] == (1, 1, 1, 1)
    assert model.s2_down.stride == (2, 2)
    restored = RVTEventEncoder(**model.architecture)
    restored.load_state_dict(model.state_dict(), strict=True)
