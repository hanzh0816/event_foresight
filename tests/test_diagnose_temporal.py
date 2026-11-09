import pytest
import torch

from foresight.diagnose_temporal import perturb_events, perturb_history, predict_future


def test_reversal_changes_predictor_order_without_changing_current_state():
    original = torch.arange(2 * 4 * 2 * 3).reshape(2, 4, 2, 3)
    saved = original.clone()
    reversed_history = perturb_history(original, "reverse_history")
    assert torch.equal(reversed_history[:, 0], original[:, -1])
    assert torch.equal(reversed_history[:, -1], original[:, 0])
    assert torch.equal(original, saved)
    assert torch.equal(perturb_history(reversed_history, "reverse_history"), original)


def test_repetition_is_latest_state_not_oldest_or_average():
    original = torch.arange(24).reshape(1, 4, 2, 3)
    repeated = perturb_history(original, "repeat_last")
    assert repeated.shape == original.shape
    for slot in range(4):
        assert torch.equal(repeated[:, slot], original[:, -1])


def test_bin_reversal_preserves_polarity_and_per_pixel_total():
    events = torch.arange(2 * 4 * 10 * 2 * 3).reshape(2, 4, 10, 2, 3)
    reversed_bins = perturb_events(events, "reverse_bins")
    assert torch.equal(reversed_bins[:, :, 0], events[:, :, 4])
    assert torch.equal(reversed_bins[:, :, 5], events[:, :, 9])
    assert torch.equal(reversed_bins[:, :, :5].sum(2), events[:, :, :5].sum(2))
    assert torch.equal(reversed_bins[:, :, 5:].sum(2), events[:, :, 5:].sum(2))
    assert torch.equal(perturb_events(reversed_bins, "reverse_bins"), events)


def test_each_control_only_changes_its_intended_stage():
    events, history = torch.zeros(1, 4, 10, 2, 3), torch.zeros(1, 4, 2, 3)
    for control in ("identity", "reverse_history", "repeat_last", "persist_latent"):
        assert perturb_events(events, control) is events
    for control in ("identity", "reverse_bins", "persist_latent"):
        assert perturb_history(history, control) is history
    with pytest.raises(ValueError):
        perturb_history(history, "unknown")
    with pytest.raises(ValueError):
        perturb_events(events[:, :, :5], "reverse_bins")


def test_persistence_bypasses_predictor_but_repeat_last_still_calls_it():
    history = torch.arange(24).reshape(1, 4, 2, 3)
    calls = []
    def predictor(value):
        calls.append(value)
        return value[:, -1] + 100
    assert torch.equal(predict_future(history, predictor, "persist_latent"), history[:, -1])
    assert not calls
    assert torch.equal(predict_future(history, predictor, "repeat_last"), history[:, -1] + 100)
    assert len(calls) == 1
    assert torch.equal(calls[0], history[:, -1:].expand_as(history))


def test_temporal_cli_supports_explicit_full_split(monkeypatch):
    from foresight import diagnose_temporal
    monkeypatch.setattr(diagnose_temporal, "run", lambda args: args)
    required = ["--encoder-checkpoint", "event.pt", "--control", "identity", "--output", "outputs/full"]
    args = diagnose_temporal.main(required + ["--full-split", "--split", "test", "--data-root", "/portable/DSEC"])
    assert args.full_split and args.manifest is None and args.data_root == "/portable/DSEC"
    with pytest.raises(SystemExit):
        diagnose_temporal.main(required)
    with pytest.raises(SystemExit):
        diagnose_temporal.main(required + ["--full-split", "--manifest", "subset.json"])
