"""Server-side diagnostic contracts; no DSEC assets or optimizer steps required.

These tests exercise tiny real models and autograd. They must be run in the
server's model environment, not the documentation-only development session.
"""

from contextlib import nullcontext
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from foresight.research import diagnose_conditioning as diagnostic
from foresight.research.diagnostic_protocol import (
    ALLOWED_VARIANTS,
    deterministic_bin_permutation,
    make_event_donor_map,
)


@pytest.fixture
def tiny_task():
    pytest.importorskip("einops")
    from foresight.models.forecaster import DINOFeaturePredictor
    from foresight.research.conditioning import ConditioningWorldModel

    class TinyTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.forecaster = nn.Module()
            self.forecaster.predictor = DINOFeaturePredictor(
                grid_size=(2, 3), history_length=4, pca_dim=6,
                hidden_dim=12, depth=12, heads=3, dropout=0,
            )
            # The predictor's own output must transmit the loss derivative.
            nn.init.normal_(self.forecaster.predictor.maskvit.fc_out.weight, std=0.1)
            self.forecaster.encoder = nn.Linear(2, 2)
            self.forecaster.register_buffer("fixed_statistics", torch.tensor([2.0, 3.0]))
            self.model = ConditioningWorldModel(
                (2, 3), 6, 12, motion_dim=4, hidden_dim=8, bins=5,
                history_frames=4, injection_layer=6, attention_heads=2,
            )
            self.config = {"loss": {"scene_weight": 1.0}}

        @staticmethod
        def _distance(prediction, target):
            return F.smooth_l1_loss(prediction, target)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        task = TinyTask().eval()
    try:
        yield task
    finally:
        torch.set_num_threads(threads)


def _raw_events(batch_size=3):
    # Every sample, interval, bin, polarity and spatial position is distinguishable.
    return torch.arange(batch_size * 3 * 5 * 2 * 8 * 12, dtype=torch.float32).reshape(
        batch_size, 3, 5, 2, 8, 12,
    ) / 1000


@pytest.mark.parametrize("variant", ["bin_reverse", "bin_shuffle"])
def test_raw_bin_intervention_preserves_sample_interval_polarity_and_pixels(variant):
    events = _raw_events()
    before = events.clone()
    sample_ids = ["sequence:100", "sequence:101", "other:50"]
    actual = diagnostic.intervene_events(events, variant, sample_ids, seed=42)
    assert actual.shape == events.shape
    for sample, sid in enumerate(sample_ids):
        order = list(reversed(range(5))) if variant == "bin_reverse" else deterministic_bin_permutation(5, 42, sid)
        assert order != list(range(5))
        for interval in range(3):
            for slot, source_bin in enumerate(order):
                assert torch.equal(actual[sample, interval, slot], before[sample, interval, source_bin])
    assert torch.equal(events, before)


def test_bin_shuffle_is_fixed_per_sample_across_batch_partition_and_order():
    events = _raw_events()
    ids = ["first:10", "second:20", "third:30"]
    together = diagnostic.intervene_events(events, "bin_shuffle", ids, seed=7)
    separate = torch.cat([
        diagnostic.intervene_events(events[index:index + 1], "bin_shuffle", [sid], seed=7)
        for index, sid in enumerate(ids)
    ])
    reordered = diagnostic.intervene_events(events[[2, 0, 1]], "bin_shuffle", [ids[i] for i in (2, 0, 1)], seed=7)
    assert torch.equal(together, separate)
    assert torch.equal(reordered, together[[2, 0, 1]])


@pytest.mark.parametrize("variant", ["bin_reverse", "bin_shuffle"])
def test_raw_bin_shuffle_keeps_chronological_embedding_slots_fixed(tiny_task, variant):
    class BinSentinelEncoder(nn.Module):
        def forward(self, bins):
            return bins[:, :1, :1, :1].expand(-1, 8, 2, 3)

    model = tiny_task.model
    model.event_stem = BinSentinelEncoder()
    events = _raw_events(batch_size=1)
    altered = diagnostic.intervene_events(events, variant, ["sequence:100"], seed=42)
    memory = model.encode_events(altered)
    # The value changes slots before adding embeddings. Moving positioned
    # key/value rows would preserve the old position with its content instead.
    positions = (model.interval_position + model.bin_position).reshape(1, 1, 15, 8)
    expected_content = altered[0, :, :, 0, 0, 0].reshape(1, 1, 15, 1).expand(1, 6, 15, 8)
    torch.testing.assert_close(memory, expected_content + positions, rtol=0, atol=0)


class RecordingDataset:
    """A donor RGB/target read is observable independently of its Event read."""

    def __init__(self):
        self.sample_ids = ["a:10", "a:11", "b:20", "c:30"]
        self.full_reads = []
        self.event_reads = []
        self.events = {
            sid: torch.full((3, 5, 2, 8, 12), float(index + 1))
            for index, sid in enumerate(self.sample_ids)
        }

    def __getitem__(self, index):
        sid = self.sample_ids[index]
        self.full_reads.append(sid)
        return {
            "history_rgb": f"RGB belongs to {sid}",
            "targets": f"target belongs to {sid}",
            "history_events": self.events[sid],
            "meta": {"sample_id": sid},
        }

    def read_history_events(self, index):
        sid = self.sample_ids[index]
        self.event_reads.append(sid)
        return self.events[sid]


def test_paired_dataset_reads_donor_events_without_donor_rgb_or_targets():
    dataset = RecordingDataset()
    mapping = make_event_donor_map(dataset.sample_ids, seed=42)
    receiver = dataset.sample_ids[0]
    paired = diagnostic.PairedDiagnosticDataset(dataset, [receiver], mapping)
    sample = paired[0]
    assert dataset.full_reads == [receiver]
    assert dataset.event_reads == [mapping[receiver]]
    assert sample["history_rgb"] == f"RGB belongs to {receiver}"
    assert sample["targets"] == f"target belongs to {receiver}"
    assert sample["meta"]["sample_id"] == receiver
    assert sample["donor_sample_id"] == mapping[receiver]
    assert sample["donor_events"] is dataset.events[mapping[receiver]]
    assert mapping[receiver].rsplit(":", 1)[0] != receiver.rsplit(":", 1)[0]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_fixed_donors_remain_foreign_at_every_batch_size(batch_size):
    dataset = RecordingDataset()
    mapping = make_event_donor_map(dataset.sample_ids, seed=42)
    # Recipients intentionally contain only sequence a; donors come from the
    # complete inventory and cannot be obtained by shuffling the current batch.
    recipients = dataset.sample_ids[:2]
    paired = diagnostic.PairedDiagnosticDataset(dataset, recipients, mapping)
    actual = {}
    for start in range(0, len(paired), batch_size):
        rows = [paired[index] for index in range(start, min(start + batch_size, len(paired)))]
        ids = [row["meta"]["sample_id"] for row in rows]
        events = torch.stack([row["history_events"] for row in rows])
        donors = torch.stack([row["donor_events"] for row in rows])
        changed = diagnostic.intervene_events(events, "mismatched", ids, seed=42, donor_events=donors)
        for index, sid in enumerate(ids):
            actual[sid] = changed[index]
            assert torch.equal(changed[index], dataset.events[mapping[sid]])
            assert not torch.equal(changed[index], events[index])
    assert set(actual) == set(recipients)
    assert dataset.full_reads == recipients


def test_mismatched_rejects_missing_or_wrong_shaped_donors():
    events = _raw_events(batch_size=1)
    for donors in (None, events[:, :, :1]):
        with pytest.raises(ValueError, match="same-shaped fixed donor"):
            diagnostic.intervene_events(events, "mismatched", ["a:1"], 42, donors)


def test_metric_accumulator_weights_regions_by_patch_count_and_diagnostics_by_sample():
    accumulator = diagnostic.MetricAccumulator()
    for values, foreground, scalar in (([2., 4., 6.], [True, False, False], 9.),
                                        ([8., 10.], [True, True], 3.)):
        prediction = torch.tensor(values).unsqueeze(-1).expand(-1, 2)
        row = diagnostic.sample_metrics(prediction, torch.zeros_like(prediction),
                                        torch.zeros_like(prediction), torch.tensor(foreground))
        row["diagnostics"] = {"known_scalar": scalar, "empty_ratio": None}
        accumulator.update(row)
    result = accumulator.compute()
    assert result["num_samples"] == 2
    assert result["features"]["all_l1"] == pytest.approx(6.)
    assert result["features"]["fg_l1"] == pytest.approx(20. / 3)
    assert result["features"]["bg_l1"] == pytest.approx(5.)
    assert result["features"]["all_smooth_l1"] == pytest.approx(5.5)
    assert result["features"]["fg_smooth_l1"] == pytest.approx(18.5 / 3)
    assert result["features"]["bg_smooth_l1"] == pytest.approx(4.5)
    assert result["features"]["foreground_patch_fraction"] == pytest.approx(3. / 5)
    assert result["prediction_change_vs_correct"]["fg_l1"] == pytest.approx(20. / 3)
    assert result["diagnostic_means"] == {"known_scalar": 6.}


@pytest.mark.parametrize("all_foreground", [False, True])
def test_empty_foreground_or_background_stays_none(all_foreground):
    accumulator = diagnostic.MetricAccumulator()
    prediction = torch.ones(4, 2)
    foreground = torch.full((4,), all_foreground, dtype=torch.bool)
    row = diagnostic.sample_metrics(prediction, torch.zeros_like(prediction), prediction, foreground)
    empty = "bg" if all_foreground else "fg"
    assert row["patch_counts"][empty] == 0
    assert row["features"][empty + "_l1"] is None
    accumulator.update(row)
    result = accumulator.compute()
    assert result["features"][empty + "_l1"] is None
    assert result["features"][empty + "_smooth_l1"] is None
    assert result["prediction_change_vs_correct"][empty + "_l1"] is None
    assert result["features"]["all_l1"] == 1.


def _snapshot(task):
    return {
        "state": {name: value.detach().clone() for name, value in task.state_dict().items()},
        "flags": {name: value.requires_grad for name, value in task.named_parameters()},
        "training": {name: module.training for name, module in task.named_modules()},
        "grads": {name: None if value.grad is None else value.grad.clone()
                  for name, value in task.named_parameters()},
    }


def _assert_snapshot(task, before):
    assert task.state_dict().keys() == before["state"].keys()
    for name, value in task.state_dict().items():
        assert torch.equal(value, before["state"][name]), name
    assert {name: value.requires_grad for name, value in task.named_parameters()} == before["flags"]
    assert {name: module.training for name, module in task.named_modules()} == before["training"]
    for name, value in task.named_parameters():
        expected = before["grads"][name]
        if expected is None:
            assert value.grad is None, name
        else:
            assert torch.equal(value.grad, expected), name


def _probe_inputs():
    # The actual runner encodes RGB under inference_mode. Its resulting tensors
    # must be cloned outside that mode before autograd saves them for backward.
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        return torch.randn(2, 4, 6, 6), torch.randn(2, 3, 5, 2, 8, 12), torch.randn(2, 1, 6, 6)


@pytest.mark.parametrize("activate_output", [False, True])
def test_gradient_probe_preserves_state_and_reveals_zero_projection_gate(tiny_task, activate_output):
    task = tiny_task
    if activate_output:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            nn.init.normal_(task.model.output.weight, std=0.15)
    # Exercise restoration of mixed flags and existing gradient buffers instead
    # of only checking the common all-trainable/grad-is-None state.
    for index, parameter in enumerate(task.parameters()):
        parameter.requires_grad_(index % 2 == 0)
        if index % 3 == 0:
            parameter.grad = torch.full_like(parameter, 0.125)
    before = _snapshot(task)
    history, events, targets = _probe_inputs()
    rows = diagnostic.probe_gradients(task, history, events, targets, ["a:10", "b:20"], nullcontext)
    assert [row["sample_id"] for row in rows] == ["a:10", "b:20"]
    for row in rows:
        assert row["optimizer_steps"] == 0
        modules = row["modules"]
        assert modules["output"]["gradient_rms"] > 0
        assert modules["event_stem"]["disconnected_tensors"] == 0
        assert modules["event_stem"]["grad_numel"] == modules["event_stem"]["numel"]
        for key in ("cross_attention_q", "cross_attention_k", "cross_attention_v"):
            assert key in modules
        if activate_output:
            assert modules["event_stem"]["gradient_rms"] > 0
            assert row["event_input_gradient_rms"] > 0
            assert modules["bin_position"]["gradient_rms"] > 0
        else:
            assert modules["event_stem"]["gradient_rms"] == 0
            assert modules["event_stem"]["nonzero_gradient_fraction"] == 0
            assert row["event_input_gradient_rms"] == 0
    _assert_snapshot(task, before)


def test_gradient_probe_restores_flags_and_state_when_loss_fails(tiny_task, monkeypatch):
    for index, parameter in enumerate(tiny_task.parameters()):
        parameter.requires_grad_(index % 2 == 0)
    before = _snapshot(tiny_task)

    def fail(*args):
        raise RuntimeError("test loss failure")

    monkeypatch.setattr(tiny_task, "_distance", fail)
    history, events, targets = _probe_inputs()
    with pytest.raises(RuntimeError, match="test loss failure"):
        diagnostic.probe_gradients(tiny_task, history, events, targets, ["a:10", "b:20"], nullcontext)
    _assert_snapshot(tiny_task, before)


def test_json_artifact_writer_never_overwrites_existing_results(tmp_path):
    path = tmp_path / "report.json"
    original = {"weights_unchanged": True, "sample_ids": ["a:10", "b:20"]}
    diagnostic._write_json(path, original)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic._write_json(path, {"weights_unchanged": False})
    assert path.read_bytes() == before
    assert json.loads(path.read_text()) == original


def test_run_writes_paired_seven_variant_report_without_changing_weights(tiny_task, tmp_path, monkeypatch):
    """Exercise the real runner, conditioner and predictor with in-memory inputs."""
    for dependency in ("numpy", "yaml", "PIL", "h5py", "hdf5plugin", "lightning.pytorch"):
        pytest.importorskip(dependency)
    from foresight.research import data as data_module
    from foresight.research import task as task_module

    datasets = []

    class MemoryDataset:
        def __init__(self, **kwargs):
            assert kwargs["future_targets"] is True
            assert kwargs["future_event_targets"] is False
            self.sample_ids = ["a:10", "a:11", "b:20", "b:21"]
            self.full_inventory_size = len(self.sample_ids)
            self.excluded_windows = {"unlabeled": 1, "cadence": 0, "coverage": 2, "noncausal": 0}
            self.events = _raw_events(len(self.sample_ids))
            self.closed = False
            self.full_reads = []
            self.donor_reads = []
            datasets.append(self)

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, index):
            sid = self.sample_ids[index]
            self.full_reads.append(sid)
            cutoff = 1_000_000 + index * 50_000
            return {
                # This test substitutes pre-encoded tensors for RGB files only;
                # the complete tiny predictor and Event conditioner remain real.
                "history_rgb": torch.linspace(-1., 1., 4 * 6 * 6).reshape(4, 6, 6) + index / 10,
                "future_rgb_target": torch.full((1, 6, 6), index / 20),
                "history_events": self.events[index],
                "targets": [{"boxes": torch.tensor([[.25, .5, .2, .5]])}],
                "meta": {
                    "sample_id": sid,
                    "sequence": sid.rsplit(":", 1)[0],
                    "target_ids": [sid + ":future"],
                    "input_cutoff_us": cutoff,
                    "event_history_start_us": cutoff - 150_000,
                    "history_start_us": cutoff - 155_000,
                    "horizon_us": [47_000 + index * 1000],
                },
            }

        def read_history_events(self, index):
            self.donor_reads.append(self.sample_ids[index])
            return self.events[index]

        def close(self):
            self.closed = True

    task = tiny_task
    task.forecaster.grid_size = (2, 3)
    task.forecaster.pca = nn.Identity().eval()
    task.encode_history = lambda batch: batch["history_rgb"].clone()
    task.encode_targets = lambda batch: batch["future_rgb_target"].clone()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        nn.init.normal_(task.model.output.weight, std=0.15)
    before = _snapshot(task)
    checkpoint = tmp_path / "mock-checkpoint.pt"
    dinof = tmp_path / "mock-dinof.pt"
    # The loader is patched below; these files exist solely for provenance hashes.
    checkpoint.write_bytes(b"in-memory tiny task fixture")
    dinof.write_bytes(b"in-memory tiny predictor fixture")
    config = {
        "stage": "conditioning",
        "data": {"root": str(tmp_path / "unused-dsec"), "batch_size": 1, "num_workers": 0},
        "model": {"mode": "event_ordered", "dinof_checkpoint": str(dinof)},
        "loss": {"kind": "smooth_l1", "scene_weight": 1.0},
        "runtime": {"accelerator": "cpu", "precision": "32-true", "cpu_threads": 1},
    }
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(data_module, "ResearchDSECDataset", MemoryDataset)
    monkeypatch.setattr(task_module, "load_task", lambda config, *, checkpoint: task)
    monkeypatch.setattr(task_module, "git_revision", lambda: "fixture-revision")
    monkeypatch.setattr(diagnostic, "_git_status", lambda: "")
    output = tmp_path / "diagnostics"
    precision_before = torch.get_float32_matmul_precision()
    try:
        with torch.random.fork_rng(devices=[]):
            report = diagnostic.run(
                config, checkpoint=checkpoint, output_dir=output, split="val",
                max_samples=3, variants=list(reversed(ALLOWED_VARIANTS)), seed=42,
                gradient_samples=1, loader_batch_size=2, num_workers=0, skip_detection=True,
            )
        # An existing output directory is rejected before any new dataset read.
        files_before = {path.name: path.read_bytes() for path in output.iterdir()}
        with pytest.raises(FileExistsError, match="new diagnostic output directory"):
            diagnostic.run(config, checkpoint=checkpoint, output_dir=output, skip_detection=True)
        assert {path.name: path.read_bytes() for path in output.iterdir()} == files_before
    finally:
        torch.set_float32_matmul_precision(precision_before)

    expected_ids = ["a:10", "a:11", "b:20"]
    assert len(datasets) == 1 and datasets[0].closed
    assert datasets[0].full_reads == expected_ids
    assert report["sample_ids"] == expected_ids
    assert tuple(report["variants"]) == ALLOWED_VARIANTS
    assert report["weights_unchanged"] is True
    assert report["diagnostic_forward_max_abs_error"] == 0
    assert report["execution"]["loader_batch_size"] == 2
    assert report["execution"]["optimizer_steps"] == 0
    assert report["scope"]["full_inventory_size"] == 4
    assert report["scope"]["available_inventory_size_before_manifest"] == 4
    assert report["scope"]["evaluated_samples"] == 3
    assert report["scope"]["is_full_inventory"] is False
    assert report["scope"]["excluded_windows"] == datasets[0].excluded_windows
    assert report["protocol_timing"] == {
        "actual_event_history_ms": {"min": 150., "max": 150., "mean": 150.},
        "rgb_observation_support_ms": {"min": 155., "max": 155., "mean": 155.},
        "cutoff_to_target_ms": {"min": 47., "max": 49., "mean": 48.},
    }

    for filename in ("run.json", "report.json", "summary.md", "manifest.json", "donor_map.json", "gradients.jsonl"):
        assert (output / filename).is_file(), filename
    assert json.loads((output / "report.json").read_text()) == report
    assert json.loads((output / "manifest.json").read_text()) == {"split": "val", "sample_ids": expected_ids}
    donor_file = json.loads((output / "donor_map.json").read_text())
    expected_donors = make_event_donor_map(datasets[0].sample_ids, 42)
    assert donor_file["mapping"] == expected_donors
    assert datasets[0].donor_reads == [expected_donors[sid] for sid in expected_ids]
    assert report["artifacts"]["predictions"] == {}
    assert not list(output.glob("*.predictions.jsonl"))
    for variant in ALLOWED_VARIANTS:
        rows = [json.loads(line) for line in (output / f"{variant}.samples.jsonl").read_text().splitlines()]
        assert [row["sample_id"] for row in rows] == expected_ids
        assert all(row["variant"] == variant for row in rows)
        assert report["variants"][variant]["num_samples"] == len(expected_ids)
        assert report["variants"][variant]["detection"] is None
        for row in rows:
            assert row["metadata"]["sample_id"] == row["sample_id"]
            expected_donor = expected_donors[row["sample_id"]] if variant == "mismatched" else None
            assert row["donor_sample_id"] == expected_donor
            if variant == "correct":
                assert row["prediction_change_vs_correct"] == {"all_l1": 0., "fg_l1": 0., "bg_l1": 0.}
    gradient_rows = [json.loads(line) for line in (output / "gradients.jsonl").read_text().splitlines()]
    assert len(gradient_rows) == report["gradient_probe"]["num_samples"] == 1
    assert gradient_rows[0]["sample_id"] == expected_ids[0]
    assert gradient_rows[0]["optimizer_steps"] == 0
    assert (output / "summary.md").read_text().strip()
    _assert_snapshot(task, before)
