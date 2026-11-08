"""Event boundaries, sensor-only inference, and the RGB alignment contract."""

from copy import deepcopy
from types import SimpleNamespace
import pickle

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

from foresight.data.events import EventDataset, EventSequenceReader, encoding_spec, events_to_voxel
from foresight.data.event_reader import EventStream
from foresight.data.definitions import WindowUnavailableError
from foresight.models.event_encoder import EventEncoder
from foresight.models.forecaster import PCAProjection
from foresight.vendor.dinov2.models.vision_transformer import DinoVisionTransformer
from foresight.event import EVENT_FORMAT, load_event_encoder, EventForecastDetector
from foresight.event_training import EventTask
from foresight.infer_event import make_history
from test_data import rgb_fixture


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def events(times):
    return dict(t=np.array(times, dtype=np.int64), x=np.zeros(len(times), dtype=np.int64),
                y=np.zeros(len(times), dtype=np.int64), p=np.zeros(len(times), dtype=np.int64))


def test_uniform_counts_boundaries_polarity_and_silence():
    source = events([-1, 0, 9999, 10000, 49999, 50000])
    source['p'][4] = 1
    result = events_to_voxel(source, 0, 50000, (2, 3), sensor_size=(2, 3))
    expected = torch.zeros(10, 2, 3)
    expected[0, 0, 0] = np.log(3) / np.log(6)
    expected[1, 0, 0] = np.log(2) / np.log(6)
    expected[9, 0, 0] = np.log(2) / np.log(6)
    torch.testing.assert_close(result, expected)
    assert not events_to_voxel(events([]), 0, 50000, (2, 3), sensor_size=(2, 3)).any()


@pytest.fixture
def event_fixture(rgb_fixture):
    root, path, times, tracks = rgb_fixture
    directory = path / 'events/left'
    directory.mkdir(parents=True)
    offset = int(times[0]) - 200000
    local = np.arange(0, 2500001, 10000, dtype=np.int64)
    with h5py.File(directory/'events.h5', 'w') as handle:
        for name, values in dict(t=local, x=np.zeros(len(local), np.uint16),
                                 y=np.zeros(len(local), np.uint16), p=np.ones(len(local), np.uint8)).items():
            handle[f'events/{name}'] = values
        handle['t_offset'] = offset
        handle['ms_to_idx'] = np.searchsorted(local, np.arange(2502)*1000)
    np.savetxt(path/'images/left/exposure_timestamps.txt',
               np.column_stack((times-1000, times+5000)), fmt='%d', delimiter=',')
    return root, path, times, offset


def test_hdf5_exact_absolute_half_open_slices(event_fixture):
    root, path, times, offset = event_fixture
    stream = EventStream(path/'events/left/events.h5')
    assert stream.read_events(offset, offset+20000)['t'].tolist() == [offset, offset+10000]
    assert stream.read_events(offset+1, offset+10000)['t'].size == 0
    with pytest.raises(WindowUnavailableError):
        stream.read_events(offset-1, offset+10000)
    stream.close()


def test_forecast_never_reads_rgb_and_alignment_reads_only_last(event_fixture, monkeypatch):
    root, path, times, offset = event_fixture
    reads = []
    original = EventSequenceReader.read_image
    def read(self, index):
        reads.append(index)
        return original(self, index)
    monkeypatch.setattr(EventSequenceReader, 'read_image', read)
    with EventDataset(root, image_size=(28, 42)) as dataset:
        sample = dataset[0]
        assert not reads and 'history_rgb' not in sample
        assert sample['history_events'].shape == (4, 10, 28, 42)
        assert sample['meta']['input_cutoff_us'] == times[39]+5000
        assert sample['meta']['effective_prediction_us'] == 45000
        assert sample['meta']['event_end_us'][-1] == sample['meta']['input_cutoff_us']
        restored = pickle.loads(pickle.dumps(dataset))
        assert not restored._readers and not restored._event_handles._handles
        restored.close()
    with EventDataset(root, image_size=(28, 42), alignment=True) as dataset:
        sample = dataset[0]
        assert reads == [39] and sample['rgb'].shape == (3, 28, 42)
        assert sample['meta']['event_start_us'] == [int(times[39])+5000-50000]


def test_raw_event_inference_rejects_future_input():
    spec = encoding_spec({})
    source = events([10000, 50000, 199999])
    assert make_history(source, [50000, 100000, 150000, 200000], spec, (14, 14)).shape == (4, 10, 14, 14)
    with pytest.raises(ValueError, match='cutoff'):
        make_history(events([200000]), [50000, 100000, 150000, 200000], spec)
    with pytest.raises(ValueError, match='50ms'):
        make_history(source, [50000, 100000, 170000, 200000], spec)


class SmallTeacher(nn.Module):
    image_size, grid_size, d_layers, feature_dim, pca_dim = (28, 28), (2, 2), [0, 1, 2, 3], 48, 8

    def __init__(self):
        super().__init__()
        self.encoder = DinoVisionTransformer(img_size=28, patch_size=14, embed_dim=12,
                                             depth=4, num_heads=3, num_register_tokens=4, block_chunks=0)
        self.pca = PCAProjection(dict(mean=torch.zeros(48), std=torch.ones(48),
                                     pca_mean=torch.zeros(48), pca_components=torch.randn(8, 48)*.1), 48, 8)
        self.predictor = nn.Linear(8, 8)

    def encode_latents(self, rgb):
        return self.pca(torch.cat(self.encoder.get_intermediate_layers(rgb, n=self.d_layers), -1))


def test_e2_initialization_and_gradient_boundary():
    teacher = SmallTeacher()
    encoder = EventEncoder(teacher)
    expected = teacher.encoder.patch_embed.proj.weight.mean(1, keepdim=True).repeat(1, 10, 1, 1)*.3
    assert torch.equal(encoder.backbone.patch_embed.proj.weight, expected)
    assert not encoder.backbone.training and not encoder.backbone.mask_token.requires_grad
    assert encoder.backbone.blocks[0].attn.qkv.weight.requires_grad
    encoder(torch.rand(2, 10, 28, 28)).square().mean().backward()
    assert encoder.backbone.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert not any(name in dict(encoder.named_parameters()) for name in ('mean','std','pca_components'))


def test_event_checkpoint_strict_loading_rejects_pca_before_mutation(tmp_path):
    encoder = EventEncoder(SmallTeacher())
    original = deepcopy(encoder.state_dict())
    path = tmp_path/'event.pt'
    state = deepcopy(original)
    state['backbone.cls_token'] += 1
    saved = dict(format=EVENT_FORMAT, image_size=[448, 896], event_encoding=encoding_spec({}), state_dict=state)
    torch.save(saved, path)
    load_event_encoder(encoder, path)
    assert torch.equal(encoder.backbone.cls_token, state['backbone.cls_token'])
    state['mean'] += 1
    before = deepcopy(encoder.state_dict())
    torch.save(saved, path)
    with pytest.raises(ValueError, match='coordinates'):
        load_event_encoder(encoder, path)
    assert all(torch.equal(before[k], v) for k,v in encoder.state_dict().items())


def task_config(tmp_path):
    return dict(data=dict(root=str(tmp_path), image_size=[28,28], context_frames=4, sample_history_frames=40,
                          batch_size=2, num_workers=0, tolerance_us=1000),
                model=dict(dinof_checkpoint='unused', predictor_checkpoint=None),
                optimization=dict(lr=1e-3, weight_decay=.01, scheduler='constant', accumulation_steps=1),
                runtime=dict(seed=42), output_dir=str(tmp_path))


class SmallEventTask(EventTask):
    def train_dataloader(self):
        g=torch.Generator().manual_seed(17)
        samples=[dict(rgb=torch.rand(3,28,28,generator=g), event_voxel=torch.rand(10,28,28,generator=g)) for _ in range(8)]
        return DataLoader(samples,batch_size=2)


def test_event_training_saves_resumes_and_only_updates_encoder(tmp_path):
    cfg=task_config(tmp_path)
    task=SmallEventTask(cfg, forecaster=SmallTeacher())
    before=deepcopy(task.forecaster.state_dict())
    pca=deepcopy(task.event_encoder.mean)
    student=deepcopy(task.event_encoder.backbone.patch_embed.proj.weight)
    def trainer(steps):
        return Trainer(accelerator='cpu', devices=1, max_epochs=3, max_steps=steps,
                       logger=False, enable_checkpointing=False, enable_progress_bar=False,
                       enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0)
    first=trainer(2)
    first.fit(task)
    path=tmp_path/'last.ckpt'
    first.save_checkpoint(path)
    assert all(torch.equal(before[k],v) for k,v in task.forecaster.state_dict().items())
    assert torch.equal(pca,task.event_encoder.mean)
    assert not torch.equal(student,task.event_encoder.backbone.patch_embed.proj.weight)
    restored=SmallEventTask(cfg, forecaster=SmallTeacher())
    second=trainer(3)
    second.fit(restored,ckpt_path=path)
    assert second.global_step==3
    assert all(int(s['step'])==3 for s in second.optimizers[0].state.values())
    assert all(torch.equal(before[k],v) for k,v in restored.forecaster.state_dict().items())
