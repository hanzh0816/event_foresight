"""End-to-end evaluation on a synthetic DSEC recording, with tiny model doubles."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from test_research_data import research_recording
from foresight.research import evaluate as ev
from foresight.research.compare import compare
from foresight.data.event_reader import EventStream


class PCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('basis', torch.eye(8))
    def inverse(self, x):
        return x @ self.basis


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(8))
    def forward(self, z):
        return z[:, -1] + self.offset


class TinyForecast(nn.Module):
    grid_size = (2, 3)
    pca_dim = 8
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 1)
        self.pca = PCA()
        self.predictor = TinyPredictor()
    def encode_latents(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(self.encoder(x), self.grid_size).flatten(2).transpose(1, 2)


class TinyTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecaster = TinyForecast()
    def predict_batch(self, batch):
        assert 'future_rgb_target' not in batch and 'future_events_target' not in batch
        return {'scene': ev.baseline_rollout(self.forecaster, batch['history_rgb'], 1, 'rgb')}


class TinyDetector(nn.Module):
    def forward(self, z):
        logits = z.new_full((len(z), 1, 8), -20.)
        logits[:, :, 2] = 20.
        boxes = z.new_tensor([25/640, 40/480, 30/640, 40/480]).reshape(1,1,4).expand(len(z),1,4)
        return {'pred_logits':logits, 'pred_boxes':boxes}


def evaluation_config(root, tmp_path):
    asset=tmp_path/'asset.pt'
    asset.write_bytes(b'synthetic-asset')
    return dict(stage='fusion',data=dict(root=str(root),image_size=[28,42],event_image_size=[6,8],
        context_frames=4,bins=5,rollout_steps=1,alignment='prealigned_event_view',
        alignment_evidence='synthetic common coordinates',batch_size=2,num_workers=0),
        model=dict(mode='rgb',detector_checkpoint=str(asset),dinof_checkpoint=str(asset)),
        runtime=dict(accelerator='cpu',precision='32-true',seed=42,cpu_threads=1))


def test_full_evaluation_and_paired_comparison(research_recording, tmp_path, monkeypatch):
    root=research_recording[0]
    cfg=evaluation_config(root,tmp_path)
    task=TinyTask()
    monkeypatch.setattr(ev,'load_task',lambda *a,**kw:task)
    monkeypatch.setattr(ev,'load_detector',lambda *a,**kw:TinyDetector())
    first=ev.evaluate(cfg,split='train',output=tmp_path/'first.json',max_samples=2)
    assert first['horizons']['1']['metrics']['AP'] > .99
    assert first['horizons']['1']['metadata']['cutoff_to_target_ms']['mean'] == 45.
    assert len(first['sample_ids'])==2
    second=ev.evaluate(cfg,split='train',output=tmp_path/'second.json',reference='rgb',feature_metrics=True,max_samples=2)
    assert second['trainable_parameters'] == 0
    assert second['loaded_task_trainable_parameters'] > 0
    assert second['reference_provenance']['kind'] == 'original_dinof_rgb_baseline'
    assert second['parameter_counts']['predictor']['total'] == 8
    assert second['horizons']['1']['features']['fg_l1'] is not None
    comparison=compare([first,second])
    assert comparison['rows'][1]['horizons']['1']['deltas']['AP_delta_points']==0.
    assert (tmp_path/'second.predictions.json').exists()
    assert json.loads((tmp_path/'second.json').read_text())['forecaster_state_sha256']
    with pytest.raises(FileExistsError):
        ev.evaluate(cfg,split='train',output=tmp_path/'first.json',max_samples=2)
    unpaired=deepcopy(second)
    unpaired['sample_digest']='different'
    with pytest.raises(ValueError,match='sample_digest'):
        compare([first,unpaired])
    unpaired=deepcopy(second)
    unpaired['samples'][0]['input_cutoff_us'] += 1000
    with pytest.raises(ValueError,match='samples'):
        compare([first,unpaired])
    unpaired=deepcopy(second)
    unpaired['config']['runtime']['precision']='bf16-mixed'
    with pytest.raises(ValueError,match='precision'):
        compare([first,unpaired])
    privileged=deepcopy(second)
    privileged['evaluation_kind']='observed_transition_diagnostic'
    with pytest.raises(ValueError):
        compare([first,privileged])
    diagnostic_base=deepcopy(first)
    diagnostic_base['evaluation_kind']='observed_transition_diagnostic'
    assert compare([diagnostic_base,privileged],diagnostic=True)['evaluation_kind']=='observed_transition_diagnostic'
    with pytest.raises(ValueError):
        compare([diagnostic_base,second],diagnostic=True)


def test_foreground_intersection_and_frozen_state_digest():
    target={'boxes':torch.tensor([[.51,.51,.04,.04]])}
    mask=ev.foreground_mask([target],(2,2),'cpu')
    assert mask.sum()==4
    model=nn.Linear(3,2)
    before=ev.module_digest(model)
    with torch.no_grad():
        model.bias.add_(1)
    assert before != ev.module_digest(model)


def test_conditioning_rgb_targets_never_read_future_events(research_recording, tmp_path, monkeypatch):
    root, _, times, _ = research_recording
    cfg = evaluation_config(root, tmp_path)
    cfg['stage'] = 'conditioning'
    cfg['model']['train_predictor'] = False
    task = TinyTask()
    monkeypatch.setattr(ev, 'load_task', lambda *a, **kw: task)
    monkeypatch.setattr(ev, 'load_detector', lambda *a, **kw: TinyDetector())
    calls = []
    original = EventStream.read_events
    def read(self, start, end):
        calls.append((start, end))
        return original(self, start, end)
    monkeypatch.setattr(EventStream, 'read_events', read)
    ev.evaluate(cfg, split='train', output=tmp_path/'features.json', reference='rgb',
                feature_metrics=True, max_samples=1)
    assert calls == list(zip(times[:3] + 5000, times[1:4] + 5000))
    ev.evaluate(cfg, split='train', output=tmp_path/'frozen.json', max_samples=1)
    cfg['model']['train_predictor'] = True
    with pytest.raises(ValueError, match='checkpoint is required'):
        ev.evaluate(cfg, split='train', output=tmp_path/'untrained.json')
    cfg['model'].update(mode='motion', train_predictor=False)
    with pytest.raises(ValueError, match='checkpoint is required'):
        ev.evaluate(cfg, split='train', output=tmp_path/'untrained-motion.json')


def test_rgb_reference_records_checkpoint_source(research_recording, tmp_path, monkeypatch):
    cfg = evaluation_config(research_recording[0], tmp_path)
    task = TinyTask()
    monkeypatch.setattr(ev, 'load_task', lambda *a, **kw: task)
    monkeypatch.setattr(ev, 'load_detector', lambda *a, **kw: TinyDetector())
    cfg['model']['predictor_checkpoint'] = cfg['model']['dinof_checkpoint']
    baseline = ev.evaluate(cfg, split='train', output=tmp_path/'predictor.json',
                           reference='rgb', max_samples=1)
    assert baseline['reference_provenance']['kind'] == 'configured_rgb_predictor_baseline'
    adapted = ev.evaluate(cfg, checkpoint=cfg['model']['dinof_checkpoint'], split='train',
                          output=tmp_path/'adapted.json', reference='rgb', max_samples=1)
    assert adapted['reference_provenance']['kind'] == 'loaded_predictor_rgb_ablation'
    cfg['model']['init_checkpoint'] = cfg['model']['dinof_checkpoint']
    initialized = ev.evaluate(cfg, split='train', output=tmp_path/'initialized.json',
                              reference='rgb', max_samples=1)
    assert initialized['reference_provenance']['kind'] == 'initialized_predictor_rgb_ablation'


def test_comparison_allows_predictor_tuning_but_rejects_feature_changes(research_recording, tmp_path, monkeypatch):
    cfg = evaluation_config(research_recording[0], tmp_path)
    task = TinyTask()
    monkeypatch.setattr(ev, 'load_task', lambda *a, **kw: task)
    monkeypatch.setattr(ev, 'load_detector', lambda *a, **kw: TinyDetector())
    def report(name):
        return ev.evaluate(cfg, split='train', output=tmp_path/f'{name}.json',
                           reference='rgb', max_samples=1)
    baseline = report('base')
    with torch.no_grad():
        task.forecaster.predictor.offset.add_(.1)
    adapted = report('tuned')
    assert baseline['forecaster_state_sha256'] != adapted['forecaster_state_sha256']
    assert baseline['predictor_state_sha256'] != adapted['predictor_state_sha256']
    assert baseline['feature_space_state_sha256'] == adapted['feature_space_state_sha256']
    assert compare([baseline, adapted])['compatibility_state_key'] == 'feature_space_state_sha256'
    with torch.no_grad():
        task.forecaster.encoder.bias.add_(.1)
    encoder_change = report('encoder')
    with pytest.raises(ValueError, match='feature_space_state_sha256'):
        compare([baseline, encoder_change])
    with torch.no_grad():
        task.forecaster.pca.basis.mul_(2)
    pca_change = report('pca')
    with pytest.raises(ValueError, match='feature_space_state_sha256'):
        compare([encoder_change, pca_change])

    legacy = deepcopy(baseline)
    for key in ('feature_space_state_sha256', 'predictor_state_sha256'):
        legacy.pop(key)
    assert compare([legacy, deepcopy(legacy)])['compatibility_state_key'] == 'forecaster_state_sha256'
    legacy_tuned = deepcopy(legacy)
    legacy_tuned['forecaster_state_sha256'] = adapted['forecaster_state_sha256']
    with pytest.raises(ValueError, match='forecaster_state_sha256'):
        compare([legacy, legacy_tuned])
    with pytest.raises(ValueError, match='mixed legacy'):
        compare([baseline, legacy])
    incomplete = deepcopy(baseline)
    incomplete.pop('predictor_state_sha256')
    with pytest.raises(ValueError, match='Incomplete'):
        compare([baseline, incomplete])
    missing = deepcopy(legacy)
    missing.pop('forecaster_state_sha256')
    with pytest.raises(ValueError, match='provenance'):
        compare([missing, deepcopy(missing)])
