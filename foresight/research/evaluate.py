"""Evaluate research models on identical DSEC windows and fixed detector weights."""
from contextlib import nullcontext
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import ResearchDSECDataset, collate_research, HorizonDatasetView
from .task import load_task
from ..evaluation import FutureDetectionEvaluator
from ..models.detector import load_detector, postprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module_digest(module):
    """Fingerprint the loaded frozen state, including checkpoint overrides."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sample_digest(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def foreground_mask(targets, grid_size, device):
    """Patch/box intersection in normalized coordinates, not center membership."""
    h, w = grid_size
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    x0, y0 = x.flatten() / w, y.flatten() / h
    x1, y1 = (x.flatten() + 1) / w, (y.flatten() + 1) / h
    masks = []
    for target in targets:
        boxes = target['boxes'].to(device)
        if len(boxes) == 0:
            masks.append(torch.zeros(h * w, dtype=torch.bool, device=device))
            continue
        lo, hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
        masks.append(((x1[:, None] > lo[:, 0]) & (x0[:, None] < hi[:, 0]) &
                      (y1[:, None] > lo[:, 1]) & (y0[:, None] < hi[:, 1])).any(-1))
    return torch.stack(masks)


def baseline_rollout(forecaster, history, steps, kind):
    batch, frames = history.shape[:2]
    z = forecaster.encode_latents(history.flatten(0, 1)).reshape(batch, frames, -1, forecaster.pca_dim)
    outputs = []
    for _ in range(steps):
        next_z = z[:, -1] if kind == 'persistence' else forecaster.predictor(z)
        outputs.append(next_z)
        z = torch.cat((z[:, 1:], next_z[:, None]), 1)
    return torch.stack(outputs, 1)


def dataset_kwargs(config, split, *, future_targets):
    data = config['data']
    allowed = ('image_size', 'event_image_size', 'context_frames', 'bins', 'rollout_steps',
               'sample_history_frames', 'tolerance_us', 'alignment', 'alignment_evidence', 'evaluation_horizons')
    kwargs = {key: data[key] for key in allowed if key in data}
    kwargs.update(root=data['root'], split=split, future_targets=future_targets,
                  stride=data.get(f'{split}_stride', data.get('stride', 1)))
    for key in ('sequences', 'sample_ids'):
        if data.get(f'{split}_{key}') is not None:
            kwargs[key] = data[f'{split}_{key}']
    return kwargs


def evaluate(config, *, checkpoint=None, split='val', output, reference=None,
             feature_metrics=False, max_samples=None, manifest=None, horizons=None):
    config = deepcopy(config)
    steps = config['data']['rollout_steps']
    horizons = list(config['data'].get('evaluation_horizons', range(1, steps + 1))) if horizons is None else list(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(h < 1 or h > steps for h in horizons):
        raise ValueError('Evaluation horizons must be unique steps within rollout_steps')
    if reference not in (None, 'rgb', 'persistence', 'future_rgb'):
        raise ValueError('Unknown reference')
    if checkpoint is None and reference is None and not ((config['stage'] == 'fusion' and config['model']['mode'] == 'rgb') or config['model'].get('copy_last_motion')):
        raise ValueError('A trained research checkpoint is required for model evaluation')
    diagnostic = reference == 'future_rgb' or (reference is None and config['stage'] == 'transition')
    require_targets = feature_metrics or diagnostic
    kwargs = dataset_kwargs(config, split, future_targets=require_targets)
    if manifest:
        saved = json.loads(Path(manifest).read_text())
        if isinstance(saved, dict):
            if saved.get('split') != split:
                raise ValueError('Manifest split mismatch')
            saved = saved['sample_ids']
        kwargs['sample_ids'] = saved
    dataset = ResearchDSECDataset(**kwargs)
    if not set(horizons) <= set(dataset.evaluation_horizons):
        dataset.close()
        raise ValueError('Requested horizons must belong to the common inventory evaluation_horizons')
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError('max_samples must be positive')
        kwargs['sample_ids'] = dataset.sample_ids[:max_samples]
        dataset.close()
        dataset = ResearchDSECDataset(**kwargs)
    if not len(dataset):
        raise ValueError('Empty evaluation dataset')
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.predictions.json').exists():
        raise FileExistsError('Use a new output path; evaluation never overwrites previous results')
    if not config['model'].get('detector_checkpoint'):
        raise ValueError('model.detector_checkpoint is required for fixed-head evaluation')
    requested = config['runtime'].get('accelerator', 'auto')
    device = torch.device('cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')
    if requested == 'gpu' and device.type != 'cuda':
        raise RuntimeError('GPU evaluation requested, but CUDA is unavailable')
    torch.set_num_threads(config['runtime'].get('cpu_threads', 4))
    torch.set_float32_matmul_precision('high')
    task = load_task(config, checkpoint=checkpoint).eval().to(device)
    detector = load_detector(config['model']['detector_checkpoint']).eval().requires_grad_(False).to(device)
    loader = DataLoader(dataset, batch_size=config['data'].get('batch_size', 1), shuffle=False,
                        num_workers=config['data'].get('num_workers', 0), collate_fn=collate_research)
    evaluators = {h: FutureDetectionEvaluator(HorizonDatasetView(dataset, h)) for h in horizons}
    records, totals = {h: [] for h in horizons}, {h: dict(all_sum=0., all_n=0, fg_sum=0., fg_n=0, bg_sum=0., bg_n=0) for h in horizons}
    samples = []
    transition_diagnostics = None
    if config['stage'] == 'transition' and reference is None and feature_metrics:
        from .diagnostics import TransitionDiagnostics
        transition_diagnostics = TransitionDiagnostics()
    started = time.monotonic()
    precision = config['runtime'].get('precision', '32-true')
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16 if precision == 'bf16-mixed' else torch.float16) if device.type == 'cuda' and precision in ('bf16-mixed', '16-mixed') else nullcontext()
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with amp():
                    if reference in ('rgb', 'persistence'):
                        pred = baseline_rollout(task.forecaster, batch['history_rgb'], steps, reference)
                    elif reference == 'future_rgb':
                        b, s = batch['future_rgb_target'].shape[:2]
                        pred = task.forecaster.encode_latents(batch['future_rgb_target'].flatten(0, 1)).reshape(b, s, -1, task.forecaster.pca_dim)
                    else:
                        # The task forward only receives observation fields except in the
                        # explicitly labelled observed-transition diagnostic stage.
                        pred = task.predict_batch(batch)['scene']
                    target_z = None
                    if feature_metrics:
                        b, s = batch['future_rgb_target'].shape[:2]
                        target_z = task.forecaster.encode_latents(batch['future_rgb_target'].flatten(0, 1)).reshape(b, s, -1, task.forecaster.pca_dim)
                        if transition_diagnostics is not None:
                            history_z = task.encode_history(batch)
                            starts = torch.cat((history_z[:, -1:], target_z[:, :-1]), dim=1)
                            transition_diagnostics.update(task.model, starts, batch['future_events_target'], target_z)
                    for h in horizons:
                        decoded = detector(task.forecaster.pca.inverse(pred[:, h - 1]))
                        detections = postprocess(decoded)
                        current = []
                        for detection, meta in zip(detections, batch['meta'], strict=True):
                            current.append(dict(sample_id=meta['sample_id'], target_id=meta['target_ids'][h - 1],
                                boxes=detection['boxes'].float().cpu().tolist(),
                                class_ids=detection['labels'].cpu().tolist(), scores=detection['scores'].float().cpu().tolist()))
                        evaluators[h].update(current)
                        records[h].extend(current)
                        if target_z is not None:
                            err = (pred[:, h-1].float() - target_z[:, h-1].float()).abs().mean(-1)
                            mask = foreground_mask([ts[h-1] for ts in batch['targets']], task.forecaster.grid_size, device)
                            total = totals[h]
                            for name, selected in (('all', torch.ones_like(mask)), ('fg', mask), ('bg', ~mask)):
                                total[f'{name}_sum'] += err[selected].sum().item()
                                total[f'{name}_n'] += selected.sum().item()
                for meta in batch['meta']:
                    samples.append({k: meta[k] for k in ('sample_id', 'input_cutoff_us', 'history_start_us', 'event_history_start_us', 'target_timestamps_us', 'horizon_us', 'transition_horizon_us') if k in meta})
        if [m['sample_id'] for m in samples] != dataset.sample_ids:
            raise RuntimeError('Evaluation sample order mismatch')
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True, cwd=Path(__file__).resolve().parents[2]).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        report = dict(format='event-wm-evaluation-v1', stage=config['stage'], mode=config['model'].get('mode'),
            reference=reference, evaluation_kind='future_rgb_readout_diagnostic' if reference == 'future_rgb' else 'observed_transition_diagnostic' if diagnostic else 'history_only_forecast',
            split=split, sample_ids=dataset.sample_ids, sample_digest=sample_digest(dataset.sample_ids),
            config=config, git_commit=commit, checkpoint=str(checkpoint) if checkpoint else None,
            checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None,
            detector_sha256=file_sha256(config['model']['detector_checkpoint']),
            dinof_sha256=file_sha256(config['model']['dinof_checkpoint']),
            forecaster_state_sha256=module_digest(task.forecaster),
            trainable_parameters=0 if reference else sum(p.numel() for p in task.parameters() if p.requires_grad),
            loaded_task_trainable_parameters=sum(p.numel() for p in task.parameters() if p.requires_grad),
            elapsed_seconds=time.monotonic()-started, samples=samples, horizons={})
        if transition_diagnostics is not None:
            report['transition_diagnostics'] = transition_diagnostics.compute()
        for h in horizons:
            value = evaluators[h].compute().to_dict()
            values = [meta['horizon_us'][h - 1] / 1000 for meta in samples]
            value['metadata']['cutoff_to_target_ms'] = dict(min=min(values), max=max(values), mean=sum(values)/len(values))
            value['features'] = {name+'_l1': totals[h][name+'_sum']/totals[h][name+'_n'] if totals[h][name+'_n'] else None for name in ('all','fg','bg')}
            value['features']['foreground_patch_fraction'] = totals[h]['fg_n']/totals[h]['all_n'] if totals[h]['all_n'] else None
            report['horizons'][str(h)] = value
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects simultaneous runs using the same path.
        # The metrics report is written last, after its prediction artifact.
        with output.with_suffix('.predictions.json').open('x') as stream:
            stream.write(json.dumps(records, allow_nan=False)+'\n')
        with output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return report
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--split', choices=('train','val','test'), default='val')
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', choices=('rgb','persistence','future_rgb'))
    parser.add_argument('--feature-metrics', action='store_true')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--manifest')
    parser.add_argument('--horizons', type=int, nargs='+')
    args = parser.parse_args(argv)
    report = evaluate(load_config(args.config), checkpoint=args.checkpoint, split=args.split,
        output=args.output, reference=args.reference, feature_metrics=args.feature_metrics,
        max_samples=args.max_samples, manifest=args.manifest, horizons=args.horizons)
    print(json.dumps({h:v['metrics'] for h,v in report['horizons'].items()}, indent=2))


if __name__ == '__main__':
    main()
