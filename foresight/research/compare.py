"""Paired descriptive comparison; rejects incompatible forecast protocols."""
import argparse
import json
from pathlib import Path


def compare(reports, *, diagnostic=False):
    if len(reports) < 2:
        raise ValueError('Provide a baseline and at least one comparison report')
    base = reports[0]
    expected_kind = 'observed_transition_diagnostic' if diagnostic else 'history_only_forecast'
    if base['evaluation_kind'] != expected_kind:
        raise ValueError('Privileged diagnostics cannot be ranked as forecasting results')
    keys = ('split', 'sample_digest', 'sample_ids', 'samples', 'detector_sha256', 'forecaster_state_sha256', 'evaluation_kind')
    data_keys = ('image_size', 'event_image_size', 'context_frames', 'bins', 'rollout_steps',
                 'sample_history_frames', 'tolerance_us', 'alignment', 'alignment_evidence', 'evaluation_horizons')
    rows = []
    for item in reports:
        for key in keys:
            if item.get(key) != base.get(key):
                raise ValueError(f'Unpaired reports: {key}')
        for key in data_keys:
            if item['config']['data'].get(key) != base['config']['data'].get(key):
                raise ValueError(f'Data protocol mismatch: {key}')
        if item['config']['runtime'].get('precision') != base['config']['runtime'].get('precision'):
            raise ValueError('Evaluation precision differs')
        if item['horizons'].keys() != base['horizons'].keys():
            raise ValueError('Prediction horizons differ')
        horizons = {}
        for h, record in item['horizons'].items():
            deltas = {}
            for metric in ('AP', 'AP50', 'AP75'):
                val, ref = record['metrics'].get(metric), base['horizons'][h]['metrics'].get(metric)
                deltas[metric + '_delta_points'] = None if val is None or ref is None else 100 * (val-ref)
            horizons[h] = dict(metrics=record['metrics'], deltas=deltas, features=record.get('features'))
        rows.append(dict(mode=item['mode'], reference=item.get('reference'), checkpoint=item.get('checkpoint'),
                         seed=item['config']['runtime'].get('seed'), horizons=horizons,
                         precision=item['config']['runtime'].get('precision'),
                         configured_optimization=item['config'].get('optimization'),
                         trainable_parameters=item.get('trainable_parameters'),
                         model_settings=item['config'].get('model'),
                         transition_diagnostics=item.get('transition_diagnostics')))
    return dict(format='event-wm-comparison-v1', sample_digest=base['sample_digest'],
                evaluation_kind=expected_kind, num_samples=len(base['sample_ids']), rows=rows,
                interpretation='Descriptive paired comparison only. Repeat seeds; inspect per-class and per-sequence results before causal or generalization claims.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostic', action='store_true', help='Compare observed-transition diagnostics only, never forecasting')
    args=parser.parse_args(argv)
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report=compare([json.loads(Path(p).read_text()) for p in args.reports], diagnostic=args.diagnostic)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
