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
    keys = ('split', 'sample_digest', 'sample_ids', 'samples', 'detector_sha256', 'evaluation_kind')
    # A tuned predictor is the scientific variable, while encoder/PCA define
    # the common feature coordinate system. Legacy reports only fingerprinted
    # the entire forecaster, and cannot establish compatibility with new ones.
    component_keys = ('feature_space_state_sha256', 'predictor_state_sha256')
    modern = []
    for item in reports:
        present = [key in item for key in component_keys]
        if any(present) and not all(present):
            raise ValueError('Incomplete component state provenance')
        modern.append(all(present))
    if any(modern) and not all(modern):
        raise ValueError('Cannot compare mixed legacy and component state provenance')
    state_key = 'feature_space_state_sha256' if all(modern) else 'forecaster_state_sha256'
    for item in reports:
        required = (*component_keys, 'forecaster_state_sha256') if all(modern) else ('forecaster_state_sha256',)
        if any(not isinstance(item.get(key), str) or not item[key] for key in required):
            raise ValueError('Missing or invalid model state provenance')
        if item[state_key] != base[state_key]:
            raise ValueError(f'Unpaired reports: {state_key}')
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
                         parameter_counts=item.get('parameter_counts'),
                         reference_provenance=item.get('reference_provenance'),
                         initialization_provenance=item.get('initialization_provenance'),
                         predictor_initialization=item.get('predictor_initialization'),
                         forecaster_state_sha256=item.get('forecaster_state_sha256'),
                         predictor_state_sha256=item.get('predictor_state_sha256'),
                         model_settings=item['config'].get('model'),
                         transition_diagnostics=item.get('transition_diagnostics')))
    return dict(format='event-wm-comparison-v1', sample_digest=base['sample_digest'],
                compatibility_state_key=state_key, compatibility_state_sha256=base[state_key],
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
