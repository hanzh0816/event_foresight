"""Configuration and external-input time contracts; run only on the test host."""

import pytest
import torch

from foresight import timing
from foresight.timing import ForecastTiming, prepare_prediction_times


def test_default_timing_prepares_regular_cpu_inputs_without_external_checks(monkeypatch):
    def unexpected_check(*args, **kwargs):
        raise AssertionError('Regular configured times must not use external-input validation')

    monkeypatch.setattr(timing, '_times', unexpected_check)
    layout = ForecastTiming.from_config({'data': {}})
    sequence = layout.sequence_kwargs()
    future = layout.forecast_kwargs(horizon_steps=3)
    torch.testing.assert_close(sequence['context_times'], torch.tensor([0., .05, .10, .15], dtype=torch.float64))
    torch.testing.assert_close(sequence['query_times'], torch.tensor([.05, .10, .15, .20], dtype=torch.float64))
    torch.testing.assert_close(future['query_times'], torch.tensor([.20, .25, .30], dtype=torch.float64))
    assert sequence['event_times'] is sequence['context_times']
    assert future['event_times'] is sequence['event_times']
    assert all(value.device.type == 'cpu' for value in sequence.values())
    assert all(value.device.type == 'cpu' for value in future.values())


def test_custom_config_preserves_event_times_across_future_horizons():
    layout = ForecastTiming.from_config({
        'data': {'context_frames': 4, 'frame_interval_s': .05},
        'timing': {'context_times_s': [.01, .06, .11, .16],
                   'event_times_s': [0., .04, .09, .14], 'forecast_interval_s': .07},
    })
    sequence = layout.sequence_kwargs()
    forecast = layout.forecast_kwargs(horizon_steps=3)
    torch.testing.assert_close(sequence['query_times'], torch.tensor([.06, .11, .16, .23], dtype=torch.float64))
    torch.testing.assert_close(forecast['query_times'], torch.tensor([.23, .30, .37], dtype=torch.float64))
    torch.testing.assert_close(forecast['event_times'], torch.tensor([0., .04, .09, .14], dtype=torch.float64))
    assert forecast['event_times'] is sequence['event_times']


@pytest.mark.parametrize('context_times,query_times', [
    ([0., .05, .10, .15], [0.]),
    ([0., .05, .10, .15], [-.01]),
    ([0., .05, .05, .15], [.2]),
    ([0., .10, .05, .15], [.2]),
    ([0., .05, .10], [.2]),
    ([0., .05, .10, float('nan')], [.2]),
    ([0., .05, .10, .15], [float('inf')]),
])
def test_invalid_or_causally_empty_external_timestamps_are_rejected(context_times, query_times):
    with pytest.raises(ValueError):
        prepare_prediction_times(context_times, query_times, context_frames=4)


def test_external_rollout_rejects_events_after_initial_observation_cutoff():
    with pytest.raises(ValueError, match='Events'):
        prepare_prediction_times([0., .05, .10, .15], [.20, .25, .30],
                                 event_times=[.05, .10, .15, .20], rollout=True)


@pytest.mark.parametrize('context_times,interval', [([1e16, 1e16 + 2], .05), ([0., 1e308], 1e308)])
def test_custom_config_rejects_timestamp_rounding_or_overflow(context_times, interval):
    with pytest.raises(ValueError):
        ForecastTiming.from_config({
            'data': {'context_frames': 2, 'frame_interval_s': .05},
            'timing': {'context_times_s': context_times, 'forecast_interval_s': interval},
        })


def test_external_timing_preserves_tensor_gradients_and_explicit_cutoff():
    context = torch.tensor([0., .05, .10, .15], dtype=torch.float64, requires_grad=True)
    queries = torch.tensor([.20, .30], dtype=torch.float64, requires_grad=True)
    events = torch.tensor([0., .04, .08, .14], dtype=torch.float64, requires_grad=True)
    prepared = prepare_prediction_times(context, queries, event_times=events,
                                        observation_times=[.05, .12], context_frames=4, event_frames=4)
    assert prepared['context_times'] is context
    assert prepared['query_times'] is queries
    assert prepared['event_times'] is events
    torch.testing.assert_close(prepared['observation_times'], torch.tensor([.05, .12], dtype=torch.float64))
