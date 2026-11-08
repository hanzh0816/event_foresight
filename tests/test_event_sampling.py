"""Nonuniform temporal bins must preserve causality and the count-rate unit."""

import numpy as np
import pytest
import torch

from foresight.data.events import events_to_voxel, encoding_spec, EventDataset
from foresight.infer_event import make_history
from foresight.train import resume_config
from test_events import events, event_fixture, rgb_fixture


def test_explicit_uniform_is_bitwise_identical_to_e2():
    rng = np.random.default_rng(42)
    source = dict(t=rng.integers(-1, 50001, 1000), x=rng.integers(0, 7, 1000),
                  y=rng.integers(0, 5, 1000), p=rng.integers(0, 2, 1000))
    a = events_to_voxel(source, 0, 50000, (14,28), sensor_size=(5,7))
    b = events_to_voxel(source, 0, 50000, (14,28), sensor_size=(5,7), event_bin_widths_us=[10000]*5)
    assert torch.equal(a,b)


@pytest.mark.parametrize('widths', [[20000,12000,8000,6000,4000], [4000,6000,8000,12000,20000],
                                   [40000]*5, [100000,50000,25000,15000,10000]])
def test_bin_edges_are_half_open_and_normalized_by_duration(widths):
    edges = np.concatenate(([0],np.cumsum(widths)))
    source = events(edges.tolist()+[-1])
    actual = events_to_voxel(source,0,int(edges[-1]),(1,1),sensor_size=(1,1),event_bin_widths_us=widths)
    expected = np.log1p(np.float32(10000)/np.array(widths,dtype=np.float32))/np.log(6)
    torch.testing.assert_close(actual[:5,0,0],torch.tensor(expected,dtype=torch.float32))
    assert not actual[5:].any()


@pytest.mark.parametrize('data', [dict(event_window_us=200000), dict(event_window_us=100000),
                                 dict(event_bin_widths_us=[10000]*4),
                                 dict(event_bin_widths_us=[0,10000,10000,10000,20000]),
                                 dict(event_bin_widths_us=[10000.,10000,10000,10000,10000]),
                                 dict(event_bin_widths_us=[True,9999,10000,10000,20000])])
def test_invalid_protocols_fail(data):
    with pytest.raises(ValueError):
        encoding_spec(data)


def test_long_window_does_not_shift_endpoint_or_future_target(event_fixture):
    root, path, times, offset = event_fixture
    with EventDataset(root,image_size=(28,42)) as short, EventDataset(
            root,image_size=(28,42),event_window_us=200000,event_bin_widths_us=[100000,50000,25000,15000,10000]) as long:
        a,b=short[0],long[0]
        for key in ('sample_id','target_id','target_timestamp_us','event_end_us','input_cutoff_us'):
            assert a['meta'][key]==b['meta'][key]
        assert all(y-x==200000 for x,y in zip(b['meta']['event_start_us'],b['meta']['event_end_us']))
        assert b['history_events'].shape==a['history_events'].shape


def test_raw_inference_uses_same_nonuniform_binning():
    source=events([1,20000,85000,149999,190000,210000,349999])
    widths=[100000,50000,25000,15000,10000]
    spec=encoding_spec(dict(event_window_us=200000,event_bin_widths_us=widths))
    ends=[200000,250000,300000,350000]
    actual=make_history(source,ends,spec,(14,28))
    expected=torch.stack([events_to_voxel(source,end-200000,end,(14,28),event_bin_widths_us=widths) for end in ends])
    assert torch.equal(actual,expected)
