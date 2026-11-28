"""CPU time layouts prepared at configuration or external-input boundaries."""

import math

import torch


def _times(values, name, count=None):
    # as_tensor preserves the autograd path when callers supply a tensor.
    values = torch.as_tensor(values, device="cpu", dtype=torch.float64)
    if values.ndim != 1 or values.numel() < 1 or (count is not None and len(values) != count):
        raise ValueError(f"{name} must be a nonempty 1D array" +
                         (f" of length {count}" if count is not None else ""))
    if not torch.isfinite(values).all() or (values[1:] <= values[:-1]).any():
        raise ValueError(f"{name} must be finite and strictly increasing")
    return values


def prepare_prediction_times(context_times, query_times, *, observation_times=None,
                             event_times=None, context_frames=None, event_frames=None,
                             rollout=False):
    """Validate caller-supplied timestamps once before invoking the model.

    Times are seconds, shared across the batch. Returned tensors stay on CPU;
    a supplied tensor retains its autograd path. Rollout queries must all follow
    the initial context, and its Event inputs cannot extend past that context.
    """
    context = _times(context_times, "context_times", context_frames)
    queries = _times(query_times, "query_times")
    if event_times is None:
        events = context
        if event_frames is not None and len(events) != event_frames:
            raise ValueError("event_times must match the number of Event frames")
    else:
        events = _times(event_times, "event_times", event_frames)
    if observation_times is None:
        if rollout:
            observations = context[-1:].expand_as(queries)
        else:
            indices = torch.searchsorted(context, queries, right=False) - 1
            if (indices < 0).any():
                raise ValueError("Every query must have at least one strictly earlier context frame")
            observations = context[indices]
    else:
        observations = torch.as_tensor(observation_times, device="cpu", dtype=torch.float64)
        if (observations.shape != queries.shape or not torch.isfinite(observations).all()
                or (observations < context[0]).any() or (observations >= queries).any()):
            raise ValueError("Each observation_time must precede its query and allow at least one RGB frame")
    if rollout:
        if (queries <= context[-1]).any() or (events > context[-1]).any():
            raise ValueError("Rollout queries must follow the initial context; Events must not exceed it")
        if observation_times is not None and (observations != context[-1]).any():
            raise ValueError("Rollout uses the initial context endpoint as its observed-data cutoff")
    result = dict(context_times=context, query_times=queries, event_times=events)
    if not rollout:
        result["observation_times"] = observations
    return result


class ForecastTiming:
    """Reusable CPU metadata; regular defaults need no timestamp validation."""

    @classmethod
    def from_config(cls, config):
        data, options = config["data"], config.get("timing", {})
        frames, interval = data.get("context_frames", 4), data.get("frame_interval_s", 0.05)
        result = cls()
        result._custom = any(value is not None for value in options.values())
        result.frame_interval_s = (interval if options.get("forecast_interval_s") is None
                                   else float(options["forecast_interval_s"]))
        result.context_times = (torch.arange(frames, device="cpu", dtype=torch.float64) * interval
                                if options.get("context_times_s") is None else
                                _times(options["context_times_s"], "timing.context_times_s", frames))
        result.event_times = (result.context_times if options.get("event_times_s") is None else
                              _times(options["event_times_s"], "timing.event_times_s", frames))
        if result._custom:
            if not math.isfinite(result.frame_interval_s) or result.frame_interval_s <= 0:
                raise ValueError("timing.forecast_interval_s must be finite and positive")
            if (result.event_times > result.context_times[-1]).any():
                raise ValueError("Configured Event times cannot exceed the initial RGB observation cutoff")
        result._forecasts = {}
        future = result.forecast_kwargs()["query_times"]
        result._sequence = dict(context_times=result.context_times,
                                query_times=torch.cat((result.context_times[1:], future)),
                                event_times=result.event_times)
        return result

    def sequence_kwargs(self):
        """Times for parallel next-frame targets; callers must not mutate them."""
        return self._sequence.copy()

    def forecast_kwargs(self, horizon_steps=1):
        """Times for future frames, cached once per requested horizon."""
        if type(horizon_steps) is not int or horizon_steps < 1:
            raise ValueError("horizon_steps must be a positive integer")
        if horizon_steps not in self._forecasts:
            queries = (self.context_times[-1:] +
                       torch.arange(1, horizon_steps + 1, device="cpu", dtype=torch.float64) * self.frame_interval_s)
            if self._custom:
                queries = _times(queries, "query_times", horizon_steps)
                if not queries[0] > self.context_times[-1]:
                    raise ValueError("The forecast interval must advance the configured context timestamp")
            self._forecasts[horizon_steps] = dict(context_times=self.context_times,
                                                 query_times=queries, event_times=self.event_times)
        return self._forecasts[horizon_steps].copy()


def validate_dsec_timing(config):
    """Custom positional times cannot relabel DSEC's fixed 50 ms targets."""
    if not any(value is not None for value in config.get("timing", {}).values()):
        return
    timing = ForecastTiming.from_config(config)
    interval = config["data"].get("frame_interval_s", 0.05)
    if (not torch.allclose(timing.context_times.diff(),
                           torch.full_like(timing.context_times[1:], interval), rtol=1e-6, atol=1e-9)
            or not torch.allclose(timing.event_times, timing.context_times, rtol=0, atol=1e-9)
            or not math.isclose(timing.frame_interval_s, interval, rel_tol=1e-6, abs_tol=1e-9)):
        raise ValueError("DSEC timing must retain 50 ms RGB/target intervals and matching Event times; "
                         "timing configuration does not resample the dataset")
