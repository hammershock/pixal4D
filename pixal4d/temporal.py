"""Zero-initialized causal residuals around unmodified Pixal3D flow blocks."""
from contextlib import ExitStack
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def relative_timestamps(timestamps, valid_frames):
    """Validate seconds and remove the first valid time; padding has no clock."""
    if isinstance(timestamps, torch.Tensor):
        if timestamps.ndim != 1:
            raise ValueError('timestamps must be one-dimensional')
        timestamps = timestamps.detach().cpu().tolist()
    if timestamps is None or len(timestamps) != len(valid_frames):
        raise ValueError('Supply one timestamp in seconds per frame; padding may use None')
    result, origin, previous = [], None, None
    for timestamp, valid in zip(timestamps, valid_frames):
        if not valid:
            result.append(None)
            continue
        if isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not math.isfinite(timestamp):
            raise ValueError('Valid frames require finite numeric timestamps in seconds')
        timestamp = float(timestamp)
        if previous is not None and timestamp <= previous:
            raise ValueError('Valid timestamps must strictly increase')
        if origin is None:
            origin = timestamp
        elapsed = timestamp - origin
        if not math.isfinite(elapsed):
            raise ValueError('Timestamp differences must be finite')
        result.append(elapsed)
        previous = timestamp
    return result


class TemporalResidual(nn.Module):
    """Attend to spatially pooled tokens from the current and earlier frames."""

    def __init__(self, channels, width=128, heads=4, memory_grid=4, time_scale=1.0):
        super().__init__()
        if min(channels, width, heads, memory_grid) < 1 or width % heads or (width // heads) % 2:
            raise ValueError('Invalid temporal dimensions')
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be a positive number of seconds')
        self.heads = heads
        self.time_scale = time_scale
        self.memory_grid = memory_grid
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, width)
        self.key_value = nn.Linear(channels, width * 2)
        self.position = nn.Linear(3, width, bias=False)
        # Rotary phases are continuous in seconds. Q/K dot products depend on
        # pairwise time differences, without a learned frame-index table.
        head_width = width // heads
        self.register_buffer('time_frequencies',
                             10000.0 ** (-torch.arange(0, head_width, 2).float() / head_width),
                             persistent=False)
        self.output = nn.Linear(width, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def rotate_time(self, values, timestamp):
        pairs = values.float().reshape(-1, self.heads, values.shape[-1] // self.heads // 2, 2)
        phase = timestamp / self.time_scale
        if not math.isfinite(phase):
            raise ValueError('Timestamp exceeds the configured time scale')
        angle = self.time_frequencies.float() * phase
        cosine, sine = angle.cos(), angle.sin()
        first, second = pairs.unbind(-1)
        rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), -1)
        return rotated.reshape_as(values).to(values.dtype)

    def forward(self, features, coords, resolution, timestamp, memory):
        # Upstream blocks may be bf16; residual parameters stay fp32 by default.
        values = self.norm(features.to(self.norm.weight.dtype))
        position = (coords.to(values.dtype) + .5) / resolution * 2 - 1
        q = self.query(values) + self.position(position)
        q = self.rotate_time(q, timestamp)
        cells = (coords.long() * self.memory_grid // resolution).clamp(0, self.memory_grid - 1)
        keys = (cells[:, 0] * self.memory_grid + cells[:, 1]) * self.memory_grid + cells[:, 2]
        unique, inverse, counts = keys.unique(sorted=True, return_inverse=True, return_counts=True)
        pooled = values.new_zeros(len(unique), values.shape[-1]).index_add(0, inverse, values)
        pooled = pooled / counts[:, None]
        positions = position.new_zeros(len(unique), 3).index_add(0, inverse, position) / counts[:, None]
        k, v = self.key_value(pooled).chunk(2, -1)
        k = k + self.position(positions)
        k = self.rotate_time(k, timestamp)
        memory.append((k, v))
        k = torch.cat([entry[0] for entry in memory])
        v = torch.cat([entry[1] for entry in memory])
        def heads(tensor):
            return tensor.reshape(-1, self.heads, tensor.shape[-1] // self.heads).transpose(0, 1)[None]
        attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        attended = attended[0].transpose(0, 1).reshape(features.shape[0], -1)
        return features + self.output(attended).to(features.dtype)


class CausalVideoFlow(nn.Module):
    """Run one variable-support sequence through an existing dense or sparse flow.

    ``frames`` and ``conditions`` are equally sized lists, with one batch element
    per valid frame. ``valid_frames`` marks padding; padded values are never read
    and their outputs are ``None``. Each call starts fresh temporal memory, so CFG
    branches and diffusion steps cannot leak into one another. Sequence execution
    is serial and must not overlap another call using the same base model.
    ``timestamps`` are strictly increasing physical seconds for valid frames;
    the diffusion ``timestep`` is a separate variable. Padding may use None.

    Zero output projections preserve the base flow exactly at initialization.
    This does not tie random noise across frames; the sampler must share noise
    when testing identical-frame asset equivalence.
    """

    def __init__(self, base, block_indices=(7, 15, 23, 29), width=128,
                 heads=4, max_frames=16, memory_grid=4, time_encoding='relative_rope_v1', time_scale=1.0):
        super().__init__()
        indices = tuple(block_indices)
        if time_encoding != 'relative_rope_v1' or max_frames < 1:
            raise ValueError('Expected continuous relative_rope_v1 and a positive frame limit')
        if not indices or len(set(indices)) != len(indices):
            raise ValueError('Choose unique temporal block indices')
        if any(i < 0 or i >= len(base.blocks) for i in indices):
            raise ValueError('Temporal block index outside base model')
        if getattr(base, 'use_checkpoint', False):
            raise ValueError('Stateful block checkpointing is unsupported; checkpoint complete sequence calls instead')
        self.base = base
        self.max_frames = max_frames
        self.config = dict(block_indices=list(indices), width=width, heads=heads,
                           max_frames=max_frames, memory_grid=memory_grid,
                           time_encoding=time_encoding, time_scale=time_scale)
        self.adapters = nn.ModuleDict({str(i): TemporalResidual(base.model_channels, width, heads, memory_grid, time_scale)
                                       for i in indices})
        self._active = False

    @property
    def device(self):
        return self.base.device

    def forward(self, frames, timestep, conditions, valid_frames=None, frame_kwargs=None, *, timestamps=None):
        length = len(frames)
        if not 1 <= length <= self.max_frames or len(conditions) != length:
            raise ValueError(f'Expected 1..{self.max_frames} frames and matching conditions')
        if valid_frames is None:
            valid_frames = [True] * length
        if isinstance(valid_frames, torch.Tensor):
            if valid_frames.dtype != torch.bool or valid_frames.ndim != 1:
                raise ValueError('valid_frames must be a one-dimensional boolean mask')
            valid_frames = valid_frames.tolist()
        if len(valid_frames) != length or any(type(v) is not bool for v in valid_frames):
            raise ValueError('valid_frames must contain one boolean per frame')
        times = relative_timestamps(timestamps, valid_frames)
        if frame_kwargs is None:
            frame_kwargs = [{} for _ in frames]
        if len(frame_kwargs) != length:
            raise ValueError('frame_kwargs must match the sequence length')
        if self._active:
            raise RuntimeError('Concurrent calls sharing a base flow are unsupported')
        self._active = True
        memory = {key: [] for key in self.adapters}
        current = [0]
        try:
            with ExitStack() as stack:
                for key, adapter in self.adapters.items():
                    def hook(module, inputs, output, adapter=adapter, key=key):
                        if hasattr(output, 'feats'):
                            if output.shape[0] != 1:
                                raise ValueError('Each frame must have batch size one')
                            coords = output.coords[:, 1:]
                            result = adapter(output.feats, coords, self.base.resolution, current[0], memory[key])
                            return output.replace(result)
                        if output.shape[0] != 1 or output.ndim != 3:
                            raise ValueError('Expected dense block features [1,N,C]')
                        r = self.base.resolution
                        if output.shape[1] != r**3:
                            raise ValueError('Unexpected dense flow token layout')
                        coords = torch.stack(torch.meshgrid(*[torch.arange(r, device=output.device)]*3, indexing='ij'),-1).reshape(-1,3)
                        return adapter(output[0], coords, r, current[0], memory[key])[None]
                    handle = self.base.blocks[int(key)].register_forward_hook(hook)
                    stack.callback(handle.remove)
                result = []
                for index, valid in enumerate(valid_frames):
                    current[0] = times[index]
                    if not valid:
                        result.append(None)
                        continue
                    if frames[index] is None:
                        raise ValueError('Valid frames cannot be None')
                    result.append(self.base(frames[index], timestep, conditions[index], **frame_kwargs[index]))
                return result
        finally:
            self._active = False


class CausalFlowStream:
    """Replay the native sampler per frame, retaining memory per model invocation.

    Conditional and unconditional calls have separate memory. All valid frames
    must use the same sampler schedule. A stream belongs to one video only.
    """

    def __init__(self, flow):
        self.flow = flow
        self.memory = {}
        self.schedule = None
        self.last_frame = -1
        self.time_origin = None
        self.last_time = None

    def frame(self, index, timestamp):
        from contextlib import contextmanager

        @contextmanager
        def context():
            flow = self.flow
            if not self.last_frame < index < flow.max_frames or flow._active:
                raise ValueError('Stream frames must be sequential, within bounds, and nonoverlapping')
            times = [timestamp] if self.last_time is None else [self.last_time, timestamp]
            relative_timestamps(times, [True] * len(times))
            seconds = float(timestamp)
            origin = seconds if self.time_origin is None else self.time_origin
            elapsed = seconds - origin
            flow._active = True
            current_call = [-1]
            schedule = []
            def begin(module, inputs):
                current_call[0] += 1
                schedule.append(tuple(inputs[1].detach().cpu().tolist()))
            try:
                with ExitStack() as stack:
                    stack.callback(flow.base.register_forward_pre_hook(begin).remove)
                    for key, adapter in flow.adapters.items():
                        def hook(module, inputs, output, key=key, adapter=adapter):
                            sparse = hasattr(output, 'feats')
                            features = output.feats if sparse else output[0]
                            if output.shape[0] != 1:
                                raise ValueError('Stream supports one sample per frame')
                            r = flow.base.resolution
                            if sparse:
                                coords = output.coords[:, 1:]
                            else:
                                coords = torch.stack(torch.meshgrid(*[torch.arange(r, device=features.device)]*3,indexing='ij'),-1).reshape(-1,3)
                            adapter.to(features.device)
                            history = self.memory.setdefault((current_call[0], key), [])
                            value = adapter(features, coords, r, elapsed, history)
                            return output.replace(value) if sparse else value[None]
                        stack.callback(flow.base.blocks[int(key)].register_forward_hook(hook).remove)
                    yield
                if self.schedule is not None and schedule != self.schedule:
                    raise ValueError('Sampler call schedule changed between video frames')
                self.schedule = schedule
                self.last_frame = index
                self.time_origin = origin
                self.last_time = seconds
            finally:
                flow._active = False
        return context()
