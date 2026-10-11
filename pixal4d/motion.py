"""Sparse Eulerian velocity queries and differentiable physical-time integration.

Velocity is in shared spatial units per second, not flow-matching denoising time.
The sparse support is shared across time and must cover the swept trajectories.
Missing support is an error, never silently interpreted as stationary material.
"""
from dataclasses import dataclass
import itertools
import math

import torch

from .proposals import validate_mesh


class SparseVelocityField:
    def __init__(self, times, coordinates, velocities, *, grid_shape,
                 origin=(-.5, -.5, -.5), spacing):
        """Trilinear space / linear time interpolation on sparse grid NODES.

        coordinates: integer [M,3], unique nodes in grid_shape, arbitrary order.
        velocities: float32/float64 [F,M,3], differentiable coefficients.
        times: [F] increasing seconds; F >= 2. Extrapolation is rejected.
        Node position is origin + coordinates * spacing. This node convention
        differs from the native SLat CELL-center convention in FrameProposal.
        """
        if (velocities.dtype not in (torch.float32, torch.float64) or velocities.ndim != 3
                or velocities.shape[2] != 3 or velocities.shape[0] < 2
                or not velocities.shape[1] or not torch.isfinite(velocities).all()):
            raise ValueError("velocities must be finite float32/float64 [F,M,3]")
        if (coordinates.shape != velocities.shape[1:] or coordinates.dtype not in (torch.int32, torch.int64)
                or coordinates.device != velocities.device):
            raise ValueError("coordinates must be integer [M,3] on the velocity device")
        if (len(grid_shape) != 3 or any(type(n) is not int or n < 2 for n in grid_shape)
                or math.prod(grid_shape) >= 2**63):
            raise ValueError("grid_shape must have three integer dimensions >= 2")
        self.grid_shape = tuple(grid_shape)
        self.shape = torch.tensor(grid_shape, device=velocities.device, dtype=torch.long)
        self.coordinates = coordinates.long()
        if (self.coordinates < 0).any() or (self.coordinates >= self.shape).any():
            raise ValueError("Grid node outside grid_shape")
        self.times = torch.as_tensor(times, dtype=torch.float64, device=velocities.device)
        if (self.times.shape != (len(velocities),) or not torch.isfinite(self.times).all()
                or not (self.times[1:] > self.times[:-1]).all()):
            raise ValueError("times must match F and strictly increase")
        self.origin = torch.as_tensor(origin, dtype=velocities.dtype, device=velocities.device)
        self.spacing = torch.as_tensor(spacing, dtype=velocities.dtype, device=velocities.device).expand(3)
        if (self.origin.shape != (3,) or not torch.isfinite(self.origin).all()
                or not torch.isfinite(self.spacing).all() or not (self.spacing > 0).all()):
            raise ValueError("Expected finite origin and positive spacing")
        self.keys, self.order = torch.sort(self._keys(self.coordinates))
        if (self.keys[1:] == self.keys[:-1]).any():
            raise ValueError("Duplicate grid nodes")
        self.velocities = velocities
        self.corners = torch.tensor(list(itertools.product((0, 1), repeat=3)),
                                    dtype=torch.long, device=velocities.device)

    def _keys(self, coordinates):
        return (coordinates[..., 0] * self.grid_shape[1] + coordinates[..., 1]) * self.grid_shape[2] + coordinates[..., 2]

    def sample(self, points, time_seconds):
        """Return (velocities, supported_mask); unsupported values are undefined.

        time_seconds is one scalar query time for all [N,3] points. No nearest
        neighbor extrapolation, border clamping or sparse-weight renormalization.
        """
        if (points.ndim != 2 or points.shape[1] != 3 or points.dtype != self.velocities.dtype
                or points.device != self.velocities.device or not torch.isfinite(points).all()):
            raise ValueError("points must be finite [N,3] and match velocity dtype/device")
        time = torch.as_tensor(time_seconds, device=self.times.device, dtype=self.times.dtype)
        if time.ndim != 0 or not torch.isfinite(time) or time < self.times[0] or time > self.times[-1]:
            raise ValueError("Query time outside observations; extrapolation is not supported")
        left = (torch.searchsorted(self.times, time, right=True) - 1).clamp(0, len(self.times) - 2)
        alpha = ((time - self.times[left]) / (self.times[left + 1] - self.times[left])).to(points.dtype)
        position = (points - self.origin) / self.spacing
        inside = ((position >= 0) & (position <= self.shape - 1)).all(-1)
        base = position.floor().long().clamp_min(0)
        base = torch.minimum(base, self.shape - 2)
        fraction = position - base
        corners = base[:, None, :] + self.corners
        keys = self._keys(corners)
        indices = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        present = self.keys[indices] == keys
        weights = torch.where(self.corners.bool(), fraction[:, None, :], 1 - fraction[:, None, :]).prod(-1)
        supported = inside & (present | (weights == 0)).all(-1)
        rows = self.order[indices]
        values = (1 - alpha) * self.velocities[left, rows] + alpha * self.velocities[left + 1, rows]
        result = (values * (weights * present)[..., None]).sum(1)
        return result, supported

    def __call__(self, points, time_seconds):
        result, supported = self.sample(points, time_seconds)
        if not supported.all():
            raise ValueError(f"Velocity support missing for {int((~supported).sum())} query points")
        return result


def advect(field, points, start_seconds, end_seconds, *, max_step_seconds=.02, max_steps=10000):
    """Integrate dx/dt=v(x,t) using signed midpoint RK2; retain autograd.

    Split at observed time knots where available. Reverse integration queries the
    SAME field with negative dt; numerical inversion is approximate. This is not
    a guarantee of collision-free motion or a learned correspondence estimator.
    """
    start, end = float(start_seconds), float(end_seconds)
    if not all(math.isfinite(t) for t in (start, end, max_step_seconds)) or max_step_seconds <= 0:
        raise ValueError("Finite endpoints and positive max_step_seconds required")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if (points.ndim != 2 or points.shape[1] != 3 or not points.is_floating_point()
            or not torch.isfinite(points).all()):
        raise ValueError("points must be finite floating [N,3]")

    def query(x, time):
        value = field(x, time)
        if value.shape != x.shape or value.device != x.device or value.dtype != x.dtype or not torch.isfinite(value).all():
            raise ValueError("Field must return finite velocities matching the points")
        return value

    knots = [float(t) for t in getattr(field, 'times', ()) if min(start, end) < float(t) < max(start, end)]
    stops = [start, *sorted(set(knots), reverse=end < start), end]
    counts = [math.ceil(abs(b - a) / max_step_seconds) for a, b in zip(stops, stops[1:])]
    if sum(counts) > max_steps:
        raise ValueError("Integration exceeds max_steps")
    x = points
    for a, b, count in zip(stops, stops[1:], counts):
        if not count:
            continue
        direction = 1. if b > a else -1.
        for i in range(count):
            # Keep earlier integration nodes fixed as the query endpoint moves.
            # Redistributing the whole interval across ceil(duration / step)
            # nodes introduces a jump whenever that integer changes.
            t = a + direction * i * max_step_seconds
            t = min(t, b) if direction > 0 else max(t, b)
            dt = direction * min(max_step_seconds, abs(b - t))
            if not dt:
                continue
            midpoint = x + .5 * dt * query(x, t)
            x = x + dt * query(midpoint, t + .5 * dt)
    query(x, end)  # Also validate final spatial/time support, including zero duration.
    return x


@dataclass
class MeshTrajectory:
    """One reference mesh advected by a supplied field; connectivity stays fixed.

    Construct the reference from appropriately consolidated geometry. This class
    does not perform consolidation, predict velocity, remesh or generate PBR.
    """
    vertices: torch.Tensor
    faces: torch.Tensor
    reference_time_seconds: float
    field: object

    def __post_init__(self):
        validate_mesh(self.vertices, self.faces)
        if not math.isfinite(self.reference_time_seconds):
            raise ValueError("Reference time must be finite")

    def vertices_at(self, time_seconds, *, max_step_seconds=.02):
        return advect(self.field, self.vertices, self.reference_time_seconds,
                      time_seconds, max_step_seconds=max_step_seconds)
