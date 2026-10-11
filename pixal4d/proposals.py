"""Per-frame native geometry and SLat features retained for motion estimation.

Coordinates must already share a sequence-wide spatial frame and scale. This
module does not estimate cameras, align frames or establish point identities.
"""
from dataclasses import dataclass
import math

import torch


def validate_mesh(vertices, faces):
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not vertices.is_floating_point() or not torch.isfinite(vertices).all()):
        raise ValueError("vertices must be a nonempty finite floating [V,3] tensor")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype not in (torch.int32, torch.int64)
            or faces.device != vertices.device):
        raise ValueError("faces must be a nonempty integer [T,3] tensor on the mesh device")
    if (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("Face index outside vertices")


@dataclass(frozen=True)
class FrameProposal:
    time_seconds: float
    vertices: torch.Tensor
    faces: torch.Tensor
    feature_positions: torch.Tensor
    features: torch.Tensor
    coordinate_system: str

    def __post_init__(self):
        validate_mesh(self.vertices, self.faces)
        if not math.isfinite(self.time_seconds) or not self.coordinate_system:
            raise ValueError("Declare a finite timestamp and shared coordinate system")
        for name, value in (("feature_positions", self.feature_positions), ("features", self.features)):
            if (value.ndim != 2 or not len(value) or not value.shape[1]
                    or not value.is_floating_point() or not torch.isfinite(value).all()
                    or value.device != self.vertices.device):
                raise ValueError(f"Invalid {name} tensor")
        if self.feature_positions.shape != (len(self.features), 3):
            raise ValueError("Each feature needs its own spatial position")

    @classmethod
    def from_native(cls, time_seconds, mesh, shape_slat, *, latent_grid_resolution,
                    coordinate_system):
        """Wrap one unbatched native output without resampling or detaching features.

        Native SLat coordinates index cell centers inside [-.5,.5]^3. Pass the
        latent grid resolution, NOT the high-resolution decoded voxel resolution.
        The wrapper does not change the native decoder or require CUDA imports.
        """
        coords = shape_slat.coords
        if (not isinstance(latent_grid_resolution, int) or latent_grid_resolution <= 0
                or coords.ndim != 2 or coords.shape[1] != 4
                or coords.dtype not in (torch.int32, torch.int64)
                or not len(coords) or (coords[:, 0] != 0).any()
                or (coords[:, 1:] < 0).any() or (coords[:, 1:] >= latent_grid_resolution).any()
                or len(coords) != len(shape_slat.feats)
                or coords.device != mesh.vertices.device):
            raise ValueError("Expected one native SLat batch with valid integer cell coordinates")
        if len(torch.unique(coords[:, 1:], dim=0)) != len(coords):
            raise ValueError("Duplicate native SLat coordinates")
        positions = (coords[:, 1:].to(dtype=mesh.vertices.dtype) + .5) / latent_grid_resolution - .5
        return cls(float(time_seconds), mesh.vertices, mesh.faces, positions,
                   shape_slat.feats, coordinate_system)


def validate_sequence(proposals):
    """Require ordered real frames; token counts and mesh topology may differ."""
    frames = tuple(proposals)
    if len(frames) < 2:
        raise ValueError("Motion estimation requires at least two real frames")
    for a, b in zip(frames, frames[1:]):
        if a.time_seconds >= b.time_seconds:
            raise ValueError("Timestamps must strictly increase in seconds")
        if (a.coordinate_system != b.coordinate_system
                or a.vertices.device != b.vertices.device
                or a.features.shape[1] != b.features.shape[1]):
            raise ValueError("Frames must share coordinates, device and feature width")
    return frames
