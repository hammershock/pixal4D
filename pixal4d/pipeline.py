"""Causal video inference reusing the complete native Pixal3D asset pipeline."""
from contextlib import ExitStack

import torch

from .temporal import CausalFlowStream, CausalVideoFlow


class Pixal4DVideoPipeline:
    """Extend an initialized Pixal3D pipeline without changing its decoders.

    The default adapters affect sparse structure and coarse geometry. High
    resolution geometry and PBR material generation retain their native priors.
    This produces independent meshes per time step; shared topology is not
    assumed. Input masks can be supplied as image alpha channels. Camera
    parameters follow the wrapped pipeline's image-to-3D convention.
    """
    stages = ('sparse_structure_flow_model', 'shape_slat_flow_model_512')

    def __init__(self, pipeline, max_frames=16, adapter_config=None):
        self.pipeline = pipeline
        config = dict(adapter_config or {})
        config['max_frames'] = max_frames
        self.flows = {name: CausalVideoFlow(pipeline.models[name], **config) for name in self.stages}
        self.max_frames = max_frames

    def load_adapters(self, stage, state_dict):
        """Load a stage's temporal adapter tensors; base weights stay untouched."""
        if stage not in self.flows:
            raise ValueError(f'Unsupported temporal stage: {stage}')
        self.flows[stage].adapters.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def run(self, images, camera_params, *, seed=42, valid_frames=None,
            progress=None, **native_options):
        length = len(images)
        if not 1 <= length <= self.max_frames:
            raise ValueError(f'Expected 1..{self.max_frames} input frames')
        if valid_frames is None:
            valid_frames = [True] * length
        if isinstance(valid_frames, torch.Tensor):
            if valid_frames.dtype != torch.bool or valid_frames.ndim != 1:
                raise ValueError('Expected a one-dimensional boolean frame mask')
            valid_frames = valid_frames.tolist()
        if len(valid_frames) != length or any(type(v) is not bool for v in valid_frames):
            raise ValueError('Frame padding mask must match the video length')
        if isinstance(camera_params, dict):
            camera_params = [camera_params] * length
        if len(camera_params) != length:
            raise ValueError('Camera parameters must match the video length')
        if native_options.get('num_samples', 1) != 1:
            raise ValueError('Video inference supports one sample per frame')
        streams = {name: CausalFlowStream(flow) for name, flow in self.flows.items()}
        results = []
        for index, valid in enumerate(valid_frames):
            if not valid:
                results.append(None)
                continue
            if progress is not None:
                progress('generating', index, length)
            with ExitStack() as stack:
                for stream in streams.values():
                    stack.enter_context(stream.frame(index))
                # Native run resets its RNG. Repeated inputs thus reproduce the
                # same complete native PBR asset at zero adapter initialization.
                asset = self.pipeline.run(images[index], camera_params[index], seed=seed, **native_options)
            results.append(asset)
            if progress is not None:
                progress('completed', index + 1, length)
        return results
