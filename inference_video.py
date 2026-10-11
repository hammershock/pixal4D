"""Generate a mesh animation from masked frames using a causal Pixal3D adapter."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True, help='Directory of ordered, equally framed RGBA PNGs')
    parser.add_argument('--cameras', type=Path, required=True, help='JSON camera object or one object per frame')
    parser.add_argument('--adapters', type=Path, help='Directory containing adapter config.json and safetensors')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='TencentARC/Pixal3D')
    parser.add_argument('--image-encoder', default='camenduru/dinov3-vitl16-pretrain-lvd1689m')
    parser.add_argument('--naf-source', type=Path, help='Optional local NAF checkout for offline inference')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--faces', type=int, default=100000)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    import o_voxel
    from safetensors.torch import load_file
    from inference import IMAGE_COND_CONFIGS, build_image_cond_model
    from pixal3d.pipelines.base import Pipeline
    from pixal3d.pipelines import samplers
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    import torch
    from pixal4d.pipeline import Pixal4DVideoPipeline

    paths = sorted(args.frames.glob('*.png'))
    if not paths:
        parser.error('No PNG frames found')
    config = json.loads((args.adapters / 'config.json').read_text()) if args.adapters else {}
    if args.adapters and (not isinstance(config.get('stages'), dict) or not config['stages']):
        parser.error('Adapter config needs a nonempty stages mapping')
    if args.adapters and any(stage not in Pixal4DVideoPipeline.stages for stage in config['stages']):
        parser.error('Adapter config contains an unsupported temporal stage')
    maximum = config.get('max_frames', 16)
    if len(paths) > maximum:
        parser.error(f'This model supports at most {maximum} frames')
    cameras = json.loads(args.cameras.read_text())
    checks = [cameras] if isinstance(cameras, dict) else cameras
    if not isinstance(cameras, dict) and len(cameras) != len(paths):
        parser.error('Camera array must match the frame count')
    for camera in checks:
        if not 0 < camera['camera_angle_x'] < np.pi or camera['distance'] <= 0 or camera.get('mesh_scale', 1) <= 0:
            parser.error('Camera FOV must be in radians and distance/scale must be positive')
    images = []
    for path in paths:
        rgba = Image.open(path)
        if rgba.mode != 'RGBA':
            parser.error(f'{path.name} needs an alpha mask')
        data = np.asarray(rgba).astype(np.float32) / 255
        if not np.any(data[..., 3] > 0):
            parser.error(f'{path.name} has an empty foreground mask')
        # Preserve framing/calibration. Do not independently recenter each frame.
        images.append(Image.fromarray((data[..., :3] * data[..., 3:] * 255).astype(np.uint8)))
    if len({image.size for image in images}) != 1:
        parser.error('Frames must have identical image dimensions')
    # Load the upstream components without an unused background-removal model.
    loaded = Pipeline.from_pretrained(args.model)
    settings = loaded._pretrained_args
    options = {'models': loaded.models, 'low_vram': True, 'rembg_model': None}
    for kind in ('sparse_structure', 'shape_slat', 'tex_slat'):
        spec = settings[kind + '_sampler']
        options[kind + '_sampler'] = getattr(samplers, spec['name'])(**spec['args'])
        options[kind + '_sampler_params'] = spec['params']
    for kind in ('shape_slat', 'tex_slat'):
        options[kind + '_normalization'] = settings[kind + '_normalization']
    naf = None
    if args.naf_source:
        naf = torch.hub.load(str(args.naf_source), 'naf', source='local', pretrained=True, device='cpu').eval()
        naf.requires_grad_(False)
    for stage, parameters in IMAGE_COND_CONFIGS.items():
        feature = build_image_cond_model(dict(parameters, model_name=args.image_encoder))
        if feature.use_naf_upsample and naf is not None:
            feature.naf_model = naf
        options['image_cond_model_' + stage] = feature
    native = Pixal3DImageTo3DPipeline(**options)
    native.to('cuda')
    # The manifest selects stages explicitly, retaining older two-stage bundles.
    video = Pixal4DVideoPipeline(native, max_frames=maximum, adapter_config=config.get('adapter'),
                                stages=config['stages'] if args.adapters else None)
    if args.adapters:
        for stage, filename in config['stages'].items():
            video.load_adapters(stage, load_file(str(args.adapters / filename)))
    else:
        print('Using zero-initialized temporal adapters; this is the untrained Pixal3D video initialization.')
    args.output.mkdir(parents=True, exist_ok=True)
    def progress(stage, index, total):
        print(f'{stage}: {index}/{total}', flush=True)
    outputs = video.run(images, cameras, seed=args.seed, preprocess_image=False,
                        progress=progress, return_latent=True, pipeline_type='1024_cascade',
                        sparse_structure_sampler_params={'steps': args.steps},
                        shape_slat_sampler_params={'steps': args.steps},
                        tex_slat_sampler_params={'steps': args.steps})
    frames = []
    for index, (meshes, (_, _, resolution)) in enumerate(outputs):
        mesh = meshes[0]
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=native.pbr_attr_layout, grid_size=resolution,
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=args.faces,
            texture_size=args.texture_size, remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
        # o_voxel exports (x, z, -y). Restore the pipeline's view-aligned
        # coordinates with one fixed rotation, shared by every frame.
        glb.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                     [0, 1, 0, 0], [0, 0, 0, 1]]))
        filename = f'{index:04d}.glb'
        glb.export(args.output / filename)
        frames.append({'index': index, 'source': paths[index].name, 'mesh': filename})
        print(f'exported: {index + 1}/{len(outputs)}', flush=True)
    (args.output / 'animation.json').write_text(json.dumps({'seed': args.seed, 'frames': frames,
        'temporal_stages': list(video.flows),
        'coordinates': 'native Pixal3D view-aligned coordinates; no per-frame registration'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
