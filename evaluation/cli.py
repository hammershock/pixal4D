"""The single public evaluation entry; no model checkpoints or private imports."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .geometry import load_mesh, surface_metrics


PATH_COLUMNS = ('pred_mesh', 'gt_mesh', 'camera', 'pred_transform', 'gt_transform', 'pred_rgb', 'gt_rgb')
COLUMNS = {'sequence_id', 'frame_id', 'timestamp', 'reference_length', 'view_id', 'view_type', *PATH_COLUMNS}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def matrix(value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Expected finite matrix with shape {shape}')
    return value


def transform(path):
    value = np.eye(4) if path is None else matrix(json.loads(path.read_text()), (4, 4))
    if not np.allclose(value[3], [0, 0, 0, 1]) or abs(np.linalg.det(value[:3, :3])) < 1e-12:
        raise ValueError('Mesh transforms must be invertible affine matrices')
    return value


def read_camera(path):
    camera = json.loads(path.read_text())
    k = matrix(camera['K'], (3, 3))
    ext = matrix(camera['world_to_camera'], (4, 4))
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('K must contain positive focal lengths in pixels')
    if (not np.allclose(ext[3], [0, 0, 0, 1]) or
            not np.allclose(ext[:3, :3].T @ ext[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(ext[:3, :3]), 1, atol=1e-5)):
        raise ValueError('world_to_camera must be a rigid, right-handed OpenCV transform')
    if any(type(camera[key]) is not int or camera[key] < 32 for key in ('width', 'height')):
        raise ValueError('Camera width/height must be integers >= 32')
    if not 0 < camera.get('near', .001) < camera.get('far', 100.) < float('inf'):
        raise ValueError('Invalid camera clipping planes')
    return camera


def read_manifest(path, metrics):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) - COLUMNS:
            raise ValueError(f'Duplicate/unknown CSV columns; allowed: {sorted(COLUMNS)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty evaluation manifest')
    required = {'sequence_id', 'frame_id', 'timestamp'}
    if set(metrics) & {'geometry', 'pixel'}:
        required |= {'pred_mesh', 'gt_mesh'}
    if 'geometry' in metrics:
        required.add('reference_length')
    if set(metrics) & {'pixel', 'rgb'}:
        required |= {'view_id', 'view_type'}
    if 'pixel' in metrics:
        required.add('camera')
    if 'rgb' in metrics:
        required |= {'pred_rgb', 'gt_rgb'}
    seen, frames, sequences, timestamps = set(), {}, {}, {}
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(key) is None for key in headers):
            raise ValueError(f'CSV row {index}: incorrect number of columns')
        row = {k: v.strip() for k, v in row.items()}
        if any(not row.get(key) for key in required):
            raise ValueError(f'CSV row {index}: required values: {sorted(required)}')
        row['timestamp'] = float(row['timestamp'])
        row['reference_length'] = float(row.get('reference_length') or 1)
        if not np.isfinite(row['timestamp']) or not 0 < row['reference_length'] < float('inf'):
            raise ValueError(f'CSV row {index}: invalid time or reference length')
        for key in PATH_COLUMNS:
            row[key] = (path.parent / row[key]).resolve() if row.get(key) else None
            if row[key] is not None and not row[key].is_file():
                raise FileNotFoundError(f'CSV row {index}, {key}: {row[key]}')
        row['view_id'] = row.get('view_id') or 'input'
        row['view_type'] = row.get('view_type') or 'input'
        if row['view_type'] not in ('input', 'novel'):
            raise ValueError('view_type must be input or novel')
        key = (row['sequence_id'], row['frame_id'])
        view_key = (*key, row['view_id'])
        if view_key in seen:
            raise ValueError(f'Duplicate frame/view: {view_key}')
        seen.add(view_key)
        frame = tuple(row.get(k) for k in ('timestamp', 'pred_mesh', 'gt_mesh'))
        if key in frames and frame != frames[key]:
            raise ValueError(f'Conflicting frame metadata across views: {key}')
        frames[key] = frame
        time_key = (row['sequence_id'], row['timestamp'])
        if time_key in timestamps and timestamps[time_key] != row['frame_id']:
            raise ValueError(f'Different frame IDs at the same timestamp: {time_key}')
        timestamps[time_key] = row['frame_id']
        row['pred_matrix'], row['gt_matrix'] = transform(row['pred_transform']), transform(row['gt_transform'])
        signature = (row['reference_length'], row['pred_matrix'].tolist(), row['gt_matrix'].tolist())
        if row['sequence_id'] in sequences and sequences[row['sequence_id']] != signature:
            raise ValueError('Use one fixed reference length and mesh transform pair per sequence')
        sequences[row['sequence_id']] = signature
        if 'pixel' in metrics:
            row['camera_data'] = read_camera(row['camera'])
        rows[index - 2] = row
    return rows


def aggregate(records):
    """Views -> frames -> sequences -> equal-weight dataset mean, with coverage."""
    summary = {}
    for scope in sorted({r['scope'] for r in records}):
        selected = [r for r in records if r['scope'] == scope]
        metrics = sorted({k for r in selected for k in r['metrics']})
        by_sequence = {}
        for sequence in sorted({r['sequence_id'] for r in selected}):
            sr = [r for r in selected if r['sequence_id'] == sequence]
            by_sequence[sequence] = {}
            for metric in metrics:
                by_frame = defaultdict(list)
                for r in sr:
                    if r['metrics'].get(metric) is not None:
                        by_frame[r['frame_id']].append(r['metrics'][metric])
                values = [np.mean(v) for v in by_frame.values()]
                by_sequence[sequence][metric] = {
                    'mean': float(np.mean(values)) if values else None,
                    'valid_frames': len(values),
                    'total_frames': len({r['frame_id'] for r in sr}),
                    'valid_views': sum(len(v) for v in by_frame.values()),
                    'total_views': len(sr),
                }
        macro = {}
        for metric in metrics:
            values = [r[metric]['mean'] for r in by_sequence.values() if r[metric]['mean'] is not None]
            macro[metric] = {'mean': float(np.mean(values)) if values else None,
                             'valid_sequences': len(values), 'total_sequences': len(by_sequence)}
        summary[scope] = {'macro': macro, 'sequences': by_sequence}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True, help='One row per frame/view; paths relative to CSV')
    parser.add_argument('--output', type=Path, required=True, help='A new or empty output directory')
    parser.add_argument('--metrics', nargs='+', choices=('geometry', 'pixel', 'rgb'), default=['geometry', 'pixel'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[.01, .02])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundary-width', type=int, default=3)
    parser.add_argument('--skip-lpips', action='store_true', help='Explicitly omit LPIPS; never silently substitute it')
    parser.add_argument('--save-previews', action='store_true')
    args = parser.parse_args()
    if args.samples < 100 or args.seed < 0 or args.boundary_width < 1 or any(not 0 < t < float('inf') for t in args.thresholds):
        parser.error('Invalid samples, seed, boundary width or F-score thresholds')
    rows = read_manifest(args.csv.resolve(), args.metrics)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty; use a new directory for each comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    progress = {'status': 'running', 'completed_rows': 0, 'total_rows': len(rows)}
    write_json(args.output / 'progress.json', progress)
    try:
        run(args, rows, progress)
    except Exception as error:
        progress.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(args.output / 'progress.json', progress)
        raise


def run(args, rows, progress):
    renderer = images = None
    if set(args.metrics) & {'pixel', 'rgb'}:
        from .pixel import ImageMetrics, NormalRenderer, normal_metrics, read_rgb
        if 'pixel' in args.metrics:
            renderer = NormalRenderer(args.device)
        images = ImageMetrics(args.device, not args.skip_lpips)
    records, done_geometry, fingerprints = [], set(), {}
    previous_key, pair = None, None
    for index, row in enumerate(rows):
        key = (row['sequence_id'], row['frame_id'])
        metadata = {k: row[k] for k in ('sequence_id', 'frame_id', 'timestamp')}
        for column in PATH_COLUMNS:
            path = row[column]
            if path is not None and str(path) not in fingerprints:
                fingerprints[str(path)] = sha256(path)
        if set(args.metrics) & {'geometry', 'pixel'} and key != previous_key:
            pair = [load_mesh(row[f'{kind}_mesh'], row[f'{kind}_matrix']) for kind in ('pred', 'gt')]
            previous_key = key
        if 'geometry' in args.metrics and key not in done_geometry:
            seed = (args.seed + int(hashlib.sha256('/'.join(key).encode()).hexdigest()[:8], 16)) % (2 ** 32 - 1)
            scores = surface_metrics(*pair, row['reference_length'], args.samples, args.thresholds, seed)
            records.append({**metadata, 'scope': 'geometry', 'view_id': '', 'metrics': scores})
            done_geometry.add(key)
        scores = {}
        if 'pixel' in args.metrics:
            pred, pmask = renderer.render(pair[0], row['camera_data'])
            truth, gmask = renderer.render(pair[1], row['camera_data'])
            if not gmask.any():
                raise ValueError(f'Ground truth is outside the camera in CSV row {index + 2}')
            scores.update(normal_metrics(pred, truth, pmask, gmask, images, args.boundary_width))
            if args.save_previews:
                from PIL import Image
                folder = args.output / 'previews'
                folder.mkdir(exist_ok=True)
                p, g = (pred + 1) / 2, (truth + 1) / 2
                p[~pmask] = g[~gmask] = 1
                Image.fromarray((np.concatenate([g, p], 1).clip(0, 1) * 255).astype('uint8')).save(folder / f'{index:06d}.png')
        if 'rgb' in args.metrics:
            pred_rgb, gt_rgb = read_rgb(row['pred_rgb']), read_rgb(row['gt_rgb'])
            if 'pixel' in args.metrics and gt_rgb.shape[:2] != (row['camera_data']['height'], row['camera_data']['width']):
                raise ValueError('RGB images must match the declared camera canvas')
            scores.update({f'rgb_{k}': v for k, v in images.compare(pred_rgb, gt_rgb, np.ones(gt_rgb.shape[:2], bool)).items()})
        if scores:
            records.append({**metadata, 'scope': row['view_type'], 'view_id': row['view_id'], 'metrics': scores})
        progress['completed_rows'] = index + 1
        write_json(args.output / 'progress.json', progress)
        print(f'[{index + 1}/{len(rows)}] {key[0]}/{key[1]} {row["view_id"]}', flush=True)
    flattened = [{**{k: v for k, v in r.items() if k != 'metrics'}, **r['metrics']} for r in records]
    headers = list(dict.fromkeys(k for r in flattened for k in r))
    with (args.output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flattened)
    import importlib.metadata
    versions = {}
    for name in ('numpy', 'scipy', 'trimesh', 'scikit-image', 'torch', 'lpips', 'nvdiffrast'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    implementation = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    implementation['evaluate.py'] = sha256(Path(__file__).resolve().parents[1] / 'evaluate.py')
    summary = {'protocol': 'pixal4d-eval-v1', 'status': 'complete',
               'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'manifest_sha256': sha256(args.csv), 'input_sha256': fingerprints,
               'implementation_sha256': implementation,
               'versions': versions, 'results': aggregate(records),
               'notes': ['No fitted registration or per-frame normalization.',
                         'Normal metrics follow Pixal3D metric families, not an exact paper-table reproduction.',
                         'Missing overlap/window metrics are null; inspect coverage, never treat them as zero error.',
                         'RGB pairs are externally rendered on a matching canvas/lighting; RGBA uses white background.',
                         'These are per-frame metrics, not a temporal consistency measure.']}
    write_json(args.output / 'summary.json', summary)
    progress['status'] = 'complete'
    write_json(args.output / 'progress.json', progress)
    print(f'Complete: {args.output / "summary.json"}', flush=True)
