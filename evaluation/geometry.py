"""Surface metrics in a shared coordinate system; never fit registration."""
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path, transform=None):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            mesh = trimesh.Trimesh(data['vertices'], data['faces'], process=False)
    else:
        scene = trimesh.load_scene(path, process=False)
        if any(not isinstance(g, trimesh.Trimesh) for g in scene.geometry.values()):
            raise ValueError(f'Only triangle meshes are supported: {path}')
        mesh = scene.to_mesh()  # Includes every instance and its scene transform.
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Empty or non-finite mesh: {path}')
    if transform is not None:
        mesh.apply_transform(transform)
    valid = np.isfinite(mesh.area_faces) & (mesh.area_faces > 0)
    mesh.update_faces(valid)
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError(f'Mesh has no surface area: {path}')
    return mesh


def surface_metrics(pred, truth, reference_length, samples, thresholds, seed):
    """Area-uniform samples; CD-L1 means Euclidean distance, not Manhattan."""
    p, pi = trimesh.sample.sample_surface(pred, samples, seed=seed)
    g, gi = trimesh.sample.sample_surface(truth, samples, seed=seed + 1)
    dp, ip = cKDTree(g).query(p, workers=1)
    dg, ig = cKDTree(p).query(g, workers=1)
    dp, dg = dp / reference_length, dg / reference_length
    pn, gn = pred.face_normals[pi], truth.face_normals[gi]
    nc_p = np.abs(np.einsum('ij,ij->i', pn, gn[ip]))
    nc_g = np.abs(np.einsum('ij,ij->i', gn, pn[ig]))
    result = {
        'cd_l1': float((dp.mean() + dg.mean()) / 2),
        'cd_l2': float(((dp ** 2).mean() + (dg ** 2).mean()) / 2),
        'normal_consistency': float((nc_p.mean() + nc_g.mean()) / 2),
    }
    for threshold in thresholds:
        precision, recall = float((dp < threshold).mean()), float((dg < threshold).mean())
        result[f'fscore@{threshold:g}'] = 2 * precision * recall / max(precision + recall, 1e-15)
    return result
