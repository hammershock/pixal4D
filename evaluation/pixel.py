"""Explicit normal-alignment protocol and optional pre-rendered RGB metrics."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


def read_rgb(path):
    image = np.asarray(Image.open(path).convert('RGBA'), dtype=np.float32) / 255
    return image[..., :3] * image[..., 3:] + 1 - image[..., 3:]


class ImageMetrics:
    def __init__(self, device, use_lpips):
        self.device = device
        self.network = None
        if use_lpips:
            import lpips
            self.network = lpips.LPIPS(net='alex', version='0.1', spatial=True).to(device).eval()

    def compare(self, pred, truth, mask):
        if pred.shape != truth.shape or pred.shape[:2] != mask.shape:
            raise ValueError('Images and masks must share the original pixel grid')
        if min(mask.shape) < 32:
            raise ValueError('Image metrics require width and height >= 32')
        if not mask.any():
            return {'psnr': None, 'ssim': None, **({'lpips': None} if self.network else {})}
        mse = float(np.square(pred[mask] - truth[mask]).mean())
        # A finite ceiling permits strict JSON and reproducible macro aggregation.
        result = {'psnr': float(-10 * np.log10(max(mse, 1e-12)))}
        a, b = pred.copy(), truth.copy()
        a[~mask] = b[~mask] = .5
        _, ssim = structural_similarity(a, b, channel_axis=-1, data_range=1,
                                        gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False, full=True)
        # No SSIM window is allowed to gain credit from padded background.
        interior = binary_erosion(mask, structure=np.ones((11, 11)))
        result['ssim'] = float(ssim[interior].mean()) if interior.any() else None
        if self.network is not None:
            import torch
            def tensor(x):
                return torch.from_numpy(x.copy()).permute(2, 0, 1)[None].to(self.device) * 2 - 1
            with torch.inference_mode():
                distance = self.network(tensor(a), tensor(b))[0, 0].cpu().numpy()
            result['lpips'] = float(distance[mask].mean())
        return result


def normal_metrics(pred, truth, pred_mask, gt_mask, images, boundary_width):
    intersection, union = pred_mask & gt_mask, pred_mask | gt_mask
    result = {
        'mask_iou': float(intersection.sum() / union.sum()) if union.any() else None,
        'gt_coverage': float(intersection.sum() / gt_mask.sum()) if gt_mask.any() else None,
    }
    encoded_p, encoded_g = (pred + 1) / 2, (truth + 1) / 2
    result.update({f'normal_{k}': v for k, v in images.compare(encoded_p, encoded_g, intersection).items()})
    keys = ['normal_mean_deg', 'normal_median_deg', 'normal_boundary_mean_deg',
            'normal_acc_11.25', 'normal_acc_22.5', 'normal_acc_30']
    if not intersection.any():
        result.update(dict.fromkeys(keys))
        return result
    # Normalize in float64 so identical float32 normals do not acquire a
    # spurious angular error from a norm slightly smaller than one.
    pn, gn = pred.astype(np.float64), truth.astype(np.float64)
    denominator = np.linalg.norm(pn, axis=-1) * np.linalg.norm(gn, axis=-1)
    dot = np.clip(np.einsum('ijk,ijk->ij', pn, gn) / np.maximum(denominator, 1e-15), -1, 1)
    angle = np.degrees(np.arccos(dot))
    boundary = gt_mask & ~binary_erosion(gt_mask, iterations=boundary_width)
    boundary &= intersection
    result.update(normal_mean_deg=float(angle[intersection].mean()),
                  normal_median_deg=float(np.median(angle[intersection])),
                  normal_boundary_mean_deg=float(angle[boundary].mean()) if boundary.any() else None)
    for threshold in (11.25, 22.5, 30):
        result[f'normal_acc_{threshold:g}'] = float((angle[intersection] < threshold).mean())
    return result


class NormalRenderer:
    """Flat, face-forward camera normals at pixel centers (no antialiasing)."""
    def __init__(self, device):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr, self.device = torch, dr, device
        if torch.device(device).type != 'cuda':
            raise ValueError('Normal rendering requires CUDA; geometry/RGB evaluation also supports CPU')
        self.context = dr.RasterizeCudaContext(device=device)

    def render(self, mesh, camera):
        torch, dr = self.torch, self.dr
        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(np.asarray(x).copy(), dtype=dtype, device=self.device)
        height, width = camera['height'], camera['width']
        k, ext = tensor(camera['K']), tensor(camera['world_to_camera'])
        near, far = camera.get('near', .001), camera.get('far', 100.)
        vertices, faces = tensor(mesh.vertices), tensor(mesh.faces, torch.int32)
        vertices = vertices @ ext[:3, :3].T + ext[:3, 3]
        projected = vertices @ k.T
        z = vertices[:, 2]
        clip = torch.stack((2 * projected[:, 0] / width - z,
                            z - 2 * projected[:, 1] / height,
                            (far + near) / (far - near) * z - 2 * far * near / (far - near), z), -1)
        rast, _ = dr.rasterize(self.context, clip[None], faces, (height, width))
        triangles = vertices[faces.long()]
        normals = torch.nn.functional.normalize(torch.cross(triangles[:, 1] - triangles[:, 0],
                                                            triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
        normals = torch.where((normals * triangles.mean(1)).sum(-1, keepdim=True) > 0, -normals, normals)
        ids = torch.arange(len(faces), dtype=torch.int32, device=self.device)[:, None].expand(-1, 3).contiguous()
        pixels = dr.interpolate(normals[None], rast, ids)[0][0]
        mask = rast[0, ..., 3] > 0
        return pixels.flip(0).cpu().numpy(), mask.flip(0).cpu().numpy()
