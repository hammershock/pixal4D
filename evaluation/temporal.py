"""Fixed-camera change errors against GT; not a 3D correspondence metric."""
import numpy as np


def normal_state(pred, truth, pred_mask, gt_mask, timestamp):
    """Retain normal and silhouette residuals on the original pixel grid."""
    if pred.shape != truth.shape or pred.shape != (*pred_mask.shape, 3) or pred_mask.shape != gt_mask.shape:
        raise ValueError('Temporal normal maps and masks must share a canvas')
    if not np.isfinite(timestamp) or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Temporal inputs must be finite')
    p, g = (pred + 1) / 2, (truth + 1) / 2
    p = np.where(pred_mask[..., None], p, 1.)
    g = np.where(gt_mask[..., None], g, 1.)
    return {'normal_error': p - g,
            'mask_error': pred_mask.astype(np.float32) - gt_mask.astype(np.float32),
            'foreground': pred_mask | gt_mask, 'timestamp': float(timestamp)}


def change_metrics(previous, current):
    """RMSE of (predicted change - GT change) / elapsed seconds.

    A constant reconstruction error can score zero: always accompany these
    diagnostics with spatial accuracy. No optical flow or fitted alignment is
    used. Normal changes can reflect tessellation as well as surface motion.
    """
    dt = current['timestamp'] - previous['timestamp']
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('Temporal observations must have increasing timestamps in seconds')
    if previous['foreground'].shape != current['foreground'].shape:
        raise ValueError('Temporal canvases must be identical')
    mask = previous['foreground'] | current['foreground']
    if not mask.any():
        raise ValueError('Temporal foreground is empty')
    values = {}
    for name, key in [('normal_change_rmse_per_second', 'normal_error'),
                      ('silhouette_change_rmse_per_second', 'mask_error')]:
        delta = (current[key].astype(np.float64) - previous[key]) / dt
        values[name] = float(np.sqrt(np.square(delta[mask]).mean()))
    return values
