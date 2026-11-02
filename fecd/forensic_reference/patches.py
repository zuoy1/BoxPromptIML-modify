"""RGB-only patch descriptors and explicit native/work/SAM coordinate mapping."""
from __future__ import annotations

import cv2
import numpy as np

from fecd.confidence.teacher_scores import sam_confidence


def _rgb_float(rgb, color_order):
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError('RGB input must be nonempty HWC with three channels')
    if color_order not in ('RGB', 'BGR'):
        raise ValueError('color_order must be explicitly RGB or BGR')
    if value.dtype == np.uint8:
        value = value.astype(np.float32) / 255.
    elif np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Float input must be finite and in [0,1]')
        value = value.astype(np.float32)
    else:
        raise ValueError('Only uint8 or floating [0,1] RGB input is accepted')
    if color_order == 'BGR':
        value = value[:, :, ::-1]
    return np.ascontiguousarray(value)


def _sam_arrays(mu, var, agreement, box_masks):
    arrays = [np.asarray(value, dtype=np.float32) for value in (mu, var, agreement)]
    if any(value.shape != (224, 224) for value in arrays):
        raise ValueError('SAM arrays must retain the frozen 224x224 grid')
    score = sam_confidence(*arrays)
    masks = np.asarray(box_masks)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3 or masks.shape[1:] != (224, 224):
        raise ValueError('Box masks must have shape [N,224,224]')
    if not np.isfinite(masks).all() or not np.isin(masks, (0, 1)).all():
        raise ValueError('Box masks must be finite and binary')
    union = np.any(masks, axis=0).astype(np.uint8)
    stable_negative = (arrays[0] < .5) & (score >= .9) & (arrays[1] <= .01)
    return arrays[0], stable_negative.astype(np.uint8), union


def _nearest_work(value, height, width):
    # Pixel centers, not corner-aligned resize indices, define every SAM lookup.
    xx, yy = np.meshgrid((np.arange(width, dtype=np.float32) + .5) * 224 / width - .5,
                         (np.arange(height, dtype=np.float32) + .5) * 224 / height - .5)
    return cv2.remap(value, xx, yy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def _positions(length, size):
    positions = list(range(0, length - size + 1, size // 2))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


def _records(gray, gradient, texture, residual, prediction, stable, box, dilated,
             size, native_hw):
    height, width = gray.shape
    native_h, native_w = native_hw
    records = []
    for y in _positions(height, size):
        for x in _positions(width, size):
            region = np.s_[y:y + size, x:x + size]
            center = np.s_[y + 2:y + size - 2, x + 2:x + size - 2]
            values = residual[center]
            median = np.median(values)
            bounds = [x, y, x + size, y + size]
            records.append({
                'id': f'{size}:{y}:{x}', 'index': len(records),
                'x0': x, 'y0': y, 'x1': x + size, 'y1': y + size,
                'bounds_work': bounds,
                'bounds_native': [bounds[i] * (native_w / width if i % 2 == 0 else native_h / height)
                                  for i in range(4)],
                'bounds_sam224': [bounds[i] * 224 / (width if i % 2 == 0 else height)
                                  for i in range(4)],
                'brightness': float(gray[region].mean()),
                'gradient': float(gradient[region].mean()),
                'texture': float(texture[region].std()),
                'mad': float(np.median(np.abs(values - median))),
                'quantiles': np.quantile(values, np.arange(1, 20) / 20).tolist(),
                'pred_fraction': float(prediction[region].mean()),
                'stable_negative_fraction': float(stable[region].mean()),
                'box_fraction': float(box[region].mean()),
                'outside_dilated_fraction': float(1. - dilated[region].mean()),
            })
    return records


def extract_patches(rgb, mu, var, agreement, box_masks, patch_size=64,
                    color_order='RGB', box_dilation=5):
    """Describe in-bounds patches without labels, normalization fitting or scores.

    Native float RGB is sampled onto an aspect-preserving, longest-edge<=512
    grid. Patch bounds map through continuous half-open image coordinates;
    SAM fields use nearest pixel-center lookups. Filtering uses the full work
    image, and residual summaries omit each patch's outer two pixels.
    """
    if patch_size not in (32, 64):
        raise ValueError('Only the frozen primary 64 and sensitivity 32 scales are accepted')
    if not isinstance(box_dilation, int) or box_dilation < 0:
        raise ValueError('box_dilation must be a nonnegative integer SAM-grid radius')
    native = _rgb_float(rgb, color_order)
    mu, stable, box = _sam_arrays(mu, var, agreement, box_masks)
    native_h, native_w = native.shape[:2]
    scale = min(1., 512. / max(native_h, native_w))
    height, width = max(1, round(native_h * scale)), max(1, round(native_w * scale))
    work = cv2.resize(native, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
    residual = gray - cv2.medianBlur(gray, 3)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1., borderType=cv2.BORDER_REFLECT_101)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_REFLECT_101)
    gradient = np.hypot(dx, dy)
    texture = gray - cv2.GaussianBlur(gray, (0, 0), 2., borderType=cv2.BORDER_REFLECT_101)
    kernel = np.ones((2 * box_dilation + 1, 2 * box_dilation + 1), np.uint8)
    dilated = cv2.dilate(box, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    projected = [_nearest_work(value, height, width) for value in
                 ((mu >= .5).astype(np.uint8), stable, box, dilated)]
    records = [] if min(height, width) < patch_size else _records(
        gray, gradient, texture, residual, *projected, patch_size, (native_h, native_w))
    return {
        'native_hw': [native_h, native_w], 'work_hw': [height, width],
        'scale_xy': [width / native_w, height / native_h], 'patch_size': patch_size,
        'patches': records,
        'geometry': {
            'bounds': 'xyxy half-open; native/work/SAM224 boundaries use exact axis ratios',
            'sam_sampling': 'nearest at (work_index+0.5)*224/work_length-0.5; replicate border',
            'rgb_sampling': 'OpenCV INTER_AREA; aspect-preserving rounding; no upsampling',
            'filter_border': 'median3 replicate; Gaussian/Sobel REFLECT_101; patch center excludes2',
        },
    }
