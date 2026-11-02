"""Deterministic image descriptors of local inconsistency; no GT or probability claims."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


EXTRACTOR_CONFIG = {
    'version': 'fecd-evidence-v1',
    'input': 'decoded RGB uint8 or float [0,1]; no EXIF rotation',
    'work_long_edge': 512,
    'work_resize': 'shrink only; cv2 INTER_AREA; preserve aspect ratio, rounded dimensions',
    'output': '224x224 float32; cv2 INTER_LINEAR, explicit native-to-square SAM mapping',
    'boundary': 'BORDER_REFLECT_101; FFT symmetric padding for singleton dimensions',
    'gray': 'RGB BT.601 cv2 RGB2GRAY on float [0,1]',
    'hp': {'gaussian_sigma': 1.0, 'pool_kernel': 9, 'background_sigma': 16.0,
           'definition': 'absolute log contrast of pooled absolute Gaussian high-pass residual'},
    'noise': {'median_kernel': 3, 'pool_kernel': 9, 'background_sigma': 16.0,
              'residual_clip': 0.1, 'edge_compensation': 'divide by 1+8*smoothed-gradient',
              'definition': 'absolute log contrast of pooled clipped median residual'},
    'jpeg': {'quality': 90, 'subsampling': 0, 'optimize': False, 'progressive': False,
             'native_recompression': True, 'pool_kernel': 9, 'background_sigma': 16.0,
             'definition': 'native decoded RGB minus fixed JPEG; local log residual contrast'},
    'fft': {'window': 32, 'stride': 8, 'window_function': 'separable symmetric Hann',
            'high_band': 'radial frequency >= 0.25 cycles/work-pixel',
            'background_sigma_tiles': 4.0, 'epsilon_power': 1e-12,
            'definition': 'absolute difference of high/total AC power from local Gaussian mean'},
    'log_epsilon': 1.0 / 255.0,
    'direction': 'larger means stronger defined local inconsistency; not tampering probability',
    'fusion': 'median(normalized noise, jpeg, fft); HP excluded',
}

_BORDER = cv2.BORDER_REFLECT_101
_NAMES = ('hp', 'noise', 'jpeg', 'fft')


def _rgb_uint8(image, color_order):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError('Evidence input must be nonempty [H,W,3]')
    if color_order not in ('RGB', 'BGR'):
        raise ValueError('color_order must be RGB or BGR')
    if value.dtype == np.uint8:
        value = value.copy()
    elif np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Float RGB must be finite and in [0,1]')
        value = np.rint(value * 255).astype(np.uint8)
    else:
        raise ValueError('Only uint8 or floating [0,1] RGB is supported')
    if color_order == 'BGR':
        value = value[:, :, ::-1]
    return np.ascontiguousarray(value)


def _blur(value, sigma):
    return cv2.GaussianBlur(value, (0, 0), sigma, borderType=_BORDER)


def _log_anomaly(residual):
    pooled = cv2.boxFilter(residual, -1, (9, 9), borderType=_BORDER)
    logged = np.log(pooled + EXTRACTOR_CONFIG['log_epsilon'])
    return np.abs(logged - _blur(logged, 16.0))


def _fft_anomaly(gray):
    window, stride = 32, 8
    height, width = gray.shape
    # Padding centers tiles on work-grid positions 0,8,...; interpolation below
    # uses those physical centers, avoiding a half-window coordinate shift.
    padded = np.pad(gray, ((16, 16 + (-height) % stride),
                           (16, 16 + (-width) % stride)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(padded, (window, window))[::stride, ::stride]
    patches = patches - patches.mean(axis=(-2, -1), keepdims=True)
    hann = np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)
    power = np.abs(np.fft.rfft2(patches * hann, axes=(-2, -1))) ** 2
    frequency_y = np.fft.fftfreq(window)[:, None]
    frequency_x = np.fft.rfftfreq(window)[None, :]
    high = np.hypot(frequency_y, frequency_x) >= .25
    # Real FFT retains only half the horizontal spectrum; restore pair weights.
    weights = np.full(window // 2 + 1, 2.0)
    weights[[0, -1]] = 1.0
    total = (power * weights).sum(axis=(-2, -1))
    ratio = ((power * high * weights).sum(axis=(-2, -1)) / (total + 1e-12)).astype(np.float32)
    anomaly = np.abs(ratio - _blur(ratio, 4.0))
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32) / stride,
                         np.arange(height, dtype=np.float32) / stride)
    return cv2.remap(anomaly, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def extract_evidence(rgb, color_order='RGB'):
    """Return four nonnegative finite float32 maps on the frozen 224 SAM grid.

    Filtering uses an aspect-preserving grid with longest edge at most 512.
    JPEG is recompressed before resampling, retaining native codec-block meaning.
    The final square mapping explicitly matches the preexisting student/SAM grid.
    These descriptors can respond to genuine texture, edges and codec history.
    """
    native = _rgb_uint8(rgb, color_order)
    height, width = native.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    work_wh = (max(1, round(width * scale)), max(1, round(height * scale)))
    work = cv2.resize(native, work_wh, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY)
    hp = _log_anomaly(np.abs(gray - _blur(gray, 1.0)))

    residual = np.minimum(np.abs(gray - cv2.medianBlur(gray, 3)), .1)
    smooth = _blur(gray, 1.0)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8, borderType=_BORDER)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8, borderType=_BORDER)
    noise = _log_anomaly(residual / (1 + 8 * _blur(np.hypot(dx, dy), 1.0)))

    buffer = io.BytesIO()
    Image.fromarray(native).save(buffer, format='JPEG', quality=90, subsampling=0,
                                 optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        reconstructed = np.asarray(compressed.convert('RGB'), dtype=np.float32)
    jpeg_residual = np.abs(native.astype(np.float32) - reconstructed).mean(axis=2) / 255.0
    jpeg = _log_anomaly(cv2.resize(jpeg_residual, work_wh, interpolation=cv2.INTER_AREA))
    maps = dict(zip(_NAMES, (hp, noise, jpeg, _fft_anomaly(gray))))
    result = {}
    for name, value in maps.items():
        resized = cv2.resize(value, (224, 224), interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(resized).all():
            raise ValueError(f'Non-finite {name} descriptor')
        result[name] = np.ascontiguousarray(np.maximum(resized, 0), dtype=np.float32)
    return result
