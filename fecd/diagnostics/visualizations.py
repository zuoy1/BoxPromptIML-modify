"""PIL-only raster diagnostics, with a strict output boundary for GT panels."""
from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIAGNOSTIC_ROOT = Path(__file__).resolve().parents[1] / 'assets/stage3_diagnostics'
_PALETTE = np.asarray([[25, 20, 80], [35, 90, 145], [30, 155, 140],
                       [145, 205, 70], [250, 230, 40]], dtype=np.float64)


def _font(size=12):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _output_path(path, contains_gt):
    path = Path(path).resolve()
    if contains_gt and not path.is_relative_to(DIAGNOSTIC_ROOT.resolve()):
        raise ValueError(f'GT diagnostics must be under {DIAGNOSTIC_ROOT}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _color_map(name, value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (224, 224) or not np.isfinite(value).all():
        raise ValueError(f'Expected finite 224x224 diagnostic map: {name}')
    observed_min, observed_max = float(value.min()), float(value.max())
    diverging = observed_min < 0
    if diverging:
        ceiling = max(abs(observed_min), abs(observed_max), 1.)
        normalized = np.clip(value / ceiling, -1, 1)
        white = np.full(value.shape + (3,), 245., dtype=np.float64)
        positive = np.asarray([210., 40., 40.])
        negative = np.asarray([35., 90., 220.])
        color = np.where((normalized >= 0)[..., None], positive, negative)
        rgb = white + np.abs(normalized)[..., None] * (color - white)
        scale = f'scale [{-ceiling:g},{ceiling:g}] blue-/red+'
    else:
        lower = 0.
        upper = .25 if 'var' in name.lower() and observed_max <= .2501 else 1.
        if observed_max > upper:
            upper = observed_max
        normalized = np.clip((value - lower) / max(upper - lower, 1e-12), 0, 1)
        coordinates = normalized * (len(_PALETTE) - 1)
        index = np.minimum(coordinates.astype(np.int64), len(_PALETTE) - 2)
        fraction = coordinates - index
        rgb = _PALETTE[index] * (1 - fraction[..., None]) + _PALETTE[index + 1] * fraction[..., None]
        scale = f'scale [{lower:g},{upper:g}]'
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), (
        f'{name}\n{scale}\nobserved [{observed_min:.4g},{observed_max:.4g}]')


def save_diagnostic(path, rgb224, boxes224, maps, caption, contains_gt=True):
    """Save image+boxes and named maps in a labelled grid with <=4 columns.

    Display scales never alter the cached values. Probability maps use [0,1],
    variance [0,.25], and signed selection differences symmetric blue/red.
    Raw maps exceeding those ranges explicitly display their actual upper bound.
    """
    if not contains_gt and any(re.search(r'(^|[^a-z])gt([^a-z]|$)|ground.?truth',
                                        name.lower()) for name in maps):
        raise ValueError('A non-GT visualization cannot include a named GT map')
    path = _output_path(path, contains_gt)
    rgb = np.asarray(rgb224)
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError('rgb224 must be uint8 RGB with shape (224,224,3)')
    original = Image.fromarray(rgb.copy())
    draw_image = ImageDraw.Draw(original)
    for box in np.asarray(boxes224, dtype=np.float64).reshape(-1, 4):
        if not np.isfinite(box).all():
            raise ValueError('Box coordinates must be finite')
        draw_image.rectangle(tuple(box), outline=(255, 40, 40), width=2)
    panels = [(original, 'RGB + frozen boxes (224 grid)')]
    panels.extend(_color_map(name, value) for name, value in maps.items())
    columns = min(4, len(panels))
    tile_w, tile_h, margin, top = 244, 288, 12, 72
    canvas = Image.new('RGB', (columns * tile_w + margin,
                              math.ceil(len(panels) / columns) * tile_h + top), 'white')
    draw = ImageDraw.Draw(canvas)
    font = _font()
    heading = ('DIAGNOSTIC ONLY | GT included | ' if contains_gt else 'GT-free inspection | ') + str(caption)
    draw.multiline_text((margin, 8), '\n'.join(textwrap.wrap(heading, columns * 32)[:4]),
                        fill='black', font=font, spacing=3)
    for index, (panel, label) in enumerate(panels):
        left = margin + (index % columns) * tile_w
        upper = top + (index // columns) * tile_h
        canvas.paste(panel, (left, upper))
        draw.multiline_text((left, upper + 228), label, fill='black', font=font, spacing=2)
    canvas.save(path)
    return str(path)


def save_risk_curves(path, curves):
    """Save an 800x600 raster with coverage/risk axes fixed to [0,1]."""
    path = _output_path(path, contains_gt=False)
    canvas = Image.new('RGB', (800, 600), 'white')
    draw, font = ImageDraw.Draw(canvas), _font(13)
    left, right, top, bottom = 70, 770, 45, 490
    for tick in np.linspace(0, 1, 6):
        x = left + tick * (right - left)
        y = bottom - tick * (bottom - top)
        draw.line((x, top, x, bottom), fill=(225, 225, 225))
        draw.line((left, y, right, y), fill=(225, 225, 225))
        draw.text((x - 10, bottom + 7), f'{tick:.1f}', fill='black', font=font)
        draw.text((28, y - 6), f'{tick:.1f}', fill='black', font=font)
    draw.line((left, top, left, bottom, right, bottom), fill='black', width=2)
    draw.text((left, 12), 'Risk-Coverage | equal conditional candidate coverage', fill='black', font=font)
    draw.text((350, bottom + 28), 'Coverage', fill='black', font=font)
    draw.text((5, 20), 'Risk', fill='black', font=font)
    colors = [(25, 70, 180), (210, 60, 35), (25, 150, 70), (140, 50, 180),
              (200, 140, 15), (15, 160, 180), (100, 100, 100)]
    for index, (name, curve) in enumerate(curves.items()):
        coverage = np.asarray(curve['coverage'], dtype=np.float64)
        risk = np.asarray(curve['risk'], dtype=np.float64)
        if (coverage.shape != risk.shape or coverage.ndim != 1
                or not np.isfinite(coverage).all() or not np.isfinite(risk).all()
                or np.any(coverage < 0) or np.any(coverage > 1)
                or np.any(risk < 0) or np.any(risk > 1)):
            raise ValueError(f'Invalid risk curve: {name}')
        order = np.argsort(coverage, kind='stable')
        points = [(left + float(c) * (right - left), bottom - float(r) * (bottom - top))
                  for c, r in zip(coverage[order], risk[order])]
        color = colors[index % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = left + (index % 4) * 175
        legend_y = 552 + (index // 4) * 20
        draw.line((legend_x, legend_y + 6, legend_x + 18, legend_y + 6), fill=color, width=3)
        draw.text((legend_x + 23, legend_y), str(name)[:20], fill='black', font=font)
    canvas.save(path)
    return str(path)
