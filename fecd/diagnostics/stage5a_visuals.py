"""固定开发案例图；只读概率副本和已核对的开发RGB。"""
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fecd.diagnostics.stage5a_io import OUTPUT
from fecd.diagnostics.stage5a_metrics import measures
from fecd.tools.stage4f1_v2_io import write_new


def _overlay(rgb, pred, gt):
    colored = np.zeros_like(rgb)
    colored[pred & gt] = (20, 190, 70)
    colored[pred & ~gt] = (240, 50, 40)
    colored[~pred & gt] = (40, 90, 245)
    active = pred | gt
    output = rgb.copy()
    output[active] = (.5 * rgb[active] + .5 * colored[active]).astype(np.uint8)
    return output


def _probability(probability):
    color = np.empty((*probability.shape, 3), np.uint8)
    color[..., 0] = (probability * 255).astype(np.uint8)
    color[..., 1] = (np.minimum(probability, 1 - probability) * 300).astype(np.uint8)
    color[..., 2] = ((1 - probability) * 255).astype(np.uint8)
    return color


def _threshold_plot():
    rows = json.loads((OUTPUT / 'threshold_sensitivity.json').read_text())
    colors = {'P0': '#1764ab', 'P1': '#d45b16', 'P2F': '#16814a', 'P3F': '#9245ad'}
    panels = [('ALL', key) for key in ('image_f1', 'image_precision', 'image_recall', 'image_prediction_area')]
    panels.extend((d, 'image_f1') for d in ('CASIAv2', 'NIST16', 'Columbia', 'Coverage'))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="1250">',
           '<rect width="1050" height="1250" fill="white"/>',
           '<g font-family="sans-serif" font-size="12">',
           '<text x="45" y="25" font-size="18">Stage5A new_dev diagnostic threshold grid: official threshold remains 0.5</text>']
    for i, (domain, metric) in enumerate(panels):
        left, top = 55 + (i % 2) * 520, 80 + (i // 2) * 280
        width, height = 410, 190
        svg.append(f'<text x="{left}" y="{top-15}">{domain}: {metric}</text>')
        for tick in range(6):
            y = top + height - tick / 5 * height
            svg.append(f'<path d="M{left} {y}h{width}" stroke="#ddd"/><text x="{left-25}" y="{y+4}">{tick/5:.1f}</text>')
        for threshold in (.1, .3, .5, .7, .9):
            x = left + (threshold - .1) / .8 * width
            svg.append(f'<text x="{x-8}" y="{top+height+20}">{threshold}</text>')
        for model, color in colors.items():
            selected = sorted([r for r in rows if r['model'] == model and r['dataset'] == domain],
                              key=lambda r: r['threshold'])
            points = ' '.join(f'{left+(r["threshold"]-.1)/.8*width:.2f},{top+height-r[metric]*height:.2f}' for r in selected)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
    for i, (model, color) in enumerate(colors.items()):
        svg.append(f'<text x="{140+i*210}" y="1220" fill="{color}">{model}</text>')
    svg.append('</g></svg>')
    with (OUTPUT / 'threshold_curves.svg').open('x') as stream:
        stream.write('\n'.join(svg))


def main():
    output = OUTPUT / 'cases'
    output.mkdir(exist_ok=False)
    allow = json.loads((OUTPUT / 'allowlist.json').read_text())
    members = {r['sample_key']: r for r in allow['members']}
    selected = json.loads((OUTPUT / 'case_selection.json').read_text())
    categories = defaultdict(list)
    for case in selected:
        categories[case['sample_key']].append(case['category'])
    manifest = []
    with (OUTPUT / 'logs/visual_rgb_access.jsonl').open('x') as access:
        for index, (key, labels) in enumerate(sorted(categories.items())):
            row = members[key]
            content = Path(row['image_path']).read_bytes()
            import hashlib
            if hashlib.sha256(content).hexdigest() != row['image_sha256']:
                raise ValueError('案例RGB身份错误')
            access.write(json.dumps({'sample_key': key, 'role': 'new_dev_RGB_only',
                                     'path': row['image_path']}) + '\n')
            with Image.open(BytesIO(content)) as image:
                rgb = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
            with np.load(OUTPUT / 'maps' / f'{key}_reference.npz', allow_pickle=False) as ref:
                gt, k8, k1 = ref['gt'].copy(), ref['K8'].copy(), ref['K1'].copy()
            maps = {}
            for name in ('P0', 'P1', 'P2F', 'P3F'):
                with np.load(OUTPUT / 'maps' / f'{key}_{name}.npz', allow_pickle=False) as data:
                    maps[name] = data['probability'].copy()
            mask_rgb = lambda mask: np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
            panels = [('RGB (new_dev)', rgb), ('GT (diagnostic)', mask_rgb(gt)),
                      ('K8 >= 0.5', mask_rgb(k8 >= .5)), ('K8 FP/FN', _overlay(rgb, k8 >= .5, gt)),
                      ('P3F probability', _probability(maps['P3F']))]
            for name in maps:
                score = measures(maps[name] >= .5, gt)
                panels.append((f'{name} mask F1={score["f1"]:.3f}', mask_rgb(maps[name] >= .5)))
            panels.append(('P3F overlay', _overlay(rgb, maps['P3F'] >= .5, gt)))
            for name in ('P0', 'P1', 'P2F'):
                panels.append((f'{name} overlay', _overlay(rgb, maps[name] >= .5, gt)))
            panels.append(('K1 >= 0.5', mask_rgb(k1 >= .5)))
            canvas = Image.new('RGB', (5 * 232, 70 + 3 * 256), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'NEW_DEV DIAGNOSIS | {row["dataset"]} | {key[:16]}', fill='black')
            draw.text((8, 23), 'Categories: ' + ', '.join(labels), fill='black')
            draw.text((8, 40), 'Overlay: TP green / FP red / FN blue. All masks use 0.5. Not independent confirmation.', fill='black')
            for i, (title, array) in enumerate(panels):
                x, y = (i % 5) * 232 + 4, 70 + (i // 5) * 256
                draw.text((x, y), title, fill='black')
                canvas.paste(Image.fromarray(array), (x, y + 22))
            filename = f'{index + 1:02d}_{row["dataset"]}_{key[:12]}.jpg'
            canvas.save(output / filename, quality=92)
            manifest.append({'file': filename, 'sample_key': key, 'dataset': row['dataset'],
                             'sample_id': row['sample_id'], 'categories': labels})
    write_new(OUTPUT / 'case_manifest.json', manifest)
    html = '<html><meta charset="utf-8"><title>Stage5A new_dev cases</title><body>'
    html += '<h1>Stage5A 开发集固定失败案例（不是独立确认）</h1><p>红色FP，蓝色FN，绿色TP；原0.5阈值。</p>'
    for item in manifest:
        html += f'<h2>{item["dataset"]}: {item["sample_key"][:12]}</h2><p>{", ".join(item["categories"])}</p>'
        html += f'<img style="max-width:100%" src="{item["file"]}">'
    with (output / 'index.html').open('x') as stream:
        stream.write(html + '</body></html>')
    _threshold_plot()


if __name__ == '__main__':
    main()
