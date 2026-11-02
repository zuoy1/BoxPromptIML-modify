"""Standalone SVG diagnostics from the saved epoch CSV, without added dependencies."""
import csv
from html import escape

from fecd.training.f1_protocol import NAMES
from fecd.training.f1_storage import resolve_output_root, parse_output_root


def main(output_root=None):
    root = resolve_output_root(output_root)/'analysis'
    with (root/'training_curves.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['dataset'] == 'ALL']
    colors = {'P0':'#1764ab','P1':'#d45b16','P2F':'#16814a','P3F':'#9245ad'}
    panels = [('mean_loss','Mean balanced BCE',False),
              ('student_area','Training predicted foreground fraction',True),
              ('probability_mean','Training mean sigmoid probability',True)]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1050" viewBox="0 0 1000 1050">',
           '<rect width="1000" height="1050" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#222">',
           '<text x="80" y="28" font-size="19">Stage 4F1: four controlled 20-epoch students</text>',
           '<text x="80" y="50">Pre-update training outputs from changing parameters; not intermediate new_dev evaluation</text>']
    for index,(field,title,fraction) in enumerate(panels):
        top = 90 + index*310
        values = [float(r[field]) for r in rows]
        maximum = 1. if fraction else max(values)*1.1
        height, width, left = 220, 820, 80
        svg.append(f'<text x="{left}" y="{top-14}" font-size="16">{escape(title)}</text>')
        for tick in range(6):
            y = top+height-tick/5*height
            value = maximum*tick/5
            svg.append(f'<path d="M{left} {y}H{left+width}" stroke="#ddd" fill="none"/>')
            svg.append(f'<text x="{left-12}" y="{y+4}" text-anchor="end">{value:.2f}</text>')
        for epoch in (1,5,10,15,20):
            x = left+(epoch-1)/19*width
            svg.append(f'<text x="{x}" y="{top+height+20}" text-anchor="middle">{epoch}</text>')
        svg.append(f'<text x="{left+width/2}" y="{top+height+42}" text-anchor="middle">Epoch</text>')
        for variant in NAMES:
            group = sorted([r for r in rows if r['variant']==variant], key=lambda r:int(r['epoch']))
            points = ' '.join(f'{left+(int(r["epoch"])-1)/19*width:.3f},{top+height-float(r[field])/maximum*height:.3f}' for r in group)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[variant]}" stroke-width="2.5"/>')
            for r in group:
                x, y = left+(int(r['epoch'])-1)/19*width, top+height-float(r[field])/maximum*height
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[variant]}"><title>{variant} epoch {r["epoch"]}: {float(r[field]):.6f}</title></circle>')
    for i,variant in enumerate(NAMES):
        x = 140+i*200
        svg.append(f'<rect x="{x}" y="1010" width="22" height="4" fill="{colors[variant]}"/><text x="{x+30}" y="1017">{variant}</text>')
    svg.extend(['</g>','</svg>'])
    with (root/'training_curves.svg').open('x') as stream:
        stream.write('\n'.join(svg)+'\n')


if __name__ == '__main__':
    main(parse_output_root())
