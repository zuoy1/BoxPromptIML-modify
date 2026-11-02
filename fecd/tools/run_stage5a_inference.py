"""独立new_dev只读诊断入口；不导入学生Dataset、不训练。"""
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

from fecd.diagnostics.stage5a_io import OUTPUT, RUN, prepare, teacher, DiagnosticBoundary
from fecd.evaluation.data import load_rgb, load_dev_gt
from fecd.evaluation.memory import infer_one
from fecd.evaluation.strict import load_independent_model
from fecd.diagnostics.stage5a_metrics import measures
from fecd.tools.stage4f1_v2_io import sha, write_new, verify_snapshot


def main():
    start = time.perf_counter()
    temporary = OUTPUT / 'tmp'
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    baseline_path = Path('/home/projects/BoxPromtIML/fecd/assets/stage4f1_v/revalidation_20261030_v2/inputs_before.json')
    baseline = json.loads(baseline_path.read_text())
    verify_snapshot(baseline, RUN)
    rows, allowed, sources, indices, paths = prepare()
    checkpoints = {v: sha(RUN / 'artifacts' / v / 'final.pt') for v in ('P0', 'P1', 'P2F', 'P3F')}
    (OUTPUT / 'maps').mkdir(exist_ok=False)
    with (OUTPUT / 'logs/file_access.jsonl').open('x') as log:
        guard = DiagnosticBoundary(paths, log)
        sys.addaudithook(guard)
        for row in rows:
            key = row['sample_key']
            _, native_hw = load_rgb(row)
            gt = load_dev_gt(row, sources[key], native_hw, allowed, OUTPUT / 'logs/gt_access.jsonl')
            teachers = {k: teacher(indices[k][key], row, k, native_hw) for k in (1, 8)}
            path = OUTPUT / 'maps' / f'{key}_reference.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, gt=gt, K1=teachers[1][0], K8=teachers[8][0],
                                    boxes=teachers[8][1])
        write_new(OUTPUT / 'reference_inputs_sha256.json',
                  {p: sha(p) for p in sorted(paths)})
        reproduction = []
        for variant in checkpoints:
            model, snapshot, metadata = load_independent_model(RUN / 'artifacts' / variant / 'final.pt', device='cpu')
            previous = {r['sample_key']: r for r in
                        (json.loads(x) for x in (RUN / f'artifacts/evaluation/{variant}/per_image.jsonl').read_text().splitlines())}
            for i, row in enumerate(rows):
                key = row['sample_key']
                image, _ = load_rgb(row)
                logits = infer_one(model, image, snapshot)[0, 0].numpy()
                probability = np.exp(-np.logaddexp(0., -logits.astype(np.float64)))
                with (OUTPUT / 'maps' / f'{key}_{variant}.npz').open('xb') as stream:
                    np.savez_compressed(stream, probability=probability, logits=logits)
                with np.load(OUTPUT / 'maps' / f'{key}_reference.npz', allow_pickle=False) as reference:
                    metrics = measures(probability >= .5, reference['gt'])
                reproduction.append({'variant': variant, 'sample_key': key,
                                      **{f'delta_{k}': metrics[k] - previous[key][k] for k in ('tp', 'fp', 'fn', 'f1', 'iou')}})
                if (i + 1) % 50 == 0:
                    print(json.dumps({'variant': variant, 'completed': i + 1,
                                      'seconds': time.perf_counter() - start}), flush=True)
            del model, snapshot
        after = {v: sha(RUN / 'artifacts' / v / 'final.pt') for v in checkpoints}
        if checkpoints != after or guard.denied:
            raise ValueError('冻结权重或访问边界异常')
        write_new(OUTPUT / 'cpu_reproduction.json', reproduction)
        verify_snapshot(baseline, RUN)
        write_new(OUTPUT / 'inference_summary.json', {'status': 'PASS', 'device': 'cpu',
                  'images': len(rows), 'checkpoint_sha256': after, 'training': False,
                  'gt_reads': guard.counts.get('GT', 0), 'access_counts': guard.counts,
                  'denied': guard.denied, 'seconds': time.perf_counter() - start,
                  'historical_files_changed': 0, 'cpu_differences': sum(any(r[f'delta_{k}'] != 0 for k in ('tp','fp','fn')) for r in reproduction)})


if __name__ == '__main__':
    main()
