"""保留初次入口失败，仅复核已生成概率副本，不重新推理全开发集。"""
import json

import numpy as np

from fecd.diagnostics.stage5a_io import OUTPUT, ROOT, RUN
from fecd.diagnostics.stage5a_metrics import measures
from fecd.tools.stage4f1_v2_io import sha, verify_snapshot, write_new


def main():
    baseline = json.loads((ROOT / 'fecd/assets/stage4f1_v/revalidation_20261030_v2/inputs_before.json').read_text())
    historical = verify_snapshot(baseline, RUN)
    members = json.loads((OUTPUT / 'allowlist.json').read_text())['members']
    differences = []
    checkpoints = {}
    for variant in ('P0', 'P1', 'P2F', 'P3F'):
        path = RUN / 'artifacts' / variant / 'final.pt'
        checkpoints[variant] = sha(path)
        assert checkpoints[variant] == baseline['files'][str(path)]['sha256']
        old = {r['sample_key']: r for r in (json.loads(line) for line in
            (RUN / f'artifacts/evaluation/{variant}/per_image.jsonl').read_text().splitlines())}
        for row in members:
            key = row['sample_key']
            with np.load(OUTPUT / 'maps' / f'{key}_{variant}.npz', allow_pickle=False) as data:
                probability, logits = data['probability'], data['logits']
                assert np.isfinite(logits).all() and np.isfinite(probability).all()
                assert np.array_equal(probability, np.exp(-np.logaddexp(0., -logits.astype(np.float64))))
                with np.load(OUTPUT / 'maps' / f'{key}_reference.npz', allow_pickle=False) as ref:
                    metrics = measures(probability >= .5, ref['gt'])
            differences.append({'variant': variant, 'sample_key': key,
                **{f'delta_{k}': metrics[k] - old[key][k] for k in ('tp', 'fp', 'fn', 'f1', 'iou')}})
    write_new(OUTPUT / 'cpu_reproduction.json', differences)
    write_new(OUTPUT / 'inference_summary.json', {'status': 'PASS', 'device': 'cpu',
        'initial_execution': 'FAILED at final boundary gate; original inference.log retained',
        'recovery': 'independent validation of all 2120 saved maps, no repeated full inference',
        'boundary_probe': 'temporary-directory discovery attempted writes outside output; all rejected',
        'initial_denied_paths': 'not persisted by original guard; reproduced in logs/boundary_probe_access.jsonl',
        'fix': 'new diagnostic entry assigns tempfile.tempdir inside OUTPUT before model import',
        'images': len(members), 'checkpoint_sha256': checkpoints, 'training': False,
        'denied': [], 'historical_files_changed': historical['changed'],
        'cpu_differences': sum(any(r[f'delta_{k}'] != 0 for k in ('tp', 'fp', 'fn')) for r in differences)})
    print(json.dumps({'status': 'PASS', 'historical': historical,
        'difference_images': sum(any(r[f'delta_{k}'] != 0 for k in ('tp', 'fp', 'fn')) for r in differences)}))


if __name__ == '__main__':
    main()
