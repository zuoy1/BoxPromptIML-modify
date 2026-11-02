"""Evaluate only four completed Epoch20 students with the unchanged strict evaluator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json

from fecd.evaluation.memory_audit import audit_memory
from fecd.evaluation.strict import evaluate, write_json
from fecd.tools.preflight_formal_student import runtime_hashes, sha
from fecd.training.f1_protocol import ROOT, NAMES, verify_frozen
from fecd.training.f1_storage import resolve_output_root, parse_output_root


def main(output_root=None):
    output = resolve_output_root(output_root)
    verify_frozen()
    preflight = json.loads((output/'preflight.json').read_text())
    for path, digest in preflight['runtime_hashes'].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Frozen runtime changed: {path}')
    for variant in NAMES:
        result = json.loads((output/variant/'result.json').read_text())
        if result['status'] != 'PASS' or result['epochs'] != 20 or result['logical_updates'] != 17660:
            raise ValueError('All four formal students must finish before new_dev evaluation')
    target = output/'evaluation'
    target.mkdir(exist_ok=False)
    write_json(target/'code_before_gt.json', {'runtime_hashes':runtime_hashes(),
                'wrapper_sha256':sha(__file__), 'primary_checkpoint':'Epoch20 final',
                'intermediate_dev_evaluations':0, 'new_confirm_gt_reads':0})
    for variant in NAMES:
        checkpoint = output/variant/'final.pt'
        memory = audit_memory(checkpoint, target/f'{variant}_memory_gate.json')
        if memory['status'] != 'PASS':
            write_json(output/'BLOCKED_evaluation.json', {'status':'BLOCKED', 'variant':variant,
                                                         'reason':'memory order independence failed'})
            raise ValueError('Final checkpoint memory order gate failed')
        result = evaluate(checkpoint, target/variant)
        print(json.dumps({'variant':variant, 'metrics':result['domains']['ALL'],
                          'dataset_equal':result['dataset_equal'],
                          'stop_formal_request':result['stop_formal_request']}), flush=True)


if __name__ == '__main__':
    main(parse_output_root())
