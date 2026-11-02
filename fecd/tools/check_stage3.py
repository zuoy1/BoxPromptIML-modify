"""Runnable stage gates, syntax/name analysis and frozen-artifact protection."""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import symtable
from collections import defaultdict
from pathlib import Path

from fecd.diagnostics.protocol import PROTOCOL, PROTOCOL_HASH
from fecd.evidence.cache import ROOT, evidence_config_hash, read_split
from fecd.pseudo.cache_schema import canonical_hash
from fecd.tools.prepare_stage3_split import component_groups


def analyze_source(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    symbols = symtable.symtable(source, str(path), 'exec')
    known = {s.get_name() for s in symbols.get_symbols() if s.is_assigned() or s.is_imported()}
    known.update(dir(builtins))
    known.update(('__name__', '__file__', '__package__', '__doc__', '__annotations__'))
    issues = []

    def walk(table):
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                issues.append(f'{path}: unresolved global {symbol.get_name()} in {table.get_name()}')
        for child in table.get_children():
            walk(child)

    walk(symbols)
    if any(part in ('evidence', 'confidence', 'datasets') for part in path.parts):
        forbidden = ('prepared_mask', 'gt_path', 'gt_root', '/Gt/')
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(token in node.value for token in forbidden):
                    issues.append(f'{path}:{node.lineno}: forbidden GT-path field in student/evidence boundary')
    return issues


def check_static() -> dict:
    files = []
    for folder in ('evidence', 'confidence', 'diagnostics'):
        files.extend((ROOT / folder).glob('*.py'))
    files.extend((ROOT / 'tools').glob('*stage3*.py'))
    files.extend(ROOT / 'tools' / name for name in (
        'audit_pseudo.py', 'fit_evidence_stats.py', 'generate_evidence_cache.py'))
    files.extend((ROOT / 'tests').glob('test_*stage3*.py'))
    files.extend(ROOT / 'tests' / name for name in (
        'test_evidence.py', 'test_evidence_cache.py', 'test_teacher_scores.py',
        'test_diagnostics.py', 'test_sample_audit.py', 'test_summary.py'))
    files = sorted(set(files))
    issues = [issue for path in files for issue in analyze_source(path)]
    return {'files': len(files), 'checks': 'AST syntax; symbol-table unresolved globals; GT-path boundary',
            'issues': issues, 'third_party_linter': 'not installed; this is a scoped stdlib static analyzer'}


def check_gate(require_3a=False, protected=False) -> dict:
    split = read_split()
    groups = component_groups(split['records'])
    memberships = defaultdict(set)
    for row in split['records']:
        memberships[groups[row['sample_key']]].add(row['subset'])
        if groups[row['sample_key']] != row['source_group_id']:
            raise ValueError('Stored source-group component mismatch')
    if any(len(values) != 1 for values in memberships.values()):
        raise ValueError('Source group crosses train_fit/teacher_diag')
    frozen = json.loads((ROOT / 'assets/stage3_analysis_protocol.json').read_text())
    if frozen != {'protocol': PROTOCOL, 'protocol_hash': PROTOCOL_HASH}:
        raise ValueError('Pre-GT analysis protocol changed')
    result = {'split_hash': split['manifest_sha256'], 'protocol_hash': PROTOCOL_HASH,
              'group_crossings': 0, 'n_groups': len(memberships), 'static': check_static()}
    if result['static']['issues']:
        raise ValueError(json.dumps(result['static']['issues']))
    if require_3a:
        pilot = json.loads((ROOT / 'assets/stage3_evidence/pilot_report.json').read_text())
        audit = json.loads((ROOT / 'assets/stage3_evidence/stage3_evidence_cache_audit.json').read_text())
        stats = json.loads((ROOT / 'assets/stage3_evidence_stats.json').read_text())
        fit = [row for row in split['records'] if row['subset'] == 'train_fit']
        keys_hash = hashlib.sha256('\n'.join(sorted(r['sample_key'] for r in fit)).encode()).hexdigest()
        if pilot['status'] != 'passed' or len(pilot['dataset_counts']) != 4:
            raise ValueError('Multi-dataset feature pilot has not passed')
        if (audit['requested'] != len(split['records']) or audit['failed']
                or audit['succeeded'] != len(split['records']) or not audit['validate_only']
                or audit['split_manifest_sha256'] != split['manifest_sha256']
                or audit['config_hash'] != evidence_config_hash()):
            raise ValueError('Full evidence/SAM validation has not passed')
        payload = {k: v for k, v in stats.items() if k != 'stats_hash'}
        if (stats['stats_hash'] != canonical_hash(payload)
                or stats['image_count'] != len(fit)
                or stats['provenance']['partition'] != 'train_fit'
                or stats['provenance']['sample_keys_sha256'] != keys_hash
                or stats['provenance']['split_manifest_sha256'] != split['manifest_sha256']):
            raise ValueError('Complete GT-free train_fit normalization has not passed')
        result['stage3a'] = {'status': 'passed', 'fit_count': len(fit), 'cache_count': audit['succeeded'],
                            'stats_hash': stats['stats_hash'], 'extractor_hash': evidence_config_hash()}
    if protected:
        frozen_hashes = json.loads((ROOT / 'assets/stage3_protected_hashes.json').read_text())
        changed = []
        for filename, expected in frozen_hashes.items():
            path = Path(filename)
            if not path.is_absolute():
                path = ROOT.parent / path
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                changed.append(filename)
        if changed:
            raise ValueError(f'Protected files changed: {changed}')
        result['protected'] = {'checked': len(frozen_hashes), 'changed': 0}
    result['status'] = 'passed'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-3a', action='store_true')
    parser.add_argument('--protected', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/stage3_checks.json')
    args = parser.parse_args()
    result = check_gate(args.require_3a, args.protected)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
