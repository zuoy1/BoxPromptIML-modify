"""4F1-V 的严格数值及已有产物检查；不构造模型、不读取图像。"""
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch

from fecd.pseudo.cache_schema import canonical_hash


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'JSON 字段重复: {key}')
        value[key] = item
    return value


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    return not isinstance(value, (float, int)) or math.isfinite(value)


def _decode(text):
    value = json.loads(text, object_pairs_hook=_object)
    if not _finite(value):
        raise ValueError('JSON 包含非有限数值')
    return value


def read(path):
    return _decode(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [_decode(line) for line in stream]


def _float32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('数值不能表示为有限 float32') from exc


def check_loss(row):
    fields = ('loss', 'pos_mean', 'neg_mean', 'pos_contribution', 'neg_contribution')
    for field in fields:
        value = row.get(field)
        if (type(value) is not float or not math.isfinite(value) or value < 0
                or math.copysign(1, value) < 0 or _float32(value) != value):
            raise ValueError(f'不是训练端保存的非负有限 float32: {field}={value!r}')
    for sign in ('pos', 'neg'):
        if row[f'{sign}_contribution'] != _float32(.5 * row[f'{sign}_mean']):
            raise ValueError(f'{sign} 贡献不符合 float32 半权重')
    expected = _float32(row['pos_contribution'] + row['neg_contribution'])
    if not math.isfinite(expected) or row['loss'] != expected:
        raise ValueError(f"Loss 分解错误: {row['loss']} != {expected}")


def check_metrics(records):
    for row in records:
        for key in ('tp', 'fp', 'fn', 'pixels'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f'混淆计数无效: {key}')
        tp, fp, fn, pixels = (row[k] for k in ('tp', 'fp', 'fn', 'pixels'))
        if pixels <= 0 or tp + fp + fn > pixels:
            raise ValueError('混淆计数超出像素网格')
        expected = {'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                    'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
                    'gt_positive_fraction': (tp + fn) / pixels}
        if 'prediction_positive_fraction' in row:
            expected['prediction_positive_fraction'] = (tp + fp) / pixels
        for key, value in expected.items():
            if (type(row.get(key)) not in (float, int) or isinstance(row[key], bool)
                    or not math.isfinite(row[key]) or abs(row[key] - value) > 1e-12):
                raise ValueError(f'逐图指标不符合混淆计数: {key}')


def checkpoint_hash(payload):
    if not _finite(payload):
        raise ValueError('Checkpoint 含非有限数值')
    states = dict(payload['model'])
    for name, value in payload['memory'].items():
        if value is not None:
            states[f'{name}.memory_bank'] = value
    digest = hashlib.sha256()
    for name, value in sorted(states.items()):
        if not torch.is_tensor(value) or value.device.type != 'cpu':
            raise ValueError(f'Checkpoint 非 CPU tensor: {name}')
        tensor = value.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def check_research(output):
    expected = {'P0': .345859, 'P1': .336740, 'P2F': .346021, 'P3F': .342369}
    metrics = {}
    for variant, rounded in expected.items():
        summary = read(output / 'evaluation' / variant / 'summary.json')
        score = summary['domains']['ALL']['image_macro_f1']
        if round(score, 6) != rounded:
            raise ValueError(f'科研指标改变: {variant}')
        metrics[variant] = score
    paired = read(output / 'analysis/paired_bootstrap.json')
    for comparison in paired['comparisons'].values():
        for metric in ('f1', 'iou'):
            for weight in ('image_weighted', 'group_equal', 'dataset_equal'):
                low, high = comparison[metric][weight]['ci95']
                if not low <= 0 <= high:
                    raise ValueError('原总体区间跨零结论发生变化')
    return {'conclusion': 'NO CLEAR GAIN', 'macro_f1': metrics,
            'overall_intervals_cross_zero': True,
            'P3F_favorable_domains': ['NIST16', 'Columbia', 'Coverage'],
            'P3F_declining_domain': 'CASIAv2', 'independent_confirmation': False}


def preserved_training_source(original_function):
    """版本化副本的非数值逻辑由 AST 契约测试逐条比较。"""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(original_function))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and "step['loss']" in ast.unparse(node):
            if '1e-07' in ast.unparse(node):
                for parent in ast.walk(tree):
                    for _, items in ast.iter_fields(parent):
                        if isinstance(items, list) and node in items:
                            items[items.index(node)] = ast.parse('check_loss(step)').body[0]
                            changes += 1
    if changes != 1:
        raise ValueError('原验收器 Loss 断言身份不匹配')
    return ast.dump(tree)
