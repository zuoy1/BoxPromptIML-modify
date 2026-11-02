"""Authorized formal run settings derived from the immutable 4F0 proposal."""
import hashlib
import json
from pathlib import Path

import torch

from fecd.pseudo.cache_schema import canonical_hash
from fecd.training.data import OPERATIONS
from fecd.training.f0_protocol import COMMON, VARIANTS

ROOT = Path(__file__).resolve().parents[2]
FORMAL_HASH = '8b25d14ae1d90c00db59e5794eaff4d038f82829f63abbd1513404923e8d46d2'
EVAL_HASH = '54baa31ac65ec3df978af09c43b2454fd5515db1c8117b86c6ff2cd7d267416f'
MEMBER_HASH = 'cc98c8c0222ecf20d6ba887fafa954eff07983b6fa360a43c5e0057757e50110'
NAMES = ('P0', 'P1', 'P2F', 'P3F')
DOMAINS = {'CASIAv2': 3278, 'NIST16': 172, 'Columbia': 52, 'Coverage': 28}
STATISTICS = {
    'seed': 20261029, 'repeats': 2000, 'ci': 'percentile 2.5/97.5',
    'sampling': 'paired source proxy groups WITHIN each dataset; retain all images in sampled group',
    'group_identity': 'dataset plus source_group_id',
    'weights': ['image_weighted', 'group_equal', 'dataset_equal'],
    'comparisons': ['P1-P0', 'P2F-P1', 'P3F-P2F'],
    'meaning': 'conditional single-training-seed development uncertainty, not seed stability or confirmation',
    'severe_overprediction': 'predicted area >= 0.20 AND >= 3 times GT area; descriptive only',
    'failure_case': 'F1 <= 0.10; descriptive only',
    'intermediate_dev_evaluations': 0,
}


def configuration(variant):
    if variant not in NAMES:
        raise ValueError('Only the four authorized formal variants are allowed')
    value = {k: v for k, v in COMMON.items() if k not in
             ('schema', 'epochs', 'steps', 'resume_probe_update', 'future_initialization')}
    value.update(schema='stage4f1-formal20-v1', variant=variant, epochs=20,
                 steps_per_epoch=883, updates=17660, **VARIANTS[variant],
                 formal_proposal_hash=FORMAL_HASH, membership_hash=MEMBER_HASH,
                 initialization='fresh seeded original model + frozen local TinyViT',
                 primary_checkpoint='epoch20 final', checkpoint_policy='atomic last after each epoch; final at20')
    return {**value, 'config_hash': canonical_hash(value)}


def verify_frozen():
    for name, field, hash_field, expected in (
        ('formal20_proposal', 'plan', 'plan_hash', FORMAL_HASH),
        ('evaluation_protocol', 'protocol', 'protocol_hash', EVAL_HASH),
    ):
        data = json.loads((ROOT / f'fecd/configs/stage4f0/{name}.json').read_text())
        if data[hash_field] != expected or canonical_hash(data[field]) != expected:
            raise ValueError(f'Frozen {name} hash mismatch')


def epoch_batches(size, seed, epoch, batch_size):
    if size < 1 or batch_size < 1 or epoch < 0:
        raise ValueError('Invalid epoch/batch dimensions')
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    for offset in range(0, size, batch_size):
        yield order[offset:offset + batch_size]


def geometry(seed, epoch, key):
    digest = hashlib.sha256(f'{seed}:{epoch}:{key}:geometry'.encode()).digest()
    return OPERATIONS[int.from_bytes(digest[:8], 'big') % len(OPERATIONS)]


def epoch_identity(keys, seed, epoch):
    order = [i for batch in epoch_batches(len(keys), seed, epoch, 4) for i in batch]
    return {'epoch': epoch, 'order_hash': canonical_hash(order),
            'sample_order_hash': canonical_hash([keys[i] for i in order]),
            'geometry_hash': canonical_hash([geometry(seed, epoch, keys[i]) for i in order])}

