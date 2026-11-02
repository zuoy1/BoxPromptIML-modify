"""Deterministic same-image reference hypotheses, without label access."""
from __future__ import annotations

import math

import numpy as np


MIN_REFERENCES = 3
MAX_REFERENCES = 5
BRIGHTNESS_TOLERANCE = .10
LOG_RATIO_TOLERANCE = math.log(1.5)
CONTENT_EPSILON = .005


def _validate(patch):
    for name in ('brightness', 'gradient', 'texture', 'box_fraction', 'pred_fraction',
                 'stable_negative_fraction', 'outside_dilated_fraction'):
        if not np.isfinite(patch[name]) or patch[name] < 0:
            raise ValueError(f'Invalid patch descriptor: {name}')
    for name in ('brightness', 'box_fraction', 'pred_fraction',
                 'stable_negative_fraction', 'outside_dilated_fraction'):
        if patch[name] > 1:
            raise ValueError(f'Patch descriptor exceeds one: {name}')
    if (not all(np.isfinite(patch[name]) and patch[name] >= 0
                for name in ('x0', 'y0', 'x1', 'y1'))
            or not isinstance(patch['id'], str) or patch['x0'] >= patch['x1']
            or patch['y0'] >= patch['y1']):
        raise ValueError('Invalid patch identity or half-open bounds')


def reference_eligible(patch):
    """Return a reference hypothesis; stability cannot confirm true background."""
    _validate(patch)
    return (patch['outside_dilated_fraction'] == 1.
            and patch['pred_fraction'] <= .05
            and patch['stable_negative_fraction'] >= .9)


def candidate_categories(patch):
    _validate(patch)
    result = []
    if patch['pred_fraction'] >= .75:
        result.append('pos')
    if patch['pred_fraction'] <= .25:
        result.append('neg')
    if patch['box_fraction'] >= .75:
        result.append('box_inside')
    if .05 < patch['box_fraction'] < .95:
        result.append('box_edge')
    return result


def _overlap(first, second):
    return (max(first['x0'], second['x0']) < min(first['x1'], second['x1'])
            and max(first['y0'], second['y0']) < min(first['y1'], second['y1']))


def _content_distance(candidate, reference):
    return {'brightness': abs(candidate['brightness'] - reference['brightness']),
            'gradient': abs(math.log((candidate['gradient']+CONTENT_EPSILON)
                                     / (reference['gradient']+CONTENT_EPSILON))),
            'texture': abs(math.log((candidate['texture']+CONTENT_EPSILON)
                                    / (reference['texture']+CONTENT_EPSILON)))}


def _quality(distances):
    return {name: {'mean': float(np.mean([d[name] for d in distances])) if distances else None,
                   'max': max(d[name] for d in distances) if distances else None}
            for name in ('brightness', 'gradient', 'texture')}


def match_references(candidate, patches):
    """Gate content, then greedily choose nonoverlapping references by content rank.

    Each pair is evaluated inside the same image supplied by the caller. Pixel
    distance is deliberately absent from ranking. A failed minimum leaves its
    diagnostic candidates visible, without treating them as a valid reference.
    """
    _validate(candidate)
    identifiers = [patch['id'] for patch in patches]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate patch identity within one image')
    counts = dict(total=len(patches), reference_eligible=0, nonoverlap=0,
                  brightness=0, gradient=0, texture=0, joint_content=0, selected=0)
    reasons = dict(reference_ineligible=0, candidate_overlap=0, brightness=0,
                   gradient=0, texture=0, reference_overlap=0, max_references=0)
    nonoverlap_distances, ranked = [], []
    tolerances = dict(brightness=BRIGHTNESS_TOLERANCE,
                      gradient=LOG_RATIO_TOLERANCE, texture=LOG_RATIO_TOLERANCE)
    for patch in patches:
        if not reference_eligible(patch):
            reasons['reference_ineligible'] += 1
            continue
        counts['reference_eligible'] += 1
        if _overlap(candidate, patch):
            reasons['candidate_overlap'] += 1
            continue
        counts['nonoverlap'] += 1
        distances = _content_distance(candidate, patch)
        nonoverlap_distances.append(distances)
        passed = []
        for name, tolerance in tolerances.items():
            valid = distances[name] <= tolerance
            passed.append(valid)
            counts[name] += int(valid)
            reasons[name] += int(not valid)
        if all(passed):
            counts['joint_content'] += 1
            rank = sum((distances[name]/tolerances[name])**2 for name in tolerances)
            ranked.append((rank, patch['id'], patch, distances))
    chosen, chosen_distances = [], []
    for _, _, patch, distances in sorted(ranked, key=lambda row: row[:2]):
        if any(_overlap(patch, previous) for previous in chosen):
            reasons['reference_overlap'] += 1
        elif len(chosen) == MAX_REFERENCES:
            reasons['max_references'] += 1
        else:
            chosen.append(patch)
            chosen_distances.append(distances)
    counts['selected'] = len(chosen)
    applicable = len(chosen) >= MIN_REFERENCES
    return {'status': 'applicable' if applicable else 'not_applicable',
            'reason': None if applicable else 'fewer_than_three_nonoverlapping_matched_references',
            'reference_ids': [patch['id'] for patch in chosen], 'references': chosen,
            'counts': counts, 'rejection_reasons': reasons,
            'matching_quality': {'before_content_gate': _quality(nonoverlap_distances),
                                 'selected': _quality(chosen_distances)}}
