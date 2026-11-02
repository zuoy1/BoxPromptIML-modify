"""Single GT-free B-R mechanism pilot; outputs separate from frozen studies."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from fecd.diagnostics.stage3ra_split import read_split
from fecd.forensic_reference.inputs import PROJECT, SAM, ReadGuard, choose_rows, load_input
from fecd.forensic_reference.protocol import PROTOCOL, PROTOCOL_HASH
from fecd.forensic_reference.patches import extract_patches
from fecd.forensic_reference.matching import candidate_categories, reference_eligible, match_references
from fecd.forensic_reference.statistics import compare_regions
from fecd.forensic_reference.summary import summarize
from fecd.pseudo.cache_schema import canonical_hash

OUTPUT = PROJECT/'fecd/assets/stage3rb0'


def analyze_image(rgb, cache, row):
    result = {'sample_key':row['sample_key'], 'dataset':row['dataset'],
              'sample_id':row['sample_id'], 'source_group_id':row['source_group_id'],
              'num_boxes':int(cache['num_boxes']), 'scales':{}}
    for size in PROTOCOL['patch_sizes']:
        grid = extract_patches(rgb, cache['mu'], cache['var'], cache['agreement'],
                               cache['box_masks'], patch_size=size)
        patches = grid.pop('patches')
        candidates, controls = [], []
        category_pool = {name:[p for p in patches if name in candidate_categories(p)]
                         for name in ('pos','neg','box_inside','box_edge')}
        chosen_ids = set()
        for pool in category_pool.values():
            chosen_ids.update(pool[i]['id'] for i in np.linspace(
                0,len(pool)-1,min(16,len(pool)),dtype=int)) if pool else None
        eligible = [p for p in patches if reference_eligible(p)]
        control_ids = {eligible[i]['id'] for i in np.linspace(
            0,len(eligible)-1,min(16,len(eligible)),dtype=int)} if eligible else set()
        for patch in patches:
            categories = candidate_categories(patch)
            if categories and patch['id'] in chosen_ids:
                matched = match_references(patch, patches)
                candidates.append({'patch':patch, 'categories':categories,
                                   'matching':{k:v for k,v in matched.items() if k!='references'},
                                   'statistics':compare_regions(patch,matched['references'])})
            if patch['id'] in control_ids:
                matched = match_references(patch, patches)
                controls.append({'patch_id':patch['id'], 'patch':patch,
                                 'matching':{k:v for k,v in matched.items() if k!='references'},
                                 'statistics':compare_regions(patch,matched['references'])})
        result['scales'][str(size)] = {
            **grid, 'grid_patch_count':len(patches), 'candidate_count':len(candidates),
            'full_candidate_pool_counts':{k:len(v) for k,v in category_pool.items()},
            'full_candidate_union_count':sum(bool(candidate_categories(p)) for p in patches),
            'outside_patch_count':sum(p['outside_dilated_fraction']==1 for p in patches),
            'reference_pool_count':sum(reference_eligible(p) for p in patches),
            'candidates':candidates, 'controls':controls}
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refusing to overwrite B0 artifacts')
    split = read_split()
    rows = choose_rows(split['records'])
    indices = json.loads((SAM/'dataset_index.json').read_text())
    index = {r['sample_key']:r for r in indices}
    if len(index)!=len(indices):
        raise ValueError('Duplicate SAM index sample identity')
    OUTPUT.mkdir()
    protocol = {'protocol':PROTOCOL,'protocol_hash':PROTOCOL_HASH,
                'split_hash':split['manifest_sha256'],
                'selected_records':rows,'membership_hash':canonical_hash(rows),
                'gt_access':'closed for every partition'}
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cv2.setNumThreads(1)
    start = time.monotonic()
    with (OUTPUT/'per_image.jsonl').open('x') as images, (OUTPUT/'input_access.jsonl').open('x') as log:
        for i,row in enumerate(rows,1):
            rgb,cache,info = load_input(row,index[row['sample_key']])
            record = analyze_image(rgb,cache,row)
            record['input_metadata'] = info
            images.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
            images.flush()
            log.write(json.dumps(info,separators=(',',':'))+'\n')
            log.flush()
            print(json.dumps({'images':i,'total':len(rows),'dataset':row['dataset'],
                              'elapsed_seconds':time.monotonic()-start}),flush=True)
    summary = summarize(OUTPUT)
    manifest = {'status':'completed','images':len(rows),'elapsed_seconds':time.monotonic()-start,
                'protocol_hash':PROTOCOL_HASH, 'split_hash':split['manifest_sha256'],
                'gt_files_opened':0, 'new_confirm_gt_opened':False,
                'decision':summary['decision'], 'independent_confirmation':False}
    (OUTPUT/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
