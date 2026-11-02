"""Actual original-model inference order gate using only student_train RGB."""
import gc
from pathlib import Path

import torch

from fecd.diagnostics.stage3ra_split import read_split
from fecd.evaluation.data import file_sha, load_rgb
from fecd.evaluation.memory import initial_memory, infer_one, memory_hash, reset_memory
from fecd.evaluation.strict import load_independent_model, write_json
from fecd.training.f0_protocol import EVAL_HASH


def audit_memory(checkpoint,output_path):
    checkpoint=Path(checkpoint).resolve(strict=True)
    before=file_sha(checkpoint)
    rows=sorted([r for r in read_split()['records'] if r['subset']=='student_train'],key=lambda r:r['sample_key'])[:2]
    images=[load_rgb(row)[0] for row in rows]
    model,snapshot,meta=load_independent_model(checkpoint)
    original_memory=memory_hash(snapshot)
    forward=[infer_one(model,image.to(meta['device']),snapshot) for image in images]
    after_eval=memory_hash(initial_memory(model))
    reverse={i:infer_one(model,images[i].to(meta['device']),snapshot) for i in [1,0]}
    repeat=infer_one(model,images[0].to(meta['device']),snapshot)
    # Observe the original eval mutation separately; strict production inference always resets.
    reset_memory(model,snapshot)
    model.eval()
    with torch.inference_mode():
        plain_first=model(images[0].to(meta['device'])).detach().cpu()
        plain_second=model(images[0].to(meta['device'])).detach().cpu()
    original_eval_repeat_equal=torch.equal(plain_first,plain_second)
    del model
    gc.collect();torch.cuda.empty_cache()
    fresh,fresh_snapshot,_=load_independent_model(checkpoint)
    cold=infer_one(fresh,images[0].to(meta['device']),fresh_snapshot)
    checks={'repeat_exact_equal':torch.equal(forward[0],repeat),
            'reverse_order_exact_equal':all(torch.equal(forward[i],reverse[i]) for i in [0,1]),
            'cold_model_exact_equal':torch.equal(forward[0],cold),
            'snapshot_not_mutated':memory_hash(snapshot)==original_memory,
            'disk_checkpoint_not_mutated':file_sha(checkpoint)==before}
    result={'status':'PASS' if all(checks.values()) else 'BLOCKED','checks':checks,
            'eval_protocol_hash':EVAL_HASH,'checkpoint':str(checkpoint),'checkpoint_sha256':before,
            'sample_keys':[r['sample_key'] for r in rows],'rgb_sha256':[r['image_sha256'] for r in rows],
            'gt_access_count':0,'batch_size':1,'before_memory_sha256':original_memory,
            'after_eval_memory_sha256':after_eval,'original_eval_updates_memory':original_memory!=after_eval,
            'original_eval_repeat_without_reset_exact_equal':original_eval_repeat_equal,
            'original_semantics':'eval forward updates plain memory_bank; memory blending is training-only',
            'strict_semantics':'independent model; checkpoint memory restored before every image',
            'max_repeat_logit_difference':float((forward[0]-repeat).abs().max()),
            'max_cold_logit_difference':float((forward[0]-cold).abs().max())}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    write_json(output_path,result)
    return result
