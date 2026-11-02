"""Versioned single-epoch preflight and unexecuted formal plan."""
from fecd.pseudo.cache_schema import canonical_hash

EVAL_PROTOCOL = {
    'version':'stage4f0-strict-newdev-v1','role':'new_dev','images':530,
    'batch_size':1,'seed':20261029,'threshold':.5,
    'preprocessing':'PIL RGB BILINEAR224 ImageNet mean/std; no augmentation',
    'probability':'sigmoid raw logits; no min-max',
    'gt':'verified native binary0/255; NEAREST224; >127',
    'empty_both_f1_iou':1.,'empty_one_f1_iou':0.,
    'memory':'independent model; restore checkpoint memory before EACH image',
    'primary':'new_dev image-macro PixelF1','secondary':['IoU','per-domain','dataset-equal','micro'],
    'near_full_area':.99,'abnormal_high_area':.95,'abnormal_low_area':.001,
    'near_zero_area':.01,'probability_saturation_low':.01,'probability_saturation_high':.99,
    'confirmation_access':False,'meaning':'development engineering loop, not independent confirmation',
}
EVAL_HASH=canonical_hash(EVAL_PROTOCOL)
STOP_PROTOCOL = {
    'technical_failures':'order dependence,GT training access,identity/budget mismatch,NaN,resume mismatch -> BLOCKED',
    'segmentation_risk':'new_dev >=50% images zero area or area>=.99 AND image-macro F1<=.05 -> stop formal request',
    'numeric_saturation':'new_dev >=90% images probability mean<=.01 or>=.99 -> stop formal request',
    'no_tuning':'no loss/lr/threshold changes based on preflight results',
}
COMMON={
    'schema':'stage4f0-one-epoch-v1','seed':20261029,'epochs':1,'batch_size':4,
    'drop_last':False,'training_images':3530,'steps':883,
    'split_hash':'022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0',
    'tinyvit_checkpoint':'/home/weights/tiny_vit_5m_22kto1k_distill.pth',
    'device':'cuda:0','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
    'optimizer':'Adam','clip_norm':.5,'precision':'float32','scheduler':'constant','workers':0,
    'loss':'4P per-image .5pos+.5neg masked BCE; empty class zero, no redistribution',
    'augmentation':'4P seed:epoch:sample_key:geometry SHA256, six discrete ops',
    'order':'randperm sorted sample keys, torch.Generator(seed+epoch)',
    'resume_probe_update':4,'eval_protocol_hash':EVAL_HASH,
    'future_initialization':'preflight checkpoints never initialize formal20epoch',
}
VARIANTS={
    'P0':{'teacher_k':1,'selection':'all'},
    'P1':{'teacher_k':8,'selection':'all'},
    'P2F':{'teacher_k':8,'selection':'fixed_random','selection_epoch':0},
    'P3F':{'teacher_k':8,'selection':'fixed_confidence','selection_epoch':0},
    'P2E':{'teacher_k':8,'selection':'dynamic_random','selection_epoch':'actual epoch','execution':'not trained in4F0'},
}


def configuration(variant):
    value={**COMMON,'variant':variant,**VARIANTS[variant]}
    return {**value,'config_hash':canonical_hash(value)}


FORMAL_PLAN={
    'status':'frozen proposal; not authorized or executed',
    'seed':20261029,'epochs':20,'batch_size':4,'drop_last':False,'steps_per_epoch':883,
    'updates_per_variant':17660,'main_variants':['P0','P1','P2F','P3F'],
    'main_comparisons':['P1-P0','P2F-P1','P3F-P2F'],
    'supplement':'P2E dynamic random separately registered; future explicit authorization required',
    'checkpoint':'epoch20 final; no per-method best-dev epoch selection',
    'hyperparameters':{k:COMMON[k] for k in ('lr','betas','eps','weight_decay','clip_norm','precision','scheduler')},
    'initialization':'fresh 4P strategy from local TinyViT+common seeded student layers; not preflight state',
    'evaluation_protocol_hash':EVAL_HASH,'stop_rules':STOP_PROTOCOL,
    'seeds':'one common seed only; additional seeds require equal authorization for all main variants',
    'locked':'new_confirm,teacher_diag,IND/OOD tests; historical5737 B0 not comparable',
}
FORMAL_HASH=canonical_hash(FORMAL_PLAN)
