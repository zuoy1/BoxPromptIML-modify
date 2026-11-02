"""Controlled hard-label BCE baselines, separate from historical B0-B5."""
from fecd.pseudo.cache_schema import canonical_hash

COMMON = {
    'schema':'boxprompt-stage4p-smoke-v1', 'seed':20261029,
    'split_hash':'022da143a26833f1613db722ae3be152117948330621e028f02b824f2d1680d0',
    'training_role':'student_train', 'training_images':3530,
    'tinyvit_checkpoint':'/home/weights/tiny_vit_5m_22kto1k_distill.pth',
    'device':'cuda:0', 'batch_size':4, 'updates':8, 'checkpoint_update':4,
    'optimizer':'Adam', 'lr':1e-4, 'weight_decay':0.,
    'betas':[.9,.999], 'eps':1e-8, 'gradient_clip_norm':.5,
    'scheduler':'constant', 'precision':'float32', 'workers':0,
    'augmentation':'sample-key/seed/epoch deterministic discrete geometry; no photometric transforms',
    'loss':'per-image selected-positive mean and selected-negative mean; .5 each; empty class0; batch mean',
    'selection':'P2/P3 floor(n/2) separately per image/sign; shared seed/tie priorities',
    'image_protocol':'frozen FECD PIL RGB BILINEAR square224, ImageNet normalization',
    'order':'torch.randperm(sorted sample_key, generator seed+epoch), no drop_last for future epoch',
    'evaluation':'none during smoke; new_dev evaluator only by separately authorized formal protocol',
    'confirmation':'closed', 'max_epochs':1,
}
VARIANTS = {'P0':{'teacher_k':1,'target':'hard','selection':'all'},
            'P1':{'teacher_k':8,'target':'hard','selection':'all'},
            'P2':{'teacher_k':8,'target':'hard','selection':'random'},
            'P3':{'teacher_k':8,'target':'hard','selection':'confidence'}}


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError('Expected P0/P1/P2/P3')
    value={**COMMON,'variant':variant,**VARIANTS[variant]}
    return {**value,'config_hash':canonical_hash(value)}
