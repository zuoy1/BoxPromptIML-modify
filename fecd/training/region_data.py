"""区域实验只读取已冻结的训练RGB、弱Box和K8 v2缓存。"""
import torch

from fecd.confidence.teacher_scores import sam_confidence
from fecd.training.data import ControlledDataset
from fecd.training.f1_protocol import MEMBER_HASH, geometry


def spatial(tensor, operation, inverse=False):
    if operation == 'identity':
        return tensor
    if operation == 'hflip':
        return tensor.flip(-1)
    if operation == 'vflip':
        return tensor.flip(-2)
    if operation in ('rot90:1', 'rot90:2', 'rot90:3'):
        k = int(operation[-1]) * (-1 if inverse else 1)
        return tensor.rot90(k, (-2, -1))
    raise ValueError('未定义的几何变换')


class RegionDataset(ControlledDataset):
    def __init__(self, seed=20261029, epoch=0):
        super().__init__('P1', seed, epoch)
        if self.membership_hash != MEMBER_HASH or len(self) != 3530:
            raise ValueError('训练成员身份变化')

    def __getitem__(self, index):
        self._validate_input(index)
        sample = self.dataset[index]
        if sample['k'] != 8:
            raise ValueError('区域实验仅允许K8 v2缓存')
        op = geometry(self.seed, self.epoch, sample['sample_key'])
        target = (sample['mu'] >= .5).float()
        box_union = sample['box_masks'].bool().any(0, keepdim=True)
        confidence = torch.from_numpy(sam_confidence(
            sample['mu'].numpy(), sample['var'].numpy(), sample['agreement'].numpy()))
        return {'image': spatial(sample['image'], op), 'target': spatial(target, op),
                'canonical_target': target, 'box_union': box_union,
                'mu': sample['mu'], 'var': sample['var'], 'confidence': confidence,
                'sample_key': sample['sample_key'], 'dataset': sample['dataset'],
                'operation': op, 'num_boxes': sample['num_boxes']}
