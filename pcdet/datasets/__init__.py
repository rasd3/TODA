import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset

from .two_dataset import CutMixDatasetTemplate
from .cutmix_dataset.waymo_nus_cutmix_dataset import WaymoNusCutMixDataset
from .polarmix_dataset.waymo_nus_polarmix_dataset import WaymoNusPolarMixDataset
from .nuscenes.nuscenes_mixup_dataset import NuScenesMixUpDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset, 
    'CutMixDatasetTemplate': CutMixDatasetTemplate, 
    'WaymoNusCutMixDataset': WaymoNusCutMixDataset, 
    'WaymoNusPolarMixDataset': WaymoNusPolarMixDataset, 
    'NuScenesMixUpDataset': NuScenesMixUpDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


# for cutmix
def build_cutmix_dataloader(dataset_cfg, dataset_names, batch_size, dist, workers=4, 
                            logger=None, training=True):
    assert training == True

    dataset = __all__[dataset_cfg.DATASET_NAME](
        dataset_cfg=dataset_cfg, 
        training=training, 
        dataset_names=dataset_names, 
        logger=logger
    )

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            return

    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, 
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch, 
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

# for mixup
def build_mixup_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, 
                           logger=None, training=True, pseudo_info_path=None):
    assert training == True

    dataset = __all__[dataset_cfg.DATASET_NAME](
        dataset_cfg=dataset_cfg, 
        class_names=class_names, 
        root_path=root_path, 
        training=training, 
        logger=logger, 
        pseudo_info_path=pseudo_info_path
    )

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            return
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers, 
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch, 
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
