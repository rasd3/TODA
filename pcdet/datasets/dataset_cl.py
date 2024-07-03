from collections import defaultdict
from pathlib import Path

import copy
import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor_cl import DataAugmentorCL
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplateCL(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentorCL(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict_1, data_dict_2):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict_1, 'gt_boxes should be provided for training'
            assert 'gt_boxes' in data_dict_2, 'gt_boxes should be provided for training'
            gt_boxes_mask_1 = np.array([n in self.class_names for n in data_dict_1['gt_names']], dtype=np.bool_)
            gt_boxes_mask_2 = np.array([n in self.class_names for n in data_dict_2['gt_names']], dtype=np.bool_)

            data_dict_1 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_1,
                    'gt_boxes_mask': gt_boxes_mask_1
                }
            )

            data_dict_2['augmentation_list'] = copy.deepcopy(data_dict_1['augmentation_list'])
            data_dict_2['augmentation_params'] = copy.deepcopy(data_dict_1['augmentation_params'])

            data_dict_2 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_2,
                    'gt_boxes_mask': gt_boxes_mask_2
                }
            )

            gt_boxes_mask_2 = np.array([n in self.class_names for n in data_dict_2['gt_names']], dtype=np.bool_)

            data_dict_2.pop('augmentation_list')
            data_dict_2.pop('augmentation_params')

            data_dict_2 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_2,
                    'gt_boxes_mask': gt_boxes_mask_2
                }
            )

        if data_dict_1.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_1['gt_names'], self.class_names)
            data_dict_1['gt_boxes'] = data_dict_1['gt_boxes'][selected]
            data_dict_1['gt_names'] = data_dict_1['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict_1['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict_1['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict_1['gt_boxes'] = gt_boxes

            if data_dict_1.get('gt_boxes2d', None) is not None:
                data_dict_1['gt_boxes2d'] = data_dict_1['gt_boxes2d'][selected]

        if data_dict_2.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_2['gt_names'], self.class_names)
            data_dict_2['gt_boxes'] = data_dict_2['gt_boxes'][selected]
            data_dict_2['gt_names'] = data_dict_2['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict_2['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict_2['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict_2['gt_boxes'] = gt_boxes

            if data_dict_2.get('gt_boxes2d', None) is not None:
                data_dict_2['gt_boxes2d'] = data_dict_2['gt_boxes2d'][selected]

        if data_dict_1.get('points', None) is not None:
            data_dict_1 = self.point_feature_encoder.forward(data_dict_1)

        if data_dict_2.get('points', None) is not None:
            data_dict_2 = self.point_feature_encoder.forward(data_dict_2)

        data_dict_1 = self.data_processor.forward(
            data_dict=data_dict_1
        )

        data_dict_2 = self.data_processor.forward(
            data_dict=data_dict_2
        )

        #  if self.training and len(data_dict['gt_boxes']) == 0:
            #  new_index = np.random.randint(self.__len__())
            #  return self.__getitem__(new_index)
        if self.training:
            if len(data_dict_1['gt_boxes']) == 0 or len(data_dict_2['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        data_dict_1.pop('gt_names', None)
        data_dict_2.pop('gt_names', None)

        return data_dict_1, data_dict_2

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict_adv = defaultdict(list)
        data_dict_org = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample[0].items():
                data_dict_adv[key].append(val)
            for key, val in cur_sample[1].items():
                data_dict_org[key].append(val)
        batch_size = len(batch_list)

        ret_adv = {}

        for key, val in data_dict_adv.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret_adv[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret_adv[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    # print(max_gt)
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    # print(batch_gt_boxes3d.shape)
                    for k in range(batch_size):
                        # print(k)
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    # print('over')
                    ret_adv[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret_adv[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret_adv[key] = np.stack(images, axis=0)
                elif key in ['augmentation_list', 'augmentation_params']:
                    ret_adv[key] = val
                else:
                    ret_adv[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret_adv['batch_size'] = batch_size

        ret_org = {}

        for key, val in data_dict_org.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret_org[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret_org[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    # print(max_gt)
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    # print(batch_gt_boxes3d.shape)
                    for k in range(batch_size):
                        # print(k)
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    # print('over')
                    ret_org[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret_org[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret_org[key] = np.stack(images, axis=0)
                elif key in ['augmentation_list', 'augmentation_params', 'augmentation_list_1', 'augmentation_params_1', 'box_frame_idx']:
                    ret_org[key] = val
                else:
                    ret_org[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret_org['batch_size'] = batch_size
        return ret_adv, ret_org
