import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ...utils.perturb_utils import *
from ...utils.alpha_shape import alpha_shape_border
from ...utils.sampling import farthest_point_sampling
from ..processor.intra_domain_point_mixup import intra_domain_point_mixup, intra_domain_point_mixup_cd
from ..dataset_cl import DatasetTemplateCL

class NuScenesMixUpAdvDataset(DatasetTemplateCL):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, pseudo_info_path=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.pseudo_info_path = pseudo_info_path
        self.gt_infos = []
        self.ps_infos = []
        self.include_nuscenes_data(self.mode)
        self.repeat = self.dataset_cfg.REPEAT
        self.infos = self.gt_infos + self.ps_infos
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def pseudo_frame(self, nus_info_c, nus_points_c, thres=0.3):
        if 'p_score' in nus_info_c:
            box_mask = nus_info_c['p_score'] < thres
            #  exc_gt_bbox = nus_info_c['gt_boxes'][box_mask]
            #  if self.dataset_cfg.get('SHIFT_COOR', None):
                #  exc_gt_bbox[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            #  exc_points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(nus_points_c[:, :3],
                                                                        #  exc_gt_bbox[:, :7])
            #  exc_points_mask = exc_points_mask.sum(0) == 0
            #  nus_points_c = nus_points_c[exc_points_mask]
            valid_mask = box_mask == False
            nus_info_c['gt_boxes'] = nus_info_c['gt_boxes'][valid_mask]
            nus_info_c['gt_names'] = nus_info_c['gt_names'][valid_mask]
            nus_info_c['p_score'] = nus_info_c['p_score'][valid_mask]
            
        return nus_info_c, nus_points_c

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        gt_nuscenes_infos = []
        
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                gt_nuscenes_infos.extend(infos)

        self.gt_infos.extend(gt_nuscenes_infos)

        ps_nuscenes_infos = []

        if not Path(self.pseudo_info_path).exists():
            self.logger.info('Error! Pseudo labels infos dont exist!')
            return 
        with open(self.pseudo_info_path, 'rb') as f:
            infos = pickle.load(f)
            ps_nuscenes_infos.extend(infos)
        
        self.ps_infos.extend(ps_nuscenes_infos)

        self.logger.info('Total samples for NuScenes dataset: %d' % (len(self.gt_infos) + len(self.ps_infos)))
        self.logger.info('GT samples for NuScenes dataset: %d' % (len(self.gt_infos)))
        self.logger.info('Pseudo samples for NuScenes dataset: %d' % (len(self.ps_infos)))


    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points
    

    def get_gt_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.gt_infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_ps_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.ps_infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        points = np.concatenate((points, times), axis=1)
        return points
    
    def get_ps_adv_lidar_with_sweeps(self, index, thres=0.3, max_sweeps=1, eps=0.001):
        info = self.ps_infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        p_dict = dict()
        p_dict['gt_boxes'] = info['gt_boxes'].copy()
        p_dict['p_score'] = info['p_score'].copy()
        box_mask = p_dict['p_score'] > thres
        p_dict['gt_boxes'] = p_dict['gt_boxes'][box_mask]
        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            p_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
        perturb = info['p_voxel_perturb'][0]
        pts_voxel_idx = get_point_voxel_idx(info, points, self.dataset_cfg['DATA_PROCESSOR'][2]['VOXEL_SIZE'], self.dataset_cfg['POINT_CLOUD_RANGE'])
        bbox_pts_idx = get_points_idx_per_bbox(p_dict, points)
        pts_perturb = get_points_perturb(perturb, pts_voxel_idx, bbox_pts_idx)

        rm_dict = {}
        for j in range(len(bbox_pts_idx)): 
            p_idx = bbox_pts_idx[j]
            if p_idx.dtype != int:
                p_idx = p_idx.astype('int64')
            p_perturb = pts_perturb[j].copy()
            if p_perturb.dtype != float:
                p_perturb = p_perturb.astype('float32')

            p_type = np.random.randint(3)
            if p_type == 0: # modify
                x_perturb = points[p_idx, :-1].copy() - eps * p_perturb

                # random sample
                if x_perturb.shape[0] > 0:
                    k = np.random.randint(x_perturb.shape[0])
                    rand_p_idx = np.arange(p_idx.shape[0])
                    np.random.shuffle(rand_p_idx)
                    rand_p_idx = rand_p_idx[k:]
                    points[p_idx[rand_p_idx], :-1] = x_perturb[rand_p_idx]

            elif p_type == 1: # add
                x_perturb = points[p_idx, :-1].copy() - eps * p_perturb

                # random sample
                if x_perturb.shape[0] > 0:
                    k = np.random.randint(x_perturb.shape[0])
                    rand_p_idx = np.arange(p_idx.shape[0])
                    np.random.shuffle(rand_p_idx)
                    rand_p_idx = rand_p_idx[k:]
                    new_intensity = points[p_idx[rand_p_idx], -1].copy()
                    new_intensity = new_intensity.reshape(-1, 1)
                    new_points = np.concatenate((x_perturb[rand_p_idx], new_intensity), axis=1)
                    points = np.concatenate((points, new_points))
                    new_times = times[p_idx[rand_p_idx]].copy()
                    times = np.concatenate((times, new_times))

            elif p_type == 2: # remove
                # random sample
                if len(p_idx) > 5:
                    k = np.random.randint(len(p_idx))
                    rand_p_idx = np.arange(len(p_idx))
                    np.random.shuffle(rand_p_idx)
                    rand_p_idx = rand_p_idx[k:]
                    rm_dict[j] = p_idx[rand_p_idx]

        if len(rm_dict.keys()):
            rm_array = np.array([], dtype='int64')
            for r_idx in rm_dict.keys():
                rm_array = np.concatenate((rm_array, rm_dict[r_idx]))
            points = np.delete(points, rm_array, axis=0)
            times = np.delete(times, rm_array, axis=0)

        points = np.concatenate((points, times), axis=1)
        return points

    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        if self.repeat:
            return len(self.gt_infos) * self.repeat
        else:
            return len(self.infos)

    
    def __getitem__(self, index):
        assert len(self.gt_infos) != 0
        assert len(self.ps_infos) != 0
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        prob = np.random.random(1)
        #  prob = 1
        if prob > self.dataset_cfg.MIXUP_PROB:
            gt_prob = np.random.random(1)
            #  gt_prob = 1
            if gt_prob < self.dataset_cfg.GT_PROB:
                info_adv = copy.deepcopy(self.gt_infos[index % len(self.gt_infos)])
                points_adv = self.get_gt_lidar_with_sweeps(index % len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                # for shift coor
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    points_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                info_org = copy.deepcopy(self.gt_infos[index % len(self.gt_infos)])
                points_org = self.get_gt_lidar_with_sweeps(index % len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                # for shift coor
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    points_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            else:
                info_adv = copy.deepcopy(self.ps_infos[index % len(self.ps_infos)])
                points_adv = self.get_ps_adv_lidar_with_sweeps(index % len(self.ps_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                info_adv, points_adv = self.pseudo_frame(info_adv, points_adv)
                info_org = copy.deepcopy(self.ps_infos[index % len(self.ps_infos)])
                points_org = self.get_ps_lidar_with_sweeps(index % len(self.ps_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                info_org, points_org = self.pseudo_frame(info_org, points_org)

            input_dict_adv = {
                'points': points_adv, 
                'frame_id': Path(info_adv['lidar_path']).stem, 
                'metadata': {'token': info_adv['token']}
            }
            input_dict_org = {
                'points': points_org, 
                'frame_id': Path(info_org['lidar_path']).stem, 
                'metadata': {'token': info_org['token']}
            }

            if 'gt_boxes' in info_adv:
                if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (info_adv['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                input_dict_adv.update({
                    'gt_names': info_adv['gt_names'] if mask is None else info_adv['gt_names'][mask],
                    'gt_boxes': info_adv['gt_boxes'] if mask is None else info_adv['gt_boxes'][mask]
                })

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    input_dict_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR


            if 'gt_boxes' in info_org:
                if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (info_org['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                input_dict_org.update({
                    'gt_names': info_org['gt_names'] if mask is None else info_org['gt_names'][mask],
                    'gt_boxes': info_org['gt_boxes'] if mask is None else info_org['gt_boxes'][mask]
                })

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    input_dict_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            data_dict_adv, data_dict_org = self.prepare_data(input_dict_adv, input_dict_org)

            if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
                gt_boxes = data_dict_adv['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict_adv['gt_boxes'] = gt_boxes
                gt_boxes = data_dict_org['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict_org['gt_boxes'] = gt_boxes

            if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict_adv:
                data_dict_adv['gt_boxes'] = data_dict_adv['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
                data_dict_org['gt_boxes'] = data_dict_org['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]


        else:
            if self.dataset_cfg.MIXUP_TYPE == 'only_gt':
                idx1 = np.random.randint(len(self.gt_infos))
                idx2 = np.random.randint(len(self.gt_infos))
                nus_info_1 = copy.deepcopy(self.gt_infos[idx1])
                nus_info_2 = copy.deepcopy(self.gt_infos[idx2])
                nus_points_1 = self.get_gt_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                nus_points_2 = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
            elif self.dataset_cfg.MIXUP_TYPE == 'ps_gt':
                idx1 = np.random.randint(len(self.ps_infos))
                idx2 = np.random.randint(len(self.gt_infos))
                nus_info_1 = copy.deepcopy(self.ps_infos[idx1])
                nus_info_2 = copy.deepcopy(self.gt_infos[idx2])
                nus_points_1 = self.get_ps_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                nus_points_2 = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
            elif self.dataset_cfg.MIXUP_TYPE == 'gt_gt+ps':
                idx1 = np.random.randint(len(self.gt_infos))
                idx2 = np.random.randint(len(self.infos))
                nus_info_1_adv = copy.deepcopy(self.gt_infos[idx1])
                nus_info_2_adv = copy.deepcopy(self.infos[idx2])
                nus_points_1_adv = self.get_gt_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    nus_points_1_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                if idx2 < len(self.gt_infos):
                    nus_points_2_adv = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_2_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_2_adv = self.get_ps_adv_lidar_with_sweeps(idx2-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                nus_info_1_org = copy.deepcopy(self.gt_infos[idx1])
                nus_info_2_org = copy.deepcopy(self.infos[idx2])
                nus_points_1_org = self.get_gt_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    nus_points_1_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                if idx2 < len(self.gt_infos):
                    nus_points_2_org = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_2_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_2_org = self.get_ps_lidar_with_sweeps(idx2-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
            elif self.dataset_cfg.MIXUP_TYPE == 'gt+ps_gt+ps':
                idx1 = np.random.randint(len(self.infos))
                idx2 = np.random.randint(len(self.infos))
                nus_info_1_adv = copy.deepcopy(self.infos[idx1])
                nus_info_2_adv = copy.deepcopy(self.infos[idx2])
                if idx1 < len(self.gt_infos):
                    nus_points_1_adv = self.get_gt_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_1_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_1_adv = self.get_ps_adv_lidar_with_sweeps(idx1-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                if idx2 < len(self.gt_infos):
                    nus_points_2_adv = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_2_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_2_adv = self.get_ps_adv_lidar_with_sweeps(idx2-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                nus_info_1_org = copy.deepcopy(self.infos[idx1])
                nus_info_2_org = copy.deepcopy(self.infos[idx2])
                if idx1 < len(self.gt_infos):
                    nus_points_1_org = self.get_gt_lidar_with_sweeps(idx1, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_1_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_1_org = self.get_ps_lidar_with_sweeps(idx1-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                if idx2 < len(self.gt_infos):
                    nus_points_2_org = self.get_gt_lidar_with_sweeps(idx2, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        nus_points_2_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                else:
                    nus_points_2_org = self.get_ps_lidar_with_sweeps(idx2-len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
            elif self.dataset_cfg.MIXUP_TYPE == 'no_mixup':
                info = copy.deepcopy(self.gt_infos[index % len(self.gt_infos)])
                points = self.get_gt_lidar_with_sweeps(index % len(self.gt_infos), max_sweeps=self.dataset_cfg.MAX_SWEEPS)
                # for shift coor
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                    # self.logger.info(f'Shift coordinate: {self.dataset_cfg.SHIFT_COOR} done!')
                    

                input_dict = {
                    'points': points,
                    'frame_id': Path(info['lidar_path']).stem,
                    'metadata': {'token': info['token']}
                }

                if 'gt_boxes' in info:
                    if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                        mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                    else:
                        mask = None

                    input_dict.update({
                        'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                        'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
                    })

                    if self.dataset_cfg.get('SHIFT_COOR', None):
                        input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

                data_dict = self.prepare_data(data_dict=input_dict)

                if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
                    gt_boxes = data_dict['gt_boxes']
                    gt_boxes[np.isnan(gt_boxes)] = 0
                    data_dict['gt_boxes'] = gt_boxes

                if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
                    data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
                return data_dict
            else:
                raise NotImplementedError

            #  if self.dataset_cfg.get('SHIFT_COOR', None):
                #  nus_points_1[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                #  nus_points_2[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            nus_info_1_adv, nus_points_1_adv = self.pseudo_frame(nus_info_1_adv, nus_points_1_adv)
            nus_info_2_adv, nus_points_2_adv = self.pseudo_frame(nus_info_2_adv, nus_points_2_adv)
            nus_info_1_org, nus_points_1_org = self.pseudo_frame(nus_info_1_org, nus_points_1_org)
            nus_info_2_org, nus_points_2_org = self.pseudo_frame(nus_info_2_org, nus_points_2_org)

            input_dict_1_adv = {
                'points': nus_points_1_adv, 
                'frame_id': Path(nus_info_1_adv['lidar_path']).stem, 
                'metadata': {'token': nus_info_1_adv['token']}
            }
            
            input_dict_2_adv = {
                'points': nus_points_2_adv, 
                'frame_id': Path(nus_info_2_adv['lidar_path']).stem, 
                'metadata': {'token': nus_info_2_adv['token']}
            }
            
            
            
            assert 'gt_boxes' in nus_info_1_adv
            assert 'gt_boxes' in nus_info_2_adv
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask_1 = (nus_info_1_adv['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                mask_2 = (nus_info_2_adv['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask_1 = None
                mask_2 = None
                
            input_dict_1_adv.update({
                'gt_names': nus_info_1_adv['gt_names'] if mask_1 is None else nus_info_1_adv['gt_names'][mask_1], 
                'gt_boxes': nus_info_1_adv['gt_boxes'] if mask_1 is None else nus_info_1_adv['gt_boxes'][mask_1]
            })
            
            input_dict_2_adv.update({
                'gt_names': nus_info_2_adv['gt_names'] if mask_2 is None else nus_info_2_adv['gt_names'][mask_2], 
                'gt_boxes': nus_info_2_adv['gt_boxes'] if mask_2 is None else nus_info_2_adv['gt_boxes'][mask_2]
            })
            
            if not self.dataset_cfg.PRED_VELOCITY:
                input_dict_1_adv['gt_boxes'] = input_dict_1_adv['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]
                input_dict_2_adv['gt_boxes'] = input_dict_2_adv['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict_1_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
                input_dict_2_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict_1_org = {
                'points': nus_points_1_org, 
                'frame_id': Path(nus_info_1_org['lidar_path']).stem, 
                'metadata': {'token': nus_info_1_org['token']}
            }
            
            input_dict_2_org = {
                'points': nus_points_2_org, 
                'frame_id': Path(nus_info_2_org['lidar_path']).stem, 
                'metadata': {'token': nus_info_2_org['token']}
            }
             
            assert 'gt_boxes' in nus_info_1_org
            assert 'gt_boxes' in nus_info_2_org
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask_1 = (nus_info_1_org['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                mask_2 = (nus_info_2_org['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask_1 = None
                mask_2 = None
                
            input_dict_1_org.update({
                'gt_names': nus_info_1_org['gt_names'] if mask_1 is None else nus_info_1_org['gt_names'][mask_1], 
                'gt_boxes': nus_info_1_org['gt_boxes'] if mask_1 is None else nus_info_1_org['gt_boxes'][mask_1]
            })
            
            input_dict_2_org.update({
                'gt_names': nus_info_2_org['gt_names'] if mask_2 is None else nus_info_2_org['gt_names'][mask_2], 
                'gt_boxes': nus_info_2_org['gt_boxes'] if mask_2 is None else nus_info_2_org['gt_boxes'][mask_2]
            })
            
            if not self.dataset_cfg.PRED_VELOCITY:
                input_dict_1_org['gt_boxes'] = input_dict_1_org['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]
                input_dict_2_org['gt_boxes'] = input_dict_2_org['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]
            
            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict_1_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
                input_dict_2_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            data_dict_adv, data_dict_org = self.prepare_mixup_data(input_dict_1_adv, input_dict_2_adv, input_dict_1_org, input_dict_2_org)

            if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
                gt_boxes = data_dict_adv['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict_adv['gt_boxes'] = gt_boxes
                gt_boxes = data_dict_org['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict_org['gt_boxes'] = gt_boxes

            #  from pcdet.utils.simplevis import nuscene_vis
            #  gt_boxes = data_dict['gt_boxes'].copy()
            #  gt_boxes[:, 6] = -gt_boxes[:, 6]
            #  det = nuscene_vis(data_dict['points'], gt_boxes)
            #  cv2.imwrite('/home/spalab/hojun/hojun_ssl/ssda/ssda3d_perturb/tools/vis_ex/perturbed.png', det)

        return data_dict_adv, data_dict_org


    def prepare_mixup_data(self, data_dict_1, data_dict_2, data_dict_3, data_dict_4):
        if self.training:
            assert 'gt_boxes' in data_dict_1
            assert 'gt_boxes' in data_dict_2
            
            gt_boxes_mask_1 = np.array([n in self.class_names for n in data_dict_1['gt_names']], dtype=np.bool_)
            gt_boxes_mask_2 = np.array([n in self.class_names for n in data_dict_2['gt_names']], dtype=np.bool_)

            data_dict_1 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_1, 
                    'gt_boxes_mask': gt_boxes_mask_1
                }
            )
            
            data_dict_2 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_2, 
                    'gt_boxes_mask': gt_boxes_mask_2
                }
            )

            assert 'gt_boxes' in data_dict_3
            assert 'gt_boxes' in data_dict_4
            
            gt_boxes_mask_3 = np.array([n in self.class_names for n in data_dict_3['gt_names']], dtype=np.bool_)
            gt_boxes_mask_4 = np.array([n in self.class_names for n in data_dict_4['gt_names']], dtype=np.bool_)

            data_dict_3['augmentation_list'] = copy.deepcopy(data_dict_1['augmentation_list'])
            data_dict_3['augmentation_params'] = copy.deepcopy(data_dict_1['augmentation_params'])

            data_dict_3 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_3, 
                    'gt_boxes_mask': gt_boxes_mask_3
                }
            )

            data_dict_4['augmentation_list'] = copy.deepcopy(data_dict_2['augmentation_list'])
            data_dict_4['augmentation_params'] = copy.deepcopy(data_dict_2['augmentation_params'])
            
            data_dict_4 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_4, 
                    'gt_boxes_mask': gt_boxes_mask_4
                }
            )

            gt_boxes_mask_3 = np.array([n in self.class_names for n in data_dict_3['gt_names']], dtype=np.bool_)
            gt_boxes_mask_4 = np.array([n in self.class_names for n in data_dict_4['gt_names']], dtype=np.bool_)

            data_dict_3.pop('augmentation_list')
            data_dict_3.pop('augmentation_params')


            data_dict_3 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_3, 
                    'gt_boxes_mask': gt_boxes_mask_3
                }
            )

            data_dict_4['augmentation_list'] = copy.deepcopy(data_dict_3['augmentation_list'])
            data_dict_4['augmentation_params'] = copy.deepcopy(data_dict_3['augmentation_params'])
        
            data_dict_4 = self.data_augmentor.forward(
                data_dict={
                    **data_dict_4, 
                    'gt_boxes_mask': gt_boxes_mask_4
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
        
        if data_dict_3.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_3['gt_names'], self.class_names)
            data_dict_3['gt_boxes'] = data_dict_3['gt_boxes'][selected]
            data_dict_3['gt_names'] = data_dict_3['gt_names'][selected]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict_3['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict_3['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            
            data_dict_3['gt_boxes'] = gt_boxes
            
            if data_dict_3.get('gt_boxes2d', None) is not None:
                data_dict_3['gt_boxes2d'] = data_dict_3['gt_boxes2d'][selected]
                
        if data_dict_4.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict_4['gt_names'], self.class_names)
            data_dict_4['gt_boxes'] = data_dict_4['gt_boxes'][selected]
            data_dict_4['gt_names'] = data_dict_4['gt_names'][selected]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict_4['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict_4['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            
            data_dict_4['gt_boxes'] = gt_boxes
            
            if data_dict_4.get('gt_boxes2d', None) is not None:
                data_dict_4['gt_boxes2d'] = data_dict_4['gt_boxes2d'][selected]

        if data_dict_1.get('points', None) is not None:
            data_dict_1 = self.point_feature_encoder.forward(data_dict_1)

        if data_dict_2.get('points', None) is not None:
            data_dict_2 = self.point_feature_encoder.forward(data_dict_2)

        if data_dict_3.get('points', None) is not None:
            data_dict_3 = self.point_feature_encoder.forward(data_dict_3)

        if data_dict_4.get('points', None) is not None:
            data_dict_4 = self.point_feature_encoder.forward(data_dict_4)
        
        # if self.dataset_cfg.MIX_TYPE == 'mixup':
        #     new_data_dict = pc_mixup(data_dict_1, data_dict_2, alpha=self.dataset_cfg.ALPHA)
        # elif self.dataset_cfg.MIX_TYPE == 'cutmix':
        #     new_data_dicts = random_patch_replacement(data_dict_1, data_dict_2, self.point_cloud_range)
        #     new_data_dict = new_data_dicts[1]
        #  data_dict = intra_domain_point_mixup(data_dict_1, data_dict_2, alpha=self.dataset_cfg.ALPHA)
        data_dict_adv = intra_domain_point_mixup_cd(data_dict_1, data_dict_2, alpha=self.dataset_cfg.ALPHA)
        data_dict_org = intra_domain_point_mixup_cd(data_dict_3, data_dict_4, alpha=self.dataset_cfg.ALPHA)

        if len(data_dict_adv['gt_boxes'].shape) != 2 or len(data_dict_org['gt_boxes'].shape) != 2:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict_adv = self.data_processor.forward(
            data_dict=data_dict_adv
        )

        data_dict_org = self.data_processor.forward(
            data_dict=data_dict_org
        )
        
        if self.training:
            if len(data_dict_adv['gt_boxes']) == 0 or len(data_dict_org['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)
        
        data_dict_adv.pop('gt_names', None)
        data_dict_org.pop('gt_names', None)
        
        return data_dict_adv, data_dict_org

        
