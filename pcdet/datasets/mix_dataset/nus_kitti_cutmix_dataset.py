import os.path
import numpy as np
import pickle
import copy
from pathlib import Path

from pcdet.datasets import CutMixDatasetTemplate
from pcdet.utils import box_utils

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti

class NusKittiCutMixDataset(CutMixDatasetTemplate):
    def __init__(self, dataset_cfg=None, training=True, dataset_names=None, logger=None):
        super().__init__(dataset_cfg, training, dataset_names, logger)

        self.nus_infos = []
        self.kitti_infos = []

        # for NuScenes
        self.include_nuscenes_data(self.mode)

        # for Kitti
        self.include_kitti_data(self.mode)
        self.kitti_split = self.dataset_cfg['KittiDataset'].DATA_SPLIT[self.mode]
        self.kitti_root_split_path = self.root_path_target / ('training' if self.kitti_split != 'test' else 'testing')

        self.logger.info('Total samples for Kitti: %d' % (len(self.kitti_infos)))
        self.logger.info('Total samples for NuScenes: %d' % (len(self.nus_infos)))

    # for nus
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes Dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg['NuScenesDataset'].INFO_PATH[mode]:
            info_path = self.root_path_source / info_path
            if not info_path.exists():
                self.logger.info(f'NuScenesDataset info path: {info_path} doesnt exist!')
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.nus_infos.extend(nuscenes_infos)

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.nus_infos[index % len(self.nus_infos)]
        lidar_path = self.root_path_source / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps-1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path_source / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points)))
            )[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_calib(self, idx):
        calib_file = self.kitti_root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg['KittiDataset'].INFO_PATH[mode]:
            info_path = self.root_path_target / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def get_lidar(self, idx):
        lidar_file = self.kitti_root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file

        return sequence_file

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return (len(self.kitti_infos) + len(self.nus_infos)) * self.total_epochs
        return len(self.kitti_infos) + len(self.nus_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index * (len(self.kitti_infos) + len(self.nus_infos))

        prob = np.random.random(1)
        if prob < self.dataset_cfg.CUTMIX_PROB:
            kitti_info = copy.deepcopy(self.kitti_infos[index % len(self.kitti_infos)])
            nus_info = copy.deepcopy(self.nus_infos[index % len(self.nus_infos)])

            # for nus
            nus_points = self.get_lidar_with_sweeps(index % len(self.nus_infos), max_sweeps=self.dataset_cfg['NuScenesDataset'].MAX_SWEEPS)
            if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                nus_points[:, 0:3] += np.array(self.dataset_cfg['NuScenesDataset'].SHIFT_COOR, dtype=np.float32)
            nus_input_dict = {
                'points': nus_points,
                'frame_id': Path(nus_info['lidar_path']).stem,
                'metadata': {'token': nus_info['token']}
            }

            if 'gt_boxes' in nus_info:
                if self.dataset_cfg['NuScenesDataset'].get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (nus_info['num_lidar_pts'] > self.dataset_cfg['NuScenesDataset'].FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                nus_input_dict.update({
                    'gt_names': nus_info['gt_names'] if mask is None else nus_info['gt_names'][mask],
                    'gt_boxes': nus_info['gt_boxes'] if mask is None else nus_info['gt_boxes'][mask]
                })

                if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                    nus_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['NuScenesDataset'].SHIFT_COOR

                if self.dataset_cfg['NuScenesDataset'].get('SET_NAN_VELOCITY_TO_ZEROS', False):
                    gt_boxes = nus_input_dict['gt_boxes']
                    gt_boxes[np.isnan(gt_boxes)] = 0
                    nus_input_dict['gt_boxes'] = gt_boxes

                if not self.dataset_cfg['NuScenesDataset'].PRED_VELOCITY and 'gt_boxes' in nus_input_dict:
                    nus_input_dict['gt_boxes'] = nus_input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]


            # for kitti
            sample_idx = kitti_info['point_cloud']['lidar_idx']
            img_shape = kitti_info['image']['image_shape']
            calib = self.get_calib(sample_idx)
            get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

            kitti_input_dict = {
                'frame_id': sample_idx,
                'calib': calib,
            }

            if 'annos' in kitti_info:
                annos = kitti_info['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)


                kitti_input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if self.dataset_cfg['KittiDataset'].get('SHIFT_COOR', None):
                    kitti_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['KittiDataset'].SHIFT_COOR
                if "gt_boxes2d" in get_item_list:
                    kitti_input_dict['gt_boxes2d'] = annos["bbox"]

            if "points" in get_item_list:
                kitti_points = self.get_lidar(sample_idx)
                if self.dataset_cfg['KittiDataset'].FOV_POINTS_ONLY:
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]
                if self.dataset_cfg['KittiDataset'].get('SHIFT_COOR', None):
                    points[:, 0:3] += np.array(self.dataset_cfg['KittiDataset'].SHIFT_COOR, dtype=np.float32)
                kitti_input_dict['points'] = kitti_points

            if "images" in get_item_list:
                kitti_input_dict['images'] = self.get_image(sample_idx)

            if "depth_maps" in get_item_list:
                kitti_input_dict['depth_maps'] = self.get_depth_map(sample_idx)

            if "calib_matricies" in get_item_list:
                kitti_input_dict["trans_lidar_to_cam"], kitti_input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

            data_dict = self.prepare_data(nus_input_dict, kitti_input_dict)

            # if len(data_dict_list) != 2:
            #     new_index = np.random.randint(self.__len__())
            #     return self.__getitem__(new_index)

            # data_dict = data_dict_list[1]

        else:
            if index < len(self.nus_infos):
                nus_info = copy.deepcopy(self.nus_infos[index])
                nus_points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg['NuScenesDataset'].MAX_SWEEPS)
                if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                    nus_points[:, 0:3] += np.array(self.dataset_cfg['NuScenesDataset'].SHIFT_COOR, dtype=np.float32)

                nus_input_dict = {
                    'points': nus_points,
                    'frame_id': Path(nus_info['lidar_path']).stem,
                    'metadata': {'token': nus_info['token']}
                }

                if 'gt_boxes' in nus_info:
                    if self.dataset_cfg['NuScenesDataset'].get('FILTER_MIN_POINTS_IN_GT', False):
                        mask = (nus_info['num_lidar_pts'] > self.dataset_cfg['NuScenesDataset'].FILTER_MIN_POINTS_IN_GT - 1)
                    else:
                        mask = None

                    nus_input_dict.update({
                        'gt_names': nus_info['gt_names'] if mask is None else nus_info['gt_names'][mask],
                        'gt_boxes': nus_info['gt_boxes'] if mask is None else nus_info['gt_boxes'][mask]
                    })

                    if self.dataset_cfg['NuScenesDataset'].get('SHIFT_COOR', None):
                        nus_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['NuScenesDataset'].SHIFT_COOR

                    if self.dataset_cfg['NuScenesDataset'].get('SET_NAN_VELOCITY_TO_ZEROS', False):
                        gt_boxes = nus_input_dict['gt_boxes']
                        gt_boxes[np.isnan(gt_boxes)] = 0
                        nus_input_dict['gt_boxes'] = gt_boxes

                    if not self.dataset_cfg['NuScenesDataset'].PRED_VELOCITY and 'gt_boxes' in nus_input_dict:
                        nus_input_dict['gt_boxes'] = nus_input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

                data_dict = self.prepare_ori_data(nus_input_dict, source=True)

            else:
                kitti_info = copy.deepcopy(self.kitti_infos[index - len(self.nus_infos)])

                sample_idx = kitti_info['point_cloud']['lidar_idx']
                img_shape = kitti_info['image']['image_shape']
                calib = self.get_calib(sample_idx)
                get_item_list = self.dataset_cfg['KittiDataset'].get('GET_ITEM_LIST', ['points'])

                kitti_input_dict = {
                    'frame_id': sample_idx,
                    'calib': calib,
                }

                if 'annos' in kitti_info:
                    annos = kitti_info['annos']
                    annos = common_utils.drop_info_with_name(annos, name='DontCare')
                    loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                    gt_names = annos['name']
                    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                    gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

                    if self.dataset_cfg['KittiDataset'].get('SHIFT_COOR', None):
                        kitti_input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg['KittiDataset'].SHIFT_COOR

                    kitti_input_dict.update({
                        'gt_names': gt_names,
                        'gt_boxes': gt_boxes_lidar
                    })
                    if "gt_boxes2d" in get_item_list:
                        kitti_input_dict['gt_boxes2d'] = annos["bbox"]

                if "points" in get_item_list:
                    kitti_points = self.get_lidar(sample_idx)
                    if self.dataset_cfg['KittiDataset'].FOV_POINTS_ONLY:
                        pts_rect = calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                        points = points[fov_flag]
                    if self.dataset_cfg['KittiDataset'].get('SHIFT_COOR', None):
                        points[:, 0:3] += np.array(self.dataset_cfg['KittiDataset'].SHIFT_COOR, dtype=np.float32)
                    kitti_input_dict['points'] = kitti_points

                if "images" in get_item_list:
                    kitti_input_dict['images'] = self.get_image(sample_idx)

                if "depth_maps" in get_item_list:
                    kitti_input_dict['depth_maps'] = self.get_depth_map(sample_idx)

                if "calib_matricies" in get_item_list:
                    kitti_input_dict["trans_lidar_to_cam"], kitti_input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

                data_dict = self.prepare_ori_data(kitti_input_dict, source=False)

        return data_dict

