import copy
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from skimage import io

from pcdet.datasets.augmentor.augmentor_utils import *
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from ..processor.intra_domain_point_mixup import intra_domain_point_mixup
from ...utils.perturb_utils import *
from ..processor.intra_domain_point_mixup import intra_domain_point_mixup, intra_domain_point_mixup_cd
from ..dataset_cl import DatasetTemplateCL

class KittiMixUpAdvDataset(DatasetTemplateCL):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, pseudo_info_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.pseudo_info_path = pseudo_info_path
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '_0.01_1.txt')

        self.sample_id_list = [x.strip().split(' ')[0] for x in
                            open(split_dir).readlines()] if split_dir.exists() else None
        self.sample_index_list = [int(x.strip().split(' ')[1]) for x in
                            open(split_dir).readlines()] if split_dir.exists() else None

        self.gt_infos = []
        self.pseudo_infos = []
        self.include_kitti_data(self.mode)
        self.include_ps_data(self.mode)

        if self.training:
            all_train = len(self.gt_infos)
            self.unlabeled_index_list = list(set(list(range(all_train))) - set(self.sample_index_list))  # float()!!!
            self.ps_infos = []

            temp = []
            for i in self.sample_index_list:
                temp.append(self.gt_infos[int(i)])
            if len(self.sample_index_list) < 3712: # not 100%
                for i in self.unlabeled_index_list:
                    self.ps_infos.append(self.pseudo_infos[int(i)])
            else:
                self.unlabeled_index_list = list(range(len(self.sample_index_list)))
                for i in self.sample_index_list:
                    self.ps_infos.append(self.pseudo_infos[int(i)])
                print("full set", len(self.ps_infos))
            self.gt_infos = temp
            assert len(self.gt_infos) == len(self.sample_id_list)

        self.infos = self.gt_infos + self.ps_infos

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.infos)))
            self.logger.info('GT samples for KITTI dataset: %d' % (len(self.gt_infos)))
            self.logger.info('Pseudo samples for KITTI dataset: %d' % (len(self.ps_infos)))

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI labeled dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.gt_infos.extend(kitti_infos)

    def include_ps_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI unlabeled dataset')
        kitti_infos = []

        info_path = self.root_path / self.pseudo_info_path
        if not info_path.exists():
            self.logger.info('Error! Pseudo labels infos dont exist!')
            return 
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)

        self.pseudo_infos.extend(kitti_infos)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    
    #  def get_ps_adv_lidar_with_sweeps(self, index, thres=0.3, max_sweeps=1, eps=0.001):
        #  info = self.ps_infos[index]
        #  lidar_path = self.root_path / info['lidar_path']
        #  points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
#
        #  sweep_points_list = [points]
        #  sweep_times_list = [np.zeros((points.shape[0], 1))]
#
        #  for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            #  points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            #  sweep_points_list.append(points_sweep)
            #  sweep_times_list.append(times_sweep)
#
        #  points = np.concatenate(sweep_points_list, axis=0)
        #  times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    def get_adv_points(self, info, points, thres=0.2, eps=0.001):
        p_dict = dict()
        p_dict['gt_boxes'] = info['gt_boxes'].copy()
        p_dict['p_score'] = info['p_score'].copy()
        box_mask = p_dict['p_score'] > thres
        p_dict['gt_boxes'] = p_dict['gt_boxes'][box_mask]
        if self.dataset_cfg.get('SHIFT_COOR', None):
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
                    #  new_times = times[p_idx[rand_p_idx]].copy()
                    #  times = np.concatenate((times, new_times))

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
            #  times = np.delete(times, rm_array, axis=0)

        #  points = np.concatenate((points, times), axis=1)
        return points

    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        else:
            return len(self.infos)

    
    def __getitem__(self, index):
        assert len(self.gt_infos) != 0
        assert len(self.ps_infos) != 0
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        index = index % len(self.infos)

        prob = np.random.random(1)
        if prob > self.dataset_cfg.MIXUP_PROB:
            gt_prob = np.random.random(1)
            if gt_prob < self.dataset_cfg.GT_PROB:
                kitti_info_adv = copy.deepcopy(self.gt_infos[index % len(self.gt_infos)])
                sample_idx_adv = kitti_info_adv['point_cloud']['lidar_idx']
                img_shape_adv = kitti_info_adv['image']['image_shape']
                calib_adv = self.get_calib(sample_idx_adv)
                get_item_list_adv = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

                kitti_info_org = copy.deepcopy(self.gt_infos[index % len(self.gt_infos)])
                sample_idx_org = kitti_info_org['point_cloud']['lidar_idx']
                img_shape_org = kitti_info_org['image']['image_shape']
                calib_org = self.get_calib(sample_idx_org)
                get_item_list_org = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            else:
                kitti_info_adv = copy.deepcopy(self.ps_infos[index % len(self.ps_infos)])
                sample_idx_adv = kitti_info_adv['point_cloud']['lidar_idx']
                img_shape_adv = kitti_info_adv['image']['image_shape']
                calib_adv = self.get_calib(sample_idx_adv)
                get_item_list_adv = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

                kitti_info_org = copy.deepcopy(self.ps_infos[index % len(self.ps_infos)])
                sample_idx_org = kitti_info_org['point_cloud']['lidar_idx']
                img_shape_org = kitti_info_org['image']['image_shape']
                calib_org = self.get_calib(sample_idx_org)
                get_item_list_org = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

            kitti_input_dict_adv = {
                'frame_id': sample_idx_adv,
                'calib': calib_adv,
            }
            kitti_input_dict_org = {
                'frame_id': sample_idx_org,
                'calib': calib_org,
            }

            if 'gt_boxes' in kitti_info_adv:
                kitti_input_dict_adv.update({
                    'gt_names': kitti_info_adv['gt_names'],
                    'gt_boxes': kitti_info_adv['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            elif 'annos' in kitti_info_adv:
                annos = kitti_info_adv['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_adv)


                kitti_input_dict_adv.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_adv:
                    kitti_input_dict_adv['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if 'gt_boxes' in kitti_info_org:
                kitti_input_dict_org.update({
                    'gt_names': kitti_info_org['gt_names'],
                    'gt_boxes': kitti_info_org['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            elif 'annos' in kitti_info_org:
                annos = kitti_info_org['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_org)


                kitti_input_dict_org.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_org:
                    kitti_input_dict_org['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if "points" in get_item_list_adv:
                kitti_points_adv = self.get_lidar(sample_idx_adv)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_adv.lidar_to_rect(kitti_points_adv[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_adv, calib_adv)
                    kitti_points_adv = kitti_points_adv[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                if gt_prob >= self.dataset_cfg.GT_PROB:
                    kitti_points_adv = self.get_adv_points(kitti_info_adv, kitti_points_adv)
                kitti_input_dict_adv['points'] = kitti_points_adv

            if "points" in get_item_list_org:
                kitti_points_org = self.get_lidar(sample_idx_org)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_adv.lidar_to_rect(kitti_points_org[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_org, calib_org)
                    kitti_points_org = kitti_points_org[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                kitti_input_dict_org['points'] = kitti_points_org


            data_dict_adv, data_dict_org = self.prepare_data(kitti_input_dict_adv, kitti_input_dict_org)


        else:
            idx1 = np.random.randint(len(self.infos))
            idx2 = np.random.randint(len(self.infos))
            kitti_info_1_adv = copy.deepcopy(self.infos[idx1])
            kitti_info_2_adv = copy.deepcopy(self.infos[idx2])
            sample_idx_1_adv = kitti_info_1_adv['point_cloud']['lidar_idx']
            img_shape_1_adv = kitti_info_1_adv['image']['image_shape']
            calib_1_adv = self.get_calib(sample_idx_1_adv)
            get_item_list_1_adv = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            sample_idx_2_adv = kitti_info_2_adv['point_cloud']['lidar_idx']
            img_shape_2_adv = kitti_info_2_adv['image']['image_shape']
            calib_2_adv = self.get_calib(sample_idx_2_adv)
            get_item_list_2_adv = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            kitti_info_1_org = copy.deepcopy(self.infos[idx1])
            kitti_info_2_org = copy.deepcopy(self.infos[idx2])
            sample_idx_1_org = kitti_info_1_org['point_cloud']['lidar_idx']
            img_shape_1_org = kitti_info_1_org['image']['image_shape']
            calib_1_org = self.get_calib(sample_idx_1_org)
            get_item_list_1_org = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            sample_idx_2_org = kitti_info_2_org['point_cloud']['lidar_idx']
            img_shape_2_org = kitti_info_2_org['image']['image_shape']
            calib_2_org = self.get_calib(sample_idx_2_org)
            get_item_list_2_org = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

            kitti_input_dict_1_adv = {
                'frame_id': sample_idx_1_adv,
                'calib': calib_1_adv,
            }
            kitti_input_dict_2_adv = {
                'frame_id': sample_idx_2_adv,
                'calib': calib_2_adv,
            }
            kitti_input_dict_1_org = {
                'frame_id': sample_idx_1_org,
                'calib': calib_1_org,
            }
            kitti_input_dict_2_org = {
                'frame_id': sample_idx_2_org,
                'calib': calib_2_org,
            }
            
            
            if 'gt_boxes' in kitti_info_1_adv:
                kitti_input_dict_1_adv.update({
                    'gt_names': kitti_info_1_adv['gt_names'],
                    'gt_boxes': kitti_info_1_adv['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_1_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            elif 'annos' in kitti_info_1_adv:
                annos = kitti_info_1_adv['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_1_adv)

                kitti_input_dict_1_adv.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_1_adv:
                    kitti_input_dict_1_adv['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_1_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if 'gt_boxes' in kitti_info_2_adv:
                kitti_input_dict_2_adv.update({
                    'gt_names': kitti_info_2_adv['gt_names'],
                    'gt_boxes': kitti_info_2_adv['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_2_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            elif 'annos' in kitti_info_2_adv:
                annos = kitti_info_2_adv['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_2_adv)

                kitti_input_dict_2_adv.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_2_adv:
                    kitti_input_dict_2_adv['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_2_adv['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            
            if 'gt_boxes' in kitti_info_1_org:
                kitti_input_dict_1_org.update({
                    'gt_names': kitti_info_1_org['gt_names'],
                    'gt_boxes': kitti_info_1_org['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_1_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            elif 'annos' in kitti_info_1_org:
                annos = kitti_info_1_org['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_1_org)

                kitti_input_dict_1_org.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_1_org:
                    kitti_input_dict_1_org['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_1_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if 'gt_boxes' in kitti_info_2_org:
                kitti_input_dict_2_org.update({
                    'gt_names': kitti_info_2_org['gt_names'],
                    'gt_boxes': kitti_info_2_org['gt_boxes']
                })
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_2_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR
            elif 'annos' in kitti_info_2_org:
                annos = kitti_info_2_org['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib_2_org)

                kitti_input_dict_2_org.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list_2_org:
                    kitti_input_dict_2_org['gt_boxes2d'] = annos["bbox"]

                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_input_dict_2_org['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if "points" in get_item_list_1_adv:
                kitti_points_1_adv = self.get_lidar(sample_idx_1_adv)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_1_adv.lidar_to_rect(kitti_points_1_adv[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_1_adv, calib_1_adv)
                    kitti_points_1_adv = kitti_points_1_adv[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_1_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                if idx1 >= len(self.gt_infos):
                    kitti_points_1_adv = self.get_adv_points(kitti_info_1_adv, kitti_points_1_adv)
                kitti_input_dict_1_adv['points'] = kitti_points_1_adv
            if "points" in get_item_list_2_adv:
                kitti_points_2_adv = self.get_lidar(sample_idx_2_adv)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_2_adv.lidar_to_rect(kitti_points_2_adv[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_2_adv, calib_2_adv)
                    kitti_points_2_adv = kitti_points_2_adv[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_2_adv[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                if idx2 >= len(self.gt_infos):
                    kitti_points_2_adv = self.get_adv_points(kitti_info_2_adv, kitti_points_2_adv)
                kitti_input_dict_2_adv['points'] = kitti_points_2_adv

            if "points" in get_item_list_1_org:
                kitti_points_1_org = self.get_lidar(sample_idx_1_org)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_1_org.lidar_to_rect(kitti_points_1_org[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_1_org, calib_1_org)
                    kitti_points_1_org = kitti_points_1_org[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_1_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                kitti_input_dict_1_org['points'] = kitti_points_1_org
            if "points" in get_item_list_2_org:
                kitti_points_2_org = self.get_lidar(sample_idx_2_org)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = calib_2_org.lidar_to_rect(kitti_points_2_org[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape_2_org, calib_2_org)
                    kitti_points_2_org = kitti_points_2_org[fov_flag]
                if self.dataset_cfg.get('SHIFT_COOR', None):
                    kitti_points_2_org[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
                kitti_input_dict_2_org['points'] = kitti_points_2_org


            data_dict_adv, data_dict_org = self.prepare_mixup_data(kitti_input_dict_1_adv, kitti_input_dict_2_adv, kitti_input_dict_1_org, kitti_input_dict_2_org)


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

        
