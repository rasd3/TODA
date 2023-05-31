#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import List, Sequence

import torch
import numpy as np

from pcdet.utils import box_utils

def nus_vis(points, boxes=None, img_dir='test.png'):
    import cv2
    from pcdet.utils.simplevis import nuscene_vis
    boxes_vis = copy.deepcopy(boxes)
    if boxes is not None:
        boxes_vis = boxes_vis[:, :7]
        boxes_vis[:, 6] = -boxes_vis[:, 6]
    det = nuscene_vis(points, boxes_vis)
    cv2.imwrite('%s.png' % img_dir, det)

def laser_mix_transform_sph(input_dict: dict, mix_results: dict,
                            pitch_angles: Sequence[float], num_areas: List[int],
                            order: int = 0):
    """LaserMix transform function. Modified from mmdetection3d 
    for spherical coordinate
    (segmentation -> detection)

    Args:
        input_dict (dict): Result dict from loading pipeline.
        mix_results (dict): Mixed dict picked from dataset.

    Returns:
        dict: output dict after transformation.
    """
    points = input_dict['points']
    boxes = input_dict['gt_boxes']
    mix_points = mix_results['points']
    mix_boxes = mix_results['gt_boxes']

    rho = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    pitch = np.arctan2(-1.8 + points[:, 2], rho)
    pitch = np.clip(pitch, pitch_angles[0] + 1e-5,
                    pitch_angles[1] - 1e-5)
    rho_box = np.sqrt(boxes[:, 0]**2 + boxes[:, 1]**2)
    pitch_box = np.arctan2(-1.8 + boxes[:, 2], rho_box)
    pitch_box = np.clip(pitch_box, pitch_angles[0] + 1e-5,
                        pitch_angles[1] - 1e-5)

    mix_rho = np.sqrt(mix_points[:, 0]**2 +
                      mix_points[:, 1]**2)
    mix_pitch = np.arctan2(-1.8 + mix_points[:, 2], mix_rho)
    mix_pitch = np.clip(mix_pitch, pitch_angles[0] + 1e-5,
                        pitch_angles[1] - 1e-5)
    mix_rho_box = np.sqrt(mix_boxes[:, 0]**2 +
                          mix_boxes[:, 1]**2)
    mix_pitch_box = np.arctan2(-1.8 + mix_boxes[:, 2], mix_rho_box)
    mix_pitch_box = np.clip(mix_pitch_box, pitch_angles[0] + 1e-5,
                            pitch_angles[1] - 1e-5)
 
    num_areas = np.random.choice(num_areas, size=1)[0]
    angle_list = np.linspace(pitch_angles[1], pitch_angles[0],
                             num_areas + 1)
    out_points = []
    out_boxes = []
    for i in range(num_areas):
        # convert angle to radian
        start_angle = angle_list[i + 1] / 180 * np.pi
        end_angle = angle_list[i] / 180 * np.pi
        if i % 2 == order:  # pick from original point cloud
            idx = (pitch > start_angle) & (pitch <= end_angle)
            out_points.append(points[idx])
            idx_b = (pitch_box > start_angle) & (pitch_box <= end_angle)
            out_boxes.append(boxes[idx_b])
        else:  # pickle from mixed point cloud
            idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
            out_points.append(mix_points[idx])
            idx_b = (mix_pitch_box > start_angle) & (mix_pitch_box <= end_angle)
            out_boxes.append(mix_boxes[idx_b])
    out_points = np.concatenate(out_points)
    out_boxes = np.concatenate(out_boxes)
    # perform modifications on target dict
    mixed_results = copy.deepcopy(mix_results)
    mixed_results['points'] = out_points
    mixed_results['gt_boxes'] = out_boxes
    return mixed_results

def laser_mix_transform_cyc(source_dict: dict, target_dict: dict,
                            num_areas: List[int], num_angles: int, pc_range: List[int], inc_method: str):
    def in_whitin_range(yaw_p, yaw_b, yaw_bc, dis_p, dis_b, dis_bc, pts, box, yaw_range, dis_range, P_ANG):
        yaw_p = np.copy(yaw_p + P_ANG)
        yaw_b = np.copy(yaw_b + P_ANG)
        yaw_bc = np.copy(yaw_bc + P_ANG)
        yaw_p[yaw_p > 3.141592] -= 6.283184
        yaw_b[yaw_b > 3.141592] -= 6.283184
        yaw_bc[yaw_bc > 3.141592] -= 6.283184
        yaw_p[yaw_p < -3.141592] += 6.283184
        yaw_b[yaw_b < -3.141592] += 6.283184
        yaw_bc[yaw_bc < -3.141592] += 6.283184

        if inc_method == 'center':
            idx_box = (yaw_b > yaw_range[0]) & (yaw_b <= yaw_range[1])
            idx_box = np.logical_and(idx_box, (dis_b > dis_range[0]) & (dis_b <= dis_range[1]))

            idx_pts = (yaw_p > yaw_range[0]) & (yaw_p <= yaw_range[1])
            idx_pts = np.logical_and(idx_pts, (dis_p > dis_range[0]) & (dis_p <= dis_range[1]))

            add_pts, add_box = pts[idx_pts], box[idx_box]
        elif inc_method =='corner_del':
            pts_c = np.copy(pts)
            idx_box_yo = np.any((yaw_bc > yaw_range[0]) & (yaw_bc <= yaw_range[1]), axis=1)
            idx_box_ya = np.all((yaw_bc > yaw_range[0]) & (yaw_bc <= yaw_range[1]), axis=1)
            idx_box_do = np.any((dis_bc > dis_range[0]) & (dis_bc <= dis_range[1]), axis=1)
            idx_box_da = np.all((dis_bc > dis_range[0]) & (dis_bc <= dis_range[1]), axis=1)
            idx_box_del = np.logical_or(idx_box_yo != idx_box_ya, idx_box_do != idx_box_da)
            idx_box = np.logical_and(idx_box_ya, idx_box_da)
            add_box = box[idx_box]

            idx_pts = (yaw_p > yaw_range[0]) & (yaw_p <= yaw_range[1])
            idx_pts = np.logical_and(idx_pts, (dis_p > dis_range[0]) & (dis_p <= dis_range[1]))
            add_pts = box_utils.remove_points_in_boxes3d(pts_c[idx_pts], box[idx_box_del][:, :7])

        return add_pts, add_box

    P_ANG = np.random.uniform(-3.141592, 3.141952)
    dis_range = np.linspace(0, pc_range[3], num_areas + 1)
    yaw_range = np.linspace(-np.pi, np.pi, num_angles + 1)

    s_pts = source_dict['points']
    s_box = source_dict['gt_boxes']
    t_pts = target_dict['points']
    t_box = target_dict['gt_boxes']

    s_yaw_p = -np.arctan2(s_pts[:, 1], s_pts[:, 0])
    t_yaw_p = -np.arctan2(t_pts[:, 1], t_pts[:, 0])
    s_dis_p = np.clip(np.sqrt(s_pts[:, 0]**2 + s_pts[:, 1]**2), 1e-05, pc_range[3] - 1e-05)
    t_dis_p = np.clip(np.sqrt(t_pts[:, 0]**2 + t_pts[:, 1]**2), 1e-05, pc_range[3] - 1e-05)

    s_yaw_b = -np.arctan2(s_box[:, 1], s_box[:, 0])
    t_yaw_b = -np.arctan2(t_box[:, 1], t_box[:, 0])
    s_dis_b = np.clip(np.sqrt(s_box[:, 0]**2 + s_box[:, 1]**2), 1e-05, pc_range[3] - 1e-05)
    t_dis_b = np.clip(np.sqrt(t_box[:, 0]**2 + t_box[:, 1]**2), 1e-05, pc_range[3] - 1e-05)
    s_corner = box_utils.boxes_to_corners_3d(s_box)[:, :, :2]
    t_corner = box_utils.boxes_to_corners_3d(t_box)[:, :, :2]
    s_yaw_bc = -np.arctan2(s_corner[:, :, 1], s_corner[:, :, 0])
    t_yaw_bc = -np.arctan2(t_corner[:, :, 1], t_corner[:, :, 0])
    s_dis_bc = np.clip(np.sqrt(s_corner[:, :, 0]**2 + s_corner[:, :, 1]**2), 1e-05, pc_range[3] - 1e-05)
    t_dis_bc = np.clip(np.sqrt(t_corner[:, :, 0]**2 + t_corner[:, :, 1]**2), 1e-05, pc_range[3] - 1e-05)

    start_domain = np.random.choice([0, 1])
    out_pts, out_box = [], []
    for i in range(len(yaw_range) - 1):
        idx = i % 2 + start_domain
        for j in range(len(dis_range) - 1):
            if idx % 2 == 0:
                pts_add, box_add = in_whitin_range(s_yaw_p, s_yaw_b, s_yaw_bc, s_dis_p, s_dis_b, s_dis_bc, s_pts, s_box,
                                                   [yaw_range[i], yaw_range[i+1]], [dis_range[j], dis_range[j+1]], P_ANG)
                out_pts.append(pts_add)
                out_box.append(box_add)
            else:
                pts_add, box_add = in_whitin_range(t_yaw_p, t_yaw_b, t_yaw_bc, t_dis_p, t_dis_b, t_dis_bc, t_pts, t_box,
                                                   [yaw_range[i], yaw_range[i+1]], [dis_range[j], dis_range[j+1]], P_ANG)
                out_pts.append(pts_add)
                out_box.append(box_add)
            idx += 1
    out_pts = np.concatenate(out_pts)
    out_box = np.concatenate(out_box)

    mixed_results = copy.deepcopy(target_dict)
    mixed_results['points'] = out_pts
    mixed_results['gt_boxes'] = out_box

    return mixed_results
            

def inter_domain_point_lasermix(data_dict_source, data_dict_target, pitch_angle, num_areas, num_angles, pc_range, inc_method):
    if num_angles is not None:
        lasermixed_data = laser_mix_transform_cyc(data_dict_source,
                                                  data_dict_target,
                                                  num_areas,
                                                  num_angles,
                                                  pc_range,
                                                  inc_method
                                                  )
    else:
        lasermixed_data = laser_mix_transform_sph(data_dict_source,
                                                  data_dict_target,
                                                  pitch_angle,
                                                  num_areas,
                                                  inc_method
                                                  )
    #  nus_vis(data_dict_source['points'], data_dict_source['gt_boxes'], 'vis_1.png')
    #  nus_vis(data_dict_target['points'], data_dict_target['gt_boxes'], 'vis_2.png')
    #  nus_vis(lasermixed_data['points'], lasermixed_data['gt_boxes'], 'vis_3.png')
    return lasermixed_data
