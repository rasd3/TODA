#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import List, Sequence

import torch
import numpy as np

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
            print(idx.sum())
            idx_b = (pitch_box > start_angle) & (pitch_box <= end_angle)
            out_boxes.append(boxes[idx_b])
        else:  # pickle from mixed point cloud
            idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
            out_points.append(mix_points[idx])
            print(idx.sum())
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
                            num_areas: List[int], num_angles: int, pc_range: List[int]):
    dis_range = np.linspace(0, pc_range[3], num_areas + 1)
    yaw_range = np.linspace(-np.pi, np.pi, num_angles + 1)

    s_pts = source_dict['points']
    s_box = source_dict['gt_boxes']
    t_pts = target_dict['points']
    t_box = target_dict['gt_boxes']

    s_yaw_p = -np.arctan2(s_pts[:, 1], s_pts[:, 0])
    t_yaw_p = -np.arctan2(t_pts[:, 1], t_pts[:, 0])
    s_yaw_b = -np.arctan2(s_box[:, 1], s_box[:, 0])
    t_yaw_b = -np.arctan2(t_box[:, 1], t_box[:, 0])
    s_dis_p = np.clip(np.sqrt(s_pts[:, 0]**2 + s_pts[:, 1]**2), 0, pc_range[3])
    t_dis_p = np.clip(np.sqrt(t_pts[:, 0]**2 + t_pts[:, 1]**2), 0, pc_range[3])
    s_dis_b = np.clip(np.sqrt(s_box[:, 0]**2 + s_box[:, 1]**2), 0, pc_range[3])
    t_dis_b = np.clip(np.sqrt(t_box[:, 0]**2 + t_box[:, 1]**2), 0, pc_range[3])

    start_domain = np.random.choice([0, 1])
    out_pts, out_box = [], []
    for i in range(len(yaw_range) - 1):
        idx = i % 2 + start_domain
        for j in range(len(dis_range) - 1):
            if idx % 2 == 0:
                idx_pts = (s_yaw_p > yaw_range[i]) & (s_yaw_p <= yaw_range[i+1])
                idx_pts = np.logical_and(idx_pts, (s_dis_p > dis_range[j]) & (s_dis_p <= dis_range[j+1]))
                idx_box = (s_yaw_b > yaw_range[i]) & (s_yaw_b <= yaw_range[i+1])
                idx_box = np.logical_and(idx_box, (s_dis_b > dis_range[j]) & (s_dis_b <= dis_range[j+1]))
                out_pts.append(s_pts[idx_pts])
                out_box.append(s_box[idx_box])
            else:
                idx_pts = (t_yaw_p > yaw_range[i]) & (t_yaw_p <= yaw_range[i+1])
                idx_pts = np.logical_and(idx_pts, (t_dis_p > dis_range[j]) & (t_dis_p <= dis_range[j+1]))
                idx_box = (t_yaw_b > yaw_range[i]) & (t_yaw_b <= yaw_range[i+1])
                idx_box = np.logical_and(idx_box, (t_dis_b > dis_range[j]) & (t_dis_b <= dis_range[j+1]))
                out_pts.append(t_pts[idx_pts])
                out_box.append(t_box[idx_box])
            idx += 1
    out_pts = np.concatenate(out_pts)
    out_box = np.concatenate(out_box)

    mixed_results = copy.deepcopy(target_dict)
    mixed_results['points'] = out_pts
    mixed_results['gt_boxes'] = out_box

    return mixed_results
            

def inter_domain_point_lasermix(data_dict_source, data_dict_target, pitch_angle, num_areas, num_angles, pc_range):
    if num_angles is not None:
        lasermixed_data = laser_mix_transform_cyc(data_dict_source,
                                                  data_dict_target,
                                                  num_areas,
                                                  num_angles,
                                                  pc_range)
    else:
        lasermixed_data = laser_mix_transform_sph(data_dict_source,
                                                  data_dict_target,
                                                  pitch_angle,
                                                  num_areas,
                                                  )
    #  nus_vis(data_dict_source['points'], data_dict_source['gt_boxes'], 'vis_1.png')
    #  nus_vis(data_dict_target['points'], data_dict_target['gt_boxes'], 'vis_2.png')
    #  nus_vis(lasermixed_data['points'], lasermixed_data['gt_boxes'], 'vis_3.png')
    return lasermixed_data
