
import copy

import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

def nus_vis(points, boxes=None, img_dir='test.png'):
    import cv2
    from pcdet.utils.simplevis import nuscene_vis
    boxes_vis = copy.deepcopy(boxes)
    if boxes is not None:
        boxes_vis = boxes_vis[:, :7]
        boxes_vis[:, 6] = -boxes_vis[:, 6]
    det = nuscene_vis(points, boxes_vis)
    cv2.imwrite('%s.png' % img_dir, det)

def inter_domain_point_pseudobbox(data_source, data_target):
    s_points, s_gt_bbox = data_source['points'], data_source['gt_boxes']
    t_points, t_gt_bbox = data_target['points'], data_target['gt_boxes']
    #  nus_vis(s_points, s_gt_bbox, 'test_1source.png')
    #  nus_vis(t_points, t_gt_bbox, 'test_2target.png')

    # check overlap with source bbox & target bbox
    overlap = iou3d_nms_utils.boxes_bev_iou_cpu(s_gt_bbox[:, :7],
                                                t_gt_bbox[:, :7])
    overlap_mask = overlap.sum(0) == 0
    t_masked_gt_bbox = t_gt_bbox[overlap_mask]
    # gather target points in target gt bbox
    t_points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(t_points[:, :3],
                                                              t_masked_gt_bbox[:, :7])
    t_points_mask = t_points_mask.sum(0) != 0
    t_masked_points = t_points[t_points_mask]
    # remove overlap source points with target bbox
    s_points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(s_points[:, :3],
                                                              t_masked_gt_bbox[:, :7])
    s_points_mask = s_points_mask.sum(0) == 0
    s_masked_points = s_points[s_points_mask]

    mix_gt_bbox = np.concatenate([s_gt_bbox, t_masked_gt_bbox], axis=0)
    mix_points = np.concatenate([s_masked_points, t_masked_points], axis=0)
    data_target['gt_boxes'] = mix_gt_bbox
    data_target['points'] = mix_points
    #  nus_vis(mix_points, mix_gt_bbox, 'test_3mixed.png')

    return data_target

def inter_domain_point_pseudobackground(data_source, data_target):
    s_points, s_gt_bbox = data_source['points'], data_source['gt_boxes']
    t_points, t_gt_bbox = data_target['points'], data_target['gt_boxes']
    #  nus_vis(s_points, s_gt_bbox, 'test_1source.png')
    #  nus_vis(t_points, t_gt_bbox, 'test_2target.png')
    t_points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(t_points[:, :3],
                                                              t_gt_bbox[:, :7])
    t_points_mask = t_points_mask.sum(0) == 0
    t_back_points = t_points[t_points_mask]
    s_gt_points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(s_points[:, :3],
                                                                 s_gt_bbox[:, :7])
    s_gt_points_mask = s_gt_points_mask.sum(0) != 0
    s_gt_points = s_points[s_gt_points_mask]

    mix_points = np.concatenate([s_gt_points, t_back_points])
    data_target['gt_boxes'] = s_gt_bbox
    data_target['points'] = mix_points
    #  nus_vis(mix_points, s_gt_bbox, 'test_3mixed.png')

    return data_target
