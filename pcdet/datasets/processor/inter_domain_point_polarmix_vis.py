# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Aoran Xiao, 09:43 2022/03/05
# Wish for world peace!

# Modified by Yecheol Kim,
# Convert label section formatted in Seamantic KITTI to make it compatible 
# with NuScenes

import copy

import numpy as np

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils.simplevis import nuscene_vis

def swap(pt1, pt2, start_angle, end_angle, label1, label2):
    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))

    # calculate horizontal angel for each center of gt bbox
    n_label1, n_label2 = label1.shape[0], label2.shape[0]
    yaw1 = -np.arctan2(label1[:, 1], label1[:, 0])
    yaw2 = -np.arctan2(label2[:, 1], label2[:, 0])
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    comp1 = np.setdiff1d(np.arange(n_label1), idx1[0])
    label1_out = label1[comp1]
    label1_out = np.concatenate((label1_out, label2[idx2]))
    comp2 = np.setdiff1d(np.arange(n_label2), idx2[0])
    label2_out = label2[comp2]
    label2_out = np.concatenate((label2_out, label1[idx1]))

    return pt1_out, pt2_out, label1_out, label2_out

def rotate_copy(pts, labels, Omega, labels2):
    # extract instance points
    pts_inst, labels_inst = [], labels
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(pts[:, :3], labels[:, :7])
    point_masks = point_masks.sum(0) != 0
    pts_inst = pts[point_masks]

    # rotate-copy
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    labels_exist = [labels_inst, labels2]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)

        new_labels_inst = copy.deepcopy(labels_inst)
        new_labels_inst[:, :3] = np.dot(labels_inst[:, :3], rot_mat)
        new_labels_inst[:, 6] += omega_j
        # check overlap between existing boxes
        overlap = iou3d_nms_utils.boxes_bev_iou_cpu(np.concatenate(labels_exist, axis=0)[:, :7],
                                                    new_labels_inst[:, :7])
        overlap_mask = overlap.sum(0) == 0
        new_labels_inst = new_labels_inst[overlap_mask]

        labels_copy.append(new_labels_inst)
        labels_exist.append(new_labels_inst)

    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

def polarmix(pts1, labels1, pts2, labels2, alpha, beta, Omega):
    """
    Args:
        pts1: source domain points
        labels1: source domain labels (gt bbox)
        pts2: target domain points
        labels2: target domain labels (gt bbox)
        alpha (float): start angle
        beta (float): end angle 
        Omega (List[float]): Instance-level rotate-pasting angles

    Return:
        pts_out: polarmix points
        labels_out: polarmix labels
    """
    pts_out, labels_out = pts1, labels1
    # swapping
    if np.random.random() < 1.0:
        pts_out, _, labels_out, _ = swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1, label2=labels2)

    # rotate-pasting
    if np.random.random() < 0.0:
        # rotate-copy
        pts_copy, labels_copy = rotate_copy(pts2, labels2, Omega, labels_out)
        # paste
        pts_out = np.concatenate((pts_out, pts_copy), axis=0)
        labels_out = np.concatenate((labels_out, labels_copy), axis=0)

    return pts_out, labels_out

def inter_domain_point_polarmix(data_dict_source, data_dict_target, polarmix_rot_copy_num):
    if True:
        import cv2
        points = copy.deepcopy(data_dict_source['points'])
        gt_boxes = copy.deepcopy(data_dict_source['gt_boxes'])
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        det = nuscene_vis(points, gt_boxes)
        cv2.imwrite('test_1before_sour.png', det)
        points_targ = copy.deepcopy(data_dict_target['points'])
        gt_boxes_targ = copy.deepcopy(data_dict_target['gt_boxes'])
        gt_boxes_targ[:, 6] = -gt_boxes_targ[:, 6]
        det = nuscene_vis(points_targ, gt_boxes_targ)
        cv2.imwrite('test_1before_targ.png', det)

    alpha = (np.random.random() - 1) * np.pi
    beta = alpha + np.pi
    Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
    Omega = Omega[:polarmix_rot_copy_num]
    pts_out, labels_out = polarmix(data_dict_source['points'],
                                   data_dict_source['gt_boxes'],
                                   data_dict_target['points'],
                                   data_dict_target['gt_boxes'],
                                   alpha, beta, Omega
                                   )
    cutmixed_data = copy.deepcopy(data_dict_target)
    cutmixed_data['points'] = pts_out
    cutmixed_data['gt_boxes'] = labels_out

    if True:
        import cv2
        points = cutmixed_data['points']
        gt_boxes = cutmixed_data['gt_boxes']
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        det = nuscene_vis(points, gt_boxes)
        cv2.imwrite('test_2after_polarmix.png', det)
        breakpoint()

    return cutmixed_data
    
