# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Aoran Xiao, 09:43 2022/03/05
# Wish for world peace!

# Modified by Yecheol Kim,
# Convert label section formatted in Seamantic KITTI to make it compatible 
# with NuScenes

import copy

import numpy as np

from pcdet.utils import box_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

def sig_polar(x):
    # change range 0~1 -> -1~1, hyperparameter alpha:6
    return 1/(1+np.exp(-6*(x*2-1)))

def is_overlap(x1, x2, x3, x4):
    if x2 < x1:
        x1, x2 = x2, x1
    if x4 < x3:
        x3, x4 = x4, x3
    
    if x2 < x3 or x4 < x1: 
        return False
    else:
        return True

def nus_vis(points, boxes=None, img_dir='test.png'):
    import cv2
    from pcdet.utils.simplevis import nuscene_vis
    boxes_vis = copy.deepcopy(boxes)
    if boxes is not None:
        boxes_vis = boxes_vis[:, :7]
        boxes_vis[:, 6] = -boxes_vis[:, 6]
    det = nuscene_vis(points, boxes_vis)
    cv2.imwrite('%s.png' % img_dir, det)

def swap(pt1, pt2, start_angle, end_angle, label1, label2, inc_method='center', use_pitch=False):
    # calculate horizontal angel for each center of gt bbox
    n_label1, n_label2 = label1.shape[0], label2.shape[0]
    if inc_method == 'center':
        yaw1 = -np.arctan2(label1[:, 1], label1[:, 0])
        yaw2 = -np.arctan2(label2[:, 1], label2[:, 0])
        idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
        idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))
    elif inc_method == 'corner' or inc_method == 'corner_del':
        corner1 = box_utils.boxes_to_corners_3d(label1)[:, :, :2]
        corner2 = box_utils.boxes_to_corners_3d(label2)[:, :, :2]
        yaw1 = -np.arctan2(corner1[:, :, 1], corner1[:, :, 0])
        yaw2 = -np.arctan2(corner2[:, :, 1], corner2[:, :, 0])
        idx1 = np.any((yaw1>start_angle) & (yaw1<end_angle), axis=1)
        idx2 = np.all((yaw2>start_angle) & (yaw2<end_angle), axis=1)
        if inc_method == 'corner_del':
            idx1_a = np.all((yaw1>start_angle) & (yaw1<end_angle), axis=1)
            idx2_a = np.any((yaw2>start_angle) & (yaw2<end_angle), axis=1)
            # when only part of corners are wiht in region remove points in bbox
            pt1 = box_utils.remove_points_in_boxes3d(pt1, label1[idx1 != idx1_a][:, :7])
            pt2 = box_utils.remove_points_in_boxes3d(pt2, label2[idx2 != idx2_a][:, :7])
        idx1, idx2 = np.where(idx1), np.where(idx2)
    else:
        NotImplementedError()
    comp1 = np.setdiff1d(np.arange(n_label1), idx1[0])
    label1_out = label1[comp1]
    label1_out = np.concatenate((label1_out, label2[idx2]))

    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    if use_pitch:
        dis1 = np.sqrt(np.power(pt1[:, 0], 2) + np.power(pt1[:, 1], 2))
        dis2 = np.sqrt(np.power(pt2[:, 0], 2) + np.power(pt2[:, 1], 2))
        mask1 = dis1 > 1
        mask2 = dis2 > 1
        pitch1 = -np.arctan2(pt1[:, 2], dis1)
        pitch2 = -np.arctan2(pt2[:, 2], dis2)
        pitch1_min, pitch1_max = pitch1[mask1].min(), pitch1[mask1].max()
        idx2_1 = np.where(((yaw2<start_angle) | (yaw2>end_angle)) & ((pitch2 < pitch1_min) | (pitch2 > pitch1_max)) & mask2)
        idx2_2 = np.where((yaw2>start_angle) & (yaw2<end_angle))
        #  if idx2_1[0].shape[0]:
            #  print('pitch: %d' % idx2_1[0].shape[0])
        idx2 = (np.hstack((idx2_1[0], idx2_2[0])))
    else:
        idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out2 = np.concatenate((pt1_out, pt2[idx2]))


    return pt1_out2, label1_out

def swap_with_range(pt1, pt2, start_angle, end_angle, label1, label2, pc_range):
    # pick distance threshold
    dis_th = np.random.random() * pc_range[3]

    # calculate horizontal angle and distance for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])
    dis1 = np.sqrt(pt1[:, 0] ** 2 + pt1[:, 1] ** 2)
    dis2 = np.sqrt(pt2[:, 0] ** 2 + pt2[:, 1] ** 2)

    # select points in sector
    idx1 = (yaw1>start_angle) & (yaw1<end_angle)
    idx2 = (yaw2>start_angle) & (yaw2<end_angle)
    if dis_th / pc_range[3] > 0.4:
        idx1 = np.logical_and(idx1, (dis1 < dis_th))
        idx2 = np.logical_and(idx2, (dis2 < dis_th))
    else:
        idx1 = np.logical_and(idx1, (dis1 > dis_th))
        idx2 = np.logical_and(idx2, (dis2 > dis_th))
    idx1, idx2 = np.where(idx1), np.where(idx2)

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))

    # calculate horizontal angel for each center of gt bbox
    n_label1, n_label2 = label1.shape[0], label2.shape[0]
    yaw1 = -np.arctan2(label1[:, 1], label1[:, 0])
    yaw2 = -np.arctan2(label2[:, 1], label2[:, 0])
    dis1 = np.sqrt(label1[:, 0] ** 2 + label1[:, 1] ** 2)
    dis2 = np.sqrt(label2[:, 0] ** 2 + label2[:, 1] ** 2)
    idx1 = (yaw1>start_angle) & (yaw1<end_angle)
    idx2 = (yaw2>start_angle) & (yaw2<end_angle)
    if dis_th / pc_range[3] > 0.4:
        idx1 = np.logical_and(idx1, (dis1 < dis_th))
        idx2 = np.logical_and(idx2, (dis2 < dis_th))
    else:
        idx1 = np.logical_and(idx1, (dis1 > dis_th))
        idx2 = np.logical_and(idx2, (dis2 > dis_th))
    idx1, idx2 = np.where(idx1), np.where(idx2)

    comp1 = np.setdiff1d(np.arange(n_label1), idx1[0])
    label1_out = label1[comp1]
    label1_out = np.concatenate((label1_out, label2[idx2]))
    comp2 = np.setdiff1d(np.arange(n_label2), idx2[0])
    label2_out = label2[comp2]
    label2_out = np.concatenate((label2_out, label1[idx1]))

    return pt1_out, label1_out
    
def rotate_copy(pts, labels, Omega, labels2):
    labels_inst = labels

    # rotate-copy
    pts_copy = []
    labels_copy = []
    labels_exist = [labels2]
    for omega_j in Omega:
        # rotate box
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
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

        # extract points
        point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(pts[:, :3], 
                                                                labels_inst[overlap_mask][:, :7])
        point_masks = point_masks.sum(0) != 0
        pts_inst = pts[point_masks]
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)

    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    return pts_copy, labels_copy

def polarmix(pts1, labels1, pts2, labels2, swap_range, Omega, pc_range, polar_dis, inc_method,
             use_pitch):
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
        for i in range(len(swap_range)):
            if polar_dis == 'FULL':
                pts_out, labels_out = swap(pts_out, pts2, 
                                           start_angle=swap_range[i][0],
                                           end_angle=swap_range[i][1],
                                           label1=labels_out, label2=labels2,
                                           inc_method=inc_method,
                                           use_pitch=use_pitch
                                           )
            elif polar_dis == 'RAND':
                pts_out, labels_out = swap_with_range(pts_out, pts2, 
                                                      start_angle=swap_range[i][0],
                                                      end_angle=swap_range[i][1],
                                                      label1=labels_out, label2=labels2, 
                                                      pc_range=pc_range, polar_dis=polar_dis)
        #  nus_vis(pts_out, labels_out, 'vis_1.png')
        #  nus_vis(pts1, labels1, 'vis_ori.png')
        #  nus_vis(pts2, labels2, 'vis_trg.png')
        #  print('PolarMix swep')

    # rotate-pasting
    if np.random.random() < 1.0:
        # rotate-copy
        pts_copy, labels_copy = rotate_copy(pts2, labels2, Omega, labels_out)
        # paste
        #  nus_vis(pts_out, labels_out, 'vis_1.png')
        pts_out = box_utils.remove_points_in_boxes3d(pts_out, labels_copy[:, :7])
        #  nus_vis(pts_out, labels_out, 'vis_2.png')
        pts_out = np.concatenate((pts_out, pts_copy), axis=0)
        labels_out = np.concatenate((labels_out, labels_copy), axis=0)
        #  nus_vis(pts_out, labels_out, 'vis_3.png')
        #  print('PolarMix rotate-pasting')

    return pts_out, labels_out

def inter_domain_point_polarmix(data_dict_source, data_dict_target, polarmix_rot_copy_num, polarmix_degree,
                                train_percent, update_methods, pc_range, polar_dis, inc_method, use_pitch):
    if isinstance(polarmix_degree, float):
        p_degree = [polarmix_degree, polarmix_degree]
    elif isinstance(polarmix_degree, list):
        if len(polarmix_degree) == 1:
            p_degree = [polarmix_degree[0], polarmix_degree[0]]
        else:
            p_degree = [polarmix_degree[0], polarmix_degree[1]]

    swap_range = []
    for update_method in update_methods:
        if update_method == 'FIX':
            prand_degree = p_degree[0]
        elif update_method == 'RAND':
            prand_degree = np.random.uniform(p_degree[0], p_degree[1])
        elif update_method == 'ASC':
            prand_degree = p_degree[0] + (p_degree[1] - p_degree[0]) * train_percent
        elif update_method == 'ASC_SIG':
            prand_degree = p_degree[0] + (p_degree[1] - p_degree[0]) * sig_polar(train_percent)
        elif update_method == 'DESC':
            prand_degree = p_degree[1] - (p_degree[1] - p_degree[0]) * train_percent
            
        num_swap = len(swap_range)
        for _ in range(100):
            swap_st = (np.random.random() * 2 - 1) * np.pi # -pi ~ pi
            ov_flag = False
            for i in range(num_swap):
                ov_flag = is_overlap(swap_range[i][0], swap_range[i][1], swap_st, swap_st + prand_degree)
                if ov_flag:
                    break
            if ov_flag == False:
                swap_range.append([swap_st, swap_st + prand_degree])
                break

        num_swap = len(swap_range)
        for i in range(num_swap):
            if swap_range[i][1] > np.pi:
                swap_range.append([-np.pi, swap_range[i][1]-(np.pi*2)])
                swap_range[i][1] = np.pi

    Omega = [0, np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
    Omega = Omega[:polarmix_rot_copy_num]
    pts_out, labels_out = polarmix(data_dict_source['points'],
                                   data_dict_source['gt_boxes'],
                                   data_dict_target['points'],
                                   data_dict_target['gt_boxes'],
                                   swap_range, Omega, pc_range, polar_dis, inc_method, use_pitch
                                   )
    cutmixed_data = copy.deepcopy(data_dict_target)
    cutmixed_data['points'] = pts_out
    cutmixed_data['gt_boxes'] = labels_out

    return cutmixed_data

