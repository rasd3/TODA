from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_to_gpu_adv(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif key in ['voxels']:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
            batch_dict[key].requires_grad = True
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_adv():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu_adv(batch_dict)
        batch_dict, ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_cl():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, adv_batch_dict, org_batch_dict, dist):
        load_data_to_gpu(adv_batch_dict)
        load_data_to_gpu(org_batch_dict)

        if not dist:
            adv_batch_dict, adv_ret_dict, adv_tb_dict, adv_disp_dict = model(adv_batch_dict)
            org_batch_dict, org_ret_dict, org_tb_dict, org_disp_dict = model(org_batch_dict)
            model_cfg = model.model_cfg
        else:
            (adv_batch_dict, adv_ret_dict, adv_tb_dict, adv_disp_dict), \
            (org_batch_dict, org_ret_dict, org_tb_dict, org_disp_dict) \
                                                    = model(adv_batch_dict, org_batch_dict)
            model_cfg = model.module.onepass.model_cfg

        loss_adv = adv_ret_dict['loss'].mean()
        loss_org = org_ret_dict['loss'].mean()

        adv_boxes = filter_boxes_secondiou(adv_batch_dict, model_cfg)
        org_boxes = filter_boxes_secondiou(org_batch_dict, model_cfg)

        #  adv_boxes = reverse_transform(adv_boxes, adv_batch_dict)
        org_boxes = reverse_transform(org_boxes, org_batch_dict)

        center_loss, size_loss = get_consistency_loss(adv_boxes, org_boxes)

        loss = loss_adv + loss_org + 0.1 * (center_loss + size_loss)

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, adv_tb_dict, adv_disp_dict)

    return model_func

def random_world_flip(box_preds, params, reverse=False):
    if reverse:
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
    else:
        if 'x' in params:
            box_preds[:, 1] = -box_preds[:, 1]
            box_preds[:, 6] = -box_preds[:, 6]
        if 'y' in params:
            box_preds[:, 0] = -box_preds[:, 0]
            box_preds[:, 6] = -(box_preds[:, 6] + np.pi)
    return box_preds

def random_world_rotation(box_preds, params, reverse=False):
    if reverse:
        noise_rotation = -params
    else:
        noise_rotation = params

    angle = torch.tensor([noise_rotation]).to(box_preds.device)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(1)
    ones = angle.new_ones(1)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(3, 3).float()
    box_preds[:, :3] = torch.matmul(box_preds[:, :3], rot_matrix)
    box_preds[:, 6] += noise_rotation
    return box_preds

def random_world_scaling(box_preds, params, reverse=False):
    if reverse:
        noise_scale = 1.0/params
    else:
        noise_scale = params

    box_preds[:, :6] *= noise_scale
    return box_preds

@torch.no_grad()
def forward_transform(boxes, batch_dict):
    augmentation_functions = {
        'random_world_flip': random_world_flip,
        'random_world_rotation': random_world_rotation,
        'random_world_scaling': random_world_scaling
    }
    for bs_idx, box in enumerate(boxes):
        aug_list = batch_dict['augmentation_list'][bs_idx]
        aug_param = batch_dict['augmentation_params'][bs_idx]
        box_preds = box['pred_boxes']
        for key in aug_list:
            if key == 'gt_sampling':
                continue
            aug_params = aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse=False)
        box['pred_boxes'] = box_preds

    return boxes

@torch.no_grad()
def reverse_transform(boxes, batch_dict):
    augmentation_functions = {
        'random_world_flip': random_world_flip,
        'random_world_rotation': random_world_rotation,
        'random_world_scaling': random_world_scaling
    }
    for bs_idx, box in enumerate(boxes):
        aug_list = batch_dict['augmentation_list'][bs_idx]
        aug_param = batch_dict['augmentation_params'][bs_idx]
        box_preds = box['pred_boxes']
        aug_list = aug_list[::-1]
        for key in aug_list:
            if key == 'gt_sampling':
                continue
            aug_params = aug_param[key]
            aug_func = augmentation_functions[key]
            box_preds = aug_func(box_preds, aug_params, reverse=True)
        box['pred_boxes'] = box_preds

    return boxes

def get_consistency_loss(adv_boxes, org_boxes):
    center_losses, size_losses = [], []
    batch_normalizer = 0
    for adv_box, org_box in zip(adv_boxes, org_boxes):
        adv_box_preds = adv_box['pred_boxes'].detach_()
        org_box_preds = org_box['pred_boxes'].detach_()
        num_adv_boxes = adv_box_preds.shape[0]
        num_org_boxes = org_box_preds.shape[0]
        if num_adv_boxes == 0 or num_org_boxes == 0:
            batch_normalizer += 1
            continue

        adv_centers, adv_sizes, adv_rot = adv_box_preds[:, :3], adv_box_preds[:, 3:6], adv_box_preds[:, 6]
        org_centers, org_sizes, org_rot = org_box_preds[:, :3], org_box_preds[:, 3:6], org_box_preds[:, 6]

        with torch.no_grad():
            MAX_DISTANCE = 10000000
            dist = adv_centers[:, None, :] - org_centers[None, :, :]
            dist = (dist ** 2).sum(-1)

            org_dist_of_adv, org_index_of_adv = dist.min(1)
            adv_dist_of_org, adv_index_of_org = dist.min(0)

            MATCHED_DISTANCE = 1
            matched_adv_mask = (adv_dist_of_org < MATCHED_DISTANCE).float().unsqueeze(-1)
            matched_org_mask = (org_dist_of_adv < MATCHED_DISTANCE).float().unsqueeze(-1)

        matched_adv_centers = adv_centers[adv_index_of_org]
        matched_org_centers = org_centers[org_index_of_adv]

        matched_adv_sizes = adv_sizes[adv_index_of_org]
        matched_org_sizes = org_sizes[org_index_of_adv]

        center_loss = (((adv_centers - matched_org_centers) * matched_org_mask).abs().sum()
                       + ((org_centers - matched_adv_centers) * matched_adv_mask).abs().sum()) \
                       / (num_adv_boxes + num_org_boxes)
        size_loss = ((F.mse_loss(matched_org_sizes, adv_sizes, reduction='none') * matched_org_mask).sum()
                      + ((F.mse_loss(matched_adv_sizes, org_sizes, reduction='none') * matched_adv_mask).sum())) \
                      / (num_adv_boxes + num_org_boxes)

        center_losses.append(center_loss)
        size_losses.append(size_loss)
        batch_normalizer += 1

    return sum(center_losses)/batch_normalizer, sum(size_losses)/batch_normalizer

def filter_boxes_secondiou(batch_dict, model_cfg):
    post_process_cfg = model_cfg.POST_PROCESSING
    batch_size = batch_dict['batch_size']
    recall_dict = {}
    pred_dicts = []
    for index in range(batch_size):
        if batch_dict.get('batch_index', None) is not None:
            assert batch_dict['batch_cls_preds'].shape.__len__() == 2
            batch_mask = (batch_dict['batch_index'] == index)
        else:
            assert batch_dict['batch_cls_preds'].shape.__len__() == 3
            batch_mask = index

        box_preds = batch_dict['batch_box_preds'][batch_mask]
        iou_preds = batch_dict['batch_cls_preds'][batch_mask]
        cls_preds = batch_dict['roi_scores'][batch_mask]

        src_iou_preds = iou_preds
        src_box_preds = box_preds
        src_cls_preds = cls_preds

        if not batch_dict['cls_preds_normalized']:
            iou_preds = torch.sigmoid(iou_preds)
            cls_preds = torch.sigmoid(cls_preds)

        iou_preds, label_preds = torch.max(iou_preds, dim=-1)
        label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1

        nms_scores = iou_preds

        selected, selected_scores = model_nms_utils.class_agnostic_nms(
            box_scores=nms_scores, box_preds=box_preds,
            nms_config=post_process_cfg.NMS_CONFIG,
            score_thresh=post_process_cfg.SCORE_THRESH
        )

        final_scores = selected_scores
        final_boxes = box_preds[selected]

        recall_dict = generate_recall_record_secondiou(
            box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
            recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
            thresh_list=post_process_cfg.RECALL_THRESH_LIST
        )

        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores
        }

        pred_dicts.append(record_dict)

    return pred_dicts

def filter_boxes_centerpoint(batch_dict, model_cfg):
    batch_size = batch_dict['batch_size']

    post_process_cfg = model_cfg.DENSE_HEAD.POST_PROCESSING
    post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

    ret_dict = [{
        'pred_boxes': [],
        'pred_scores': [],
        #  'pred_labels': [],
    } for k in range(batch_size)]
    for idx, pred_dict in enumerate(batch_dict['pred_dicts']):
        batch_hm = pred_dict['hm'].sigmoid()
        batch_center = pred_dict['center']
        batch_center_z = pred_dict['center_z']
        batch_dim = pred_dict['dim'].exp()
        batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
        batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
        #  batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

        final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
            heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
            center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=None,
            point_cloud_range=model_cfg.CL_CFG.POINT_CLOUD_RANGE, voxel_size=model_cfg.CL_CFG.VOXEL_SIZE,
            feature_map_stride=model_cfg.CL_CFG.FEATURE_MAP_STRIDE,
            K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
            circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
            score_thresh=post_process_cfg.SCORE_THRESH,
            post_center_limit_range=post_center_limit_range
        )

        for k, final_dict in enumerate(final_pred_dicts):
            #  final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
            #  if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                #  selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    #  box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                    #  nms_config=post_process_cfg.NMS_CONFIG,
                    #  score_thresh=None
                #  )
#
                #  final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                #  final_dict['pred_scores'] = selected_scores
                #  final_dict['pred_labels'] = final_dict['pred_labels'][selected]

            ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
            ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
            #  ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

    for k in range(batch_size):
        ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
        ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
        #  ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1
    return ret_dict

def generate_recall_record_secondiou(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
    if 'gt_boxes' not in data_dict:
        return recall_dict

    rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
    gt_boxes = data_dict['gt_boxes'][batch_index]

    if recall_dict.__len__() == 0:
        recall_dict = {'gt': 0}
        for cur_thresh in thresh_list:
            recall_dict['roi_%s' % (str(cur_thresh))] = 0
            recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

    cur_gt = gt_boxes
    k = cur_gt.__len__() - 1
    while k > 0 and cur_gt[k].sum() == 0:
        k -= 1
    cur_gt = cur_gt[:k + 1]

    if cur_gt.shape[0] > 0:
        if box_preds.shape[0] > 0:
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
        else:
            iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

        if rois is not None:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

        for cur_thresh in thresh_list:
            if iou3d_rcnn.shape[0] == 0:
                recall_dict['rcnn_%s' % str(cur_thresh)] += 0
            else:
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            if rois is not None:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

        recall_dict['gt'] += cur_gt.shape[0]
    else:
        gt_iou = box_preds.new_zeros(box_preds.shape[0])
    return recall_dict
