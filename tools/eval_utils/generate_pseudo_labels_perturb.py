from pathlib import Path
import pickle
import time
import copy
import torch
import torch.nn as nn

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu, load_data_to_gpu_adv, model_fn_decorator_adv
from pcdet.utils import common_utils
from pcdet.utils.perturb_utils import * 

def generate_pseudo_label_samples(unlabel_infos_path, predict_dict, output_infos_path, score_thresh={'car': 0}):
    def check_gt_boxes(info):
        return 'gt_boxes' in info
    
    def check_gt_names(info):
        return 'gt_names' in info
    
    with open(unlabel_infos_path, 'rb') as f:
        unlabel_infos = pickle.load(f)
        
    print(f"total unlabel samples: {len(unlabel_infos)}")
    
    pseudo_results_map = {}
    num_gt_box = 0
    for idx, val_result in enumerate(predict_dict):
        pseudo_results_map[val_result["frame_id"]] = val_result
    
    for idx, raw_info in tqdm.tqdm(enumerate(unlabel_infos)):
        if check_gt_boxes(raw_info):
            unlabel_infos[idx].pop("gt_boxes")
        if check_gt_names(raw_info):
            unlabel_infos[idx].pop("gt_names")
        
        unlabel_infos[idx]["gt_boxes"] = np.array(0)
        unlabel_infos[idx]["gt_names"] = np.array(0)

        if 'lidar_path' in raw_info:
            pseudo_result = pseudo_results_map[Path(raw_info['lidar_path']).stem]
        elif 'point_cloud' in raw_info:
            pseudo_result = pseudo_results_map[raw_info['point_cloud']['lidar_idx']]

        #  if len(pseudo_result['name']) != len(pseudo_result['bbox_pts_idx']):
            #  continue
        #  if len(pseudo_result['score']) != len(pseudo_result['bbox_pts_idx']):
            #  continue

        if score_thresh is not None:
            sample_info = {'name': [], 'boxes_3d': [], 'p_score': []}
            #  sample_info = {'name': [], 'boxes_3d': [], 'pts_perturb': [], 'bbox_pts_idx':[]}
            for _, class_name in enumerate(score_thresh.keys()):
                this_class_mask = pseudo_result['name'] == class_name
                mask = pseudo_result['score'][this_class_mask] > score_thresh[class_name]
                name = pseudo_result['name'][this_class_mask][mask]
                score = pseudo_result['score'][this_class_mask][mask]
                boxes_3d = pseudo_result['boxes_lidar'][this_class_mask][mask]
                #  pts_perturb = pseudo_result['pts_perturb'][this_class_mask][mask]
                #  bbox_pts_idx = pseudo_result['bbox_pts_idx'][this_class_mask][mask]
                sample_info['name'].append(name)
                sample_info['boxes_3d'].append(boxes_3d)
                sample_info['p_score'].append(score)
                #  sample_info['pts_perturb'].append(pts_perturb)
                #  sample_info['bbox_pts_idx'].append(bbox_pts_idx)

            name = np.concatenate(sample_info['name'])
            boxes_3d = np.concatenate(sample_info['boxes_3d'])
            p_score = np.concatenate(sample_info['p_score'])
            #  pts_perturb = np.concatenate(sample_info['pts_perturb'])
            #  bbox_pts_idx = np.concatenate(sample_info['bbox_pts_idx'])
        
        else:
            name = pseudo_result['name']
            boxes_3d = pseudo_result['boxes_lidar']
            p_score = pseudo_result['score']
            #  pts_perturb = pseudo_result['pts_perturb']
            #  bbox_pts_idx = pseudo_result['bbox_pts_idx']

        num_gt_box += len(name)
        unlabel_infos[idx]['gt_names'] = name
        unlabel_infos[idx]['gt_boxes'] = boxes_3d
        unlabel_infos[idx]['p_score'] = p_score
        #  unlabel_infos[idx]['pts_perturb'] = pts_perturb
        #  unlabel_infos[idx]['bbox_pts_idx'] = bbox_pts_idx
        unlabel_infos[idx]['p_voxel_perturb'] = pseudo_result['p_voxel_perturb']
        unlabel_infos[idx]['p_voxel_coords'] = pseudo_result['p_voxel_coords']

    print("Total box num: %d" % num_gt_box)
    print("Total infos num: %d" % len(unlabel_infos))
    
    with open(output_infos_path, 'wb') as f:
        pickle.dump(unlabel_infos, f)
        
    print("NuScenes pseudo infos file is saved to %s" % (output_infos_path))


def inference_and_generate_pseudo_labes(cfg, args, model, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None, unlabel_infos_path=None, optimizer=None):
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    model_func = model_fn_decorator_adv()
    point_cloud_range = cfg['UNLABEL_DATA_CONFIG']['POINT_CLOUD_RANGE']
    voxel_size = cfg['UNLABEL_DATA_CONFIG']['DATA_PROCESSOR'][-1]['VOXEL_SIZE']

    logger.info('*************** INFERENCING UNLABELD INFOS *****************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    #  model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='test', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        batch_dict_adv = copy.deepcopy(batch_dict)
        model.eval()
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, 
            output_path=result_dir if save_to_file else None
        )
        model.train()

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.training = False
                if module.weight.requires_grad:
                    module.weight.retain_grad()
                if module.bias.requires_grad:
                    module.bias.retain_grad()

        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, batch_dict_adv)
        loss.backward()

        perturb = get_perturb(batch_dict_adv['voxels'], eps=1)
        pts_voxel_idx = get_point_voxel_indices(batch_dict_adv, voxel_size, point_cloud_range)
        bbox_pts_idx = get_points_idx_per_gt_box(batch_dict_adv, annos)
        annos, bbox_pts_idx = filtering_min_points_box(annos, bbox_pts_idx)
        pts_perturb = get_points_perturbation(perturb, pts_voxel_idx, bbox_pts_idx, batch_dict_adv['batch_size'], annos)

        optimizer.zero_grad()


        for j in range(batch_dict['batch_size']):
            annos[j]['pts_perturb'] = pts_perturb[j]
            annos[j]['bbox_pts_idx'] = bbox_pts_idx[j]
        
        det_annos += annos

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    
    logger.info('Average predicted number of objects(%d sample): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    logger.info('*************** Start to generate pseudo labels *****************')

    car_score = args.pseudo_thresh
    thresh_score = {'car': car_score,
                    'pedestrian': car_score,
                    'bicycle': car_score
                    }

    pseudo_file_name = 'score_' + str(thresh_score['car']) + '_' + str(unlabel_infos_path).split('/')[-1]
    pseudo_label_output_path = result_dir / Path(pseudo_file_name)

    generate_pseudo_label_samples(unlabel_infos_path=unlabel_infos_path, predict_dict=det_annos, output_infos_path=pseudo_label_output_path, score_thresh=thresh_score)

def inference_and_generate_pseudo_labes_k(cfg, args, model, dataloader, logger, dist_test=False, save_to_file=False, result_dir=None, unlabel_infos_path=None, optimizer=None):
    # result_dir.mkdir()
    
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    model_func = model_fn_decorator_adv()
    point_cloud_range = cfg['UNLABEL_DATA_CONFIG']['POINT_CLOUD_RANGE']
    voxel_size = cfg['UNLABEL_DATA_CONFIG']['DATA_PROCESSOR'][-1]['VOXEL_SIZE']

    logger.info('*************** INFERENCING UNLABELD INFOS *****************')
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='test', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        batch_dict_adv = copy.deepcopy(batch_dict)
        model.eval()
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names, 
            output_path=result_dir if save_to_file else None
        )
        model.train()

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.training = False
                if module.weight.requires_grad:
                    module.weight.retain_grad()
                if module.bias.requires_grad:
                    module.bias.retain_grad()

        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, batch_dict_adv)
        loss.backward()

        perturb = get_perturb(batch_dict_adv['voxels'], eps=1)
        pts_voxel_idx = get_point_voxel_indices(batch_dict_adv, voxel_size, point_cloud_range)
        bbox_pts_idx = get_points_idx_per_gt_box(batch_dict_adv, pred_dicts)
        pts_perturb, vox_coords = get_points_perturb_coords(perturb, pts_voxel_idx, bbox_pts_idx, batch_dict_adv)
        for j in range(batch_dict['batch_size']):
            annos[j]['p_voxel_perturb'] = pts_perturb
            annos[j]['p_voxel_coords'] = vox_coords

        optimizer.zero_grad()

        det_annos += annos

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    
    logger.info('Average predicted number of objects(%d sample): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    logger.info('*************** Start to generate pseudo labels *****************')

    car_score = args.pseudo_thresh
    thresh_score = {'Car': car_score
                    }

    pseudo_file_name = 'score_' + str(thresh_score['Car']) + '_' + str(unlabel_infos_path).split('/')[-1]
    pseudo_label_output_path = result_dir / Path(pseudo_file_name)

    generate_pseudo_label_samples(unlabel_infos_path=unlabel_infos_path, predict_dict=det_annos, output_infos_path=pseudo_label_output_path, score_thresh=thresh_score)
