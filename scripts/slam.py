import argparse
import os
import shutil
import subprocess
import sys
import time
import pandas as pd
from importlib.machinery import SourceFileLoader
import torch.nn as nn
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import utils.Tri_Plane_help as Tri_Plane_help
from utils.decoders import Decoders
from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera,to_gsplat_camera
from gsplat import rasterization as rasterization
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transformed_semantics2rendervar, transform_to_frame, l1_loss_v1, matrix_to_quaternion,params_to_gsplat_inputs
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

# from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from utils.sem_help import label2map


def save_decoder_and_planes(decoder, all_planes, output_dir, prefix: str = ""):
    """
    Save tri-plane decoder weights and plane tensors alongside Gaussian params.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Decoder weights
    if decoder is not None:
        dec_path = os.path.join(
            output_dir,
            f"{prefix}decoder.pth" if prefix != "" else "decoder.pth",
        )
        torch.save(decoder.state_dict(), dec_path)
    # Tri-planes
    if all_planes is not None:
        planes_to_save = {
            f"group_{gi}": [
                p.detach().cpu() if isinstance(p, torch.Tensor) else p
                for p in plane_list
            ]
            for gi, plane_list in enumerate(all_planes)
        }
        planes_path = os.path.join(
            output_dir,
            f"{prefix}planes.pth" if prefix != "" else "planes.pth",
        )
        torch.save(planes_to_save, planes_path)


def run_pointcloud_subprocess(config, output_dir, num_frames, device):
    pointcloud_cfg = config.get("pointcloud", {})
    experiment_path = config.get("_saved_experiment_path") or config.get("_experiment_path")
    if experiment_path is None:
        raise ValueError("Missing experiment path for point cloud generation.")

    cmd = [
        sys.executable,
        os.path.join(_BASE_DIR, "scripts", "gen_pointcloud_tsdf.py"),
        experiment_path,
        "--params",
        os.path.join(output_dir, "params.npz"),
        "--out_dir",
        pointcloud_cfg.get("out_dir", os.path.join(output_dir, "tsdf_pointcloud")),
        "--num_frames",
        str(pointcloud_cfg.get("num_frames", num_frames)),
        "--eval_every",
        str(pointcloud_cfg.get("eval_every", 4)),
        "--scale",
        str(pointcloud_cfg.get("scale", 1.0)),
        "--depth_trunc",
        str(pointcloud_cfg.get("depth_trunc", 30.0)),
        "--depth_clip_max",
        str(pointcloud_cfg.get("depth_clip_max", 10.0)),
        "--device",
        str(pointcloud_cfg.get("device", device)),
    ]
    print("Generating point cloud with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=_BASE_DIR, check=True)


def decode_gaussian_attributes(params, decoder, all_planes, use_rgb_Triplane=False,
                               decode_semantics=False):
    if decoder is None:
        return params
    pts = params['means3D']
    sem_feat = params.get('semantic_features', None)
    out = decoder(
        pts,
        all_planes,
        decode_rgb=use_rgb_Triplane,
        decode_sem=decode_semantics,
        sem_feat=sem_feat,
    )
    if use_rgb_Triplane and 'rgb_colors' in out:
        params['rgb_colors'] = out['rgb_colors']
    if decode_semantics and 'semantic_id' in out:
        params['semantic_id'] = out['semantic_id']
    return params

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective", device="cuda",
                   load_semantics=False, semantic_id=None, semantic_color=None,
                   num_semantic_classes=None):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).to(device).float(), 
                                    torch.arange(height).to(device).float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).to(device).float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)
    
    # Concat semantic label if load_semantics=True
    if load_semantics:
        if num_semantic_classes is not None and num_semantic_classes > 0:
            semantic_map, num_classes = label2map(semantic_id, num_semantic=num_semantic_classes, device=device)
        else:
            semantic_map, num_classes = label2map(semantic_id, num_semantic=-1, device=device)
        semantic_feats = torch.permute(semantic_map, (1, 2, 0)).reshape(-1, num_classes)
        semantic_color = torch.permute(semantic_color, (1, 2, 0)).reshape(-1, 3) # (3, H, W) -> (H, W, 3) -> (H * W, 3)
        point_cld = torch.cat((point_cld, semantic_feats, semantic_color), -1)

    # Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, device,
                      load_semantics=False, use_one_hot_semantics=False,
                      use_rgb_Triplane=False, use_sem_Triplane=False,
                      feature_3dgs=False, semantic_feature_dim=32,
                      num_semantic_classes=None):
    num_pts = init_pt_cld.shape[0]
    # channel 0-2 for 3d axis
    means3D = init_pt_cld[:, :3]
    # channel 3-5 for rgb colors
    rgb_colors = init_pt_cld[:, 3:6]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    
    
    if use_rgb_Triplane:
        params = {
        'means3D': means3D,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        }
    else:
        params = {
        'means3D': means3D,
        'rgb_colors': rgb_colors,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        }
    
    
    
    params_opt_exclude = set()
    if use_one_hot_semantics and (not use_sem_Triplane) and (not feature_3dgs):
        params_opt_exclude.add('semantic_id')
        
    if load_semantics:
        if use_one_hot_semantics:
            if feature_3dgs:
                if num_semantic_classes is not None and num_semantic_classes > 0:
                    sem_dim = num_semantic_classes
                else:
                    sem_dim = init_pt_cld.shape[1] - 9
                params['semantic_features'] = torch.empty(
                    (num_pts, semantic_feature_dim), dtype=torch.float, device=device
                ).normal_(mean=0.0, std=0.01)
            elif not use_sem_Triplane:
                if num_semantic_classes is not None and num_semantic_classes > 0:
                    sem_dim = num_semantic_classes
                else:
                    sem_dim = init_pt_cld.shape[1] - 9
                params['semantic_id'] = init_pt_cld[:, 6:6 + sem_dim]
        else:
            params['semantic_id'] = init_pt_cld[:, 6]
        # Always keep semantic colors if available
        if use_one_hot_semantics and feature_3dgs:
            start = 6 + sem_dim
            params['semantic_colors'] = init_pt_cld[:, start:start + 3]
        else:
            params['semantic_colors'] = init_pt_cld[:, 7:10]

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    
    for k, v in params.items():
        # if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).to(device).float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).to(device).float()}

    return params, variables, params_opt_exclude


def initialize_optimizer(params, lrs_dict, tracking, use_planes=False, decoder=None,
                         all_planes=None, plane_lr_config=None, plane_lr_scale=1.0,use_one_hot_semantics=False):
    lrs = lrs_dict
    # keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    # print(keys)
    param_groups = []
    for k, v in params.items():
        if not isinstance(v, torch.nn.Parameter):
            continue
        lr_key = k
        if k == 'semantic_features' and 'semantic_id' in lrs:
            lr_key = 'semantic_id'
        if lr_key not in lrs:
            continue
        param_groups.append({'params': [v], 'name': k, 'lr': lrs[lr_key]})
    
    # print([g.get('name') for g in param_groups])
        
    if isinstance(plane_lr_config, dict):
        plane_lr_values = plane_lr_config.get('lr', plane_lr_config)
    else:
        plane_lr_values = {}
    plane_lr_scale = float(plane_lr_scale)

    if decoder is not None:
        decoder_params = list(decoder.parameters())
        if decoder_params:
            decoders_lr = float(plane_lr_values.get('decoders_lr', 0.0)) * plane_lr_scale
            param_groups.append({'params': decoder_params, 'name': 'decoders', 'lr': decoders_lr})

    if use_planes:

        if all_planes is not None:
            c_plane_groups = []
            s_plane_groups = []

            planes_tuple = list(all_planes)
            c_planes_lists = planes_tuple[0:3]
            s_planes_lists = planes_tuple[3:]


            for plane_list in c_planes_lists:
                for idx, plane in enumerate(plane_list):
                    if not isinstance(plane, torch.nn.Parameter):
                        plane = nn.Parameter(plane, requires_grad=True)
                        plane_list[idx] = plane
                    c_plane_groups.append(plane)

            for plane_list in s_planes_lists:
                for idx, plane in enumerate(plane_list):
                    if not isinstance(plane, torch.nn.Parameter):
                        plane = nn.Parameter(plane, requires_grad=True)
                        plane_list[idx] = plane
                    s_plane_groups.append(plane)

            if c_plane_groups:
                c_planes_lr = float(plane_lr_values.get('c_planes_lr', 0.0)) * plane_lr_scale
                param_groups.append({'params': c_plane_groups, 'name': 'c_planes', 'lr': c_planes_lr})

            if s_plane_groups:
                s_planes_lr = float(plane_lr_values.get('s_planes_lr', plane_lr_values.get('planes_lr', 0.0))) * plane_lr_scale
                param_groups.append({'params': s_plane_groups, 'name': 'extra_planes', 'lr': s_planes_lr})

    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, device="cuda",
                              densify_dataset=None, load_semantics=False, use_one_hot_semantics=False,
                              use_rgb_Triplane=False, use_sem_Triplane=False,
                              feature_3dgs=False, semantic_feature_dim=32,
                              num_semantic_classes=None):
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[0]
    else:
        color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
    
    if load_semantics:
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) / 255 # (H, W, 3) -> (3, H, W)
    else:
        semantic_id = None
        semantic_color = None
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)
    gs_cam = to_gsplat_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                   w2c.detach().cpu().numpy(), device=device)
        densify_gs_cam = to_gsplat_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(),
                                          w2c.detach().cpu().numpy(), device=device)
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics,
                                                w2c, mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                load_semantics=load_semantics, semantic_id=semantic_id,
                                                semantic_color=semantic_color,
                                                num_semantic_classes=num_semantic_classes)

    # Initialize Parameters
    params, variables, params_opt_exclude = initialize_params(
        init_pt_cld, num_frames, mean3_sq_dist, device,
        load_semantics, use_one_hot_semantics,
        use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
        feature_3dgs=feature_3dgs, semantic_feature_dim=semantic_feature_dim,
        num_semantic_classes=num_semantic_classes)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return (params, variables, intrinsics, w2c, cam, params_opt_exclude, gs_cam,
                densify_intrinsics, densify_cam, densify_gs_cam)
    else:
        return params, variables, intrinsics, w2c, cam, params_opt_exclude, gs_cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss, sil_thres,
             use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
             plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None, load_semantics=False,
             use_one_hot_semantics=False, use_rgb_Triplane=False, use_sem_Triplane=False,
             feature_3dgs=False, decoder=None, all_planes=None, use_rgb_cond = False):
   
   


   

    # Initialize Loss Dictionary
    losses = {}
    
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=False,
                                             camera_grad=True, device=device)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=True, device=device)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, gaussians_grad=True,
                                             camera_grad=False, device=device)



    visibility_mask = compute_decoder_visibility_mask({'means3D': transformed_pts}, curr_data, variables)
    active_params, active_transformed_pts, _ = apply_visibility_mask(params, transformed_pts, visibility_mask)

    out = None
    decode_semantics = (use_sem_Triplane or feature_3dgs) and (not tracking)
    if use_rgb_Triplane or decode_semantics:
        pts = active_params['means3D']
        sem_feat = active_params['semantic_features'] if (feature_3dgs and 'semantic_features' in active_params) else None
        try:
            if use_rgb_cond and not use_rgb_Triplane:
                out = decoder(
                    pts, all_planes, decode_rgb=use_rgb_Triplane, decode_sem=decode_semantics,
                    rgb=active_params['rgb_colors'], sem_feat=sem_feat
                )
            else:
                out = decoder(
                    pts, all_planes, decode_rgb=use_rgb_Triplane, decode_sem=decode_semantics,
                    sem_feat=sem_feat
                )
        except TypeError:
            out = decoder(pts, all_planes)

    if tracking:
        mean, quat, scale, opac, color = params_to_gsplat_inputs(
            active_params, active_transformed_pts, use_semantics=load_semantics and (not tracking),
            use_one_hot_semantics=use_one_hot_semantics, out=out,
            use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=decode_semantics)
        color = color[:,:3]
    else:
        mean, quat, scale, opac, color = params_to_gsplat_inputs(
            active_params, active_transformed_pts, use_semantics=load_semantics,
            use_one_hot_semantics=use_one_hot_semantics, out=out,
            use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=decode_semantics)
    H,W,K,view=curr_data['gscam']
    RGBDS, alpha, meta = rasterization(
            mean, quat, scale, opac, color, view, K, W, H,
            render_mode="RGB+D")
    im = RGBDS[..., :3]  # 取RGB部分，[1,1200,680,3]
    # print("im shape before squeeze:", im.shape)
    im = im.squeeze(0)               # 去掉第一维的1，变成 [1200, 680, 3]
    im = im.permute(2, 0, 1)
    
    if not tracking:
        if load_semantics and not use_one_hot_semantics:
            rendered_seg = RGBDS[..., 3:6]
            rendered_seg = rendered_seg.squeeze(0).permute(2,0,1)
        if load_semantics and  use_one_hot_semantics:
            rendered_seg = RGBDS[..., 3:-1]
            rendered_seg = rendered_seg.squeeze(0).permute(2,0,1)
            
            # print("rendered_seg shape:",rendered_seg.shape)  # [52, 680, 1200]
    depth = RGBDS[..., -1]  # 取深度部分，[1,1200,680]
    # print("depth shape before squeeze:", depth.shape)
    depth= depth.squeeze(0)  
    # 去掉第一维的1，变成 [1200, 680]
    # print("depth shape before squeeze:", depth.shape)
    silhouette = alpha[0] if alpha.ndim==4 else alpha
    silhouette = silhouette.squeeze(0).permute(2,0,1)           # 去掉第一维的1，变成 [1200, 680]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth**2
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
        if load_semantics and not use_one_hot_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg)[color_mask].sum()
        # if load_semantics and use_one_hot_semantics:
        #     losses['seg'] =  F.cross_entropy(rendered_seg.unsqueeze(0), curr_data['semantic_id'].long(), reduction='none')[mask].mean()
            
            
            
            
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
        if load_semantics:
            losses['seg'] = torch.abs(curr_data['semantic_color'] - rendered_seg).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
        if load_semantics:
            if not use_one_hot_semantics:
                losses['seg'] = 0.8 * l1_loss_v1(rendered_seg, curr_data['semantic_color']) \
                    + 0.2 * (1.0 - calc_ssim(rendered_seg, curr_data['semantic_color']))
            else:
                losses['seg'] = F.cross_entropy(rendered_seg.unsqueeze(0), curr_data['semantic_id'].long(), reduction='mean')



    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    # seen = radius > 0
    # variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def get_loss_batch(params, curr_data_batch, variables, iter_time_idxs, loss_weights, use_sil_for_loss, sil_thres,
                   use_l1, ignore_outlier_depth_loss, tracking=False, mapping=False, do_ba=False, device="cuda",
                   load_semantics=False, use_one_hot_semantics=False,
                   use_rgb_Triplane=False, use_sem_Triplane=False,
                   feature_3dgs=False, decoder=None, all_planes=None, use_rgb_cond=False):
    if len(curr_data_batch) == 0:
        raise ValueError("curr_data_batch must contain at least one element.")
    if len(curr_data_batch) != len(iter_time_idxs):
        raise ValueError("iter_time_idxs and curr_data_batch should be the same length.")

    transformed_pts_batch = []
    for time_idx in iter_time_idxs:
        if tracking:
            transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                                 camera_grad=True, device=device)
        elif mapping:
            if do_ba:
                transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=True,
                                                     camera_grad=True, device=device)
            else:
                transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=True,
                                                     camera_grad=False, device=device)
        else:
            transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=True,
                                                 camera_grad=False, device=device)
        transformed_pts_batch.append(transformed_pts)

    visibility_mask = None
    for pts, sample_data in zip(transformed_pts_batch, curr_data_batch):
        mask = compute_decoder_visibility_mask({'means3D': pts}, sample_data, variables)
        if mask is None:
            visibility_mask = None
            break
        mask = mask.to(device=pts.device)
        if visibility_mask is None:
            visibility_mask = mask
        else:
            visibility_mask = visibility_mask | mask

    masked_params = params
    visible_indices = None
    if visibility_mask is not None:
        masked_params, _, visible_indices = apply_visibility_mask(params, transformed_pts_batch[0], visibility_mask)
        if visible_indices is not None:
            transformed_pts_batch = [pts[visible_indices] for pts in transformed_pts_batch]

    mean = torch.stack(transformed_pts_batch, dim=0).contiguous()

    tri_out = None
    decode_semantics = (use_sem_Triplane or feature_3dgs) and (not tracking)
    if use_rgb_Triplane or decode_semantics:
        pts = masked_params['means3D']
        sem_feat = masked_params['semantic_features'] if (feature_3dgs and 'semantic_features' in masked_params) else None
        try:
            if use_rgb_cond and (not use_rgb_Triplane):
                tri_out = decoder(
                    pts, all_planes, decode_rgb=use_rgb_Triplane, decode_sem=decode_semantics,
                    rgb=masked_params['rgb_colors'], sem_feat=sem_feat
                )
            else:
                tri_out = decoder(
                    pts, all_planes, decode_rgb=use_rgb_Triplane, decode_sem=decode_semantics,
                    sem_feat=sem_feat
                )
        except TypeError:
            tri_out = decoder(pts, all_planes)

    log_scales = masked_params['log_scales']
    if log_scales.shape[1] == 1:
        log_scales = log_scales.expand(-1, 3)
    scale = torch.exp(log_scales).contiguous()
    quat = F.normalize(masked_params['unnorm_rotations'], dim=-1)
    opac = torch.sigmoid(masked_params['logit_opacities']).reshape(-1)

    # Build per-point feature channels according to flags
    if use_rgb_Triplane:
        if tri_out is None or 'rgb_colors' not in tri_out:
            raise ValueError("Triplane RGB requested but decoder output missing 'rgb_colors'.")
        color = tri_out['rgb_colors'].contiguous()
    else:
        color = masked_params['rgb_colors'].contiguous()

    if load_semantics and (not tracking):
        if use_one_hot_semantics:
            if decode_semantics:
                if tri_out is None or 'semantic_id' not in tri_out:
                    raise ValueError("Triplane semantic requested but decoder output missing 'semantic_id'.")
                sem_ids = tri_out['semantic_id'].contiguous()
            else:
                sem_ids = masked_params['semantic_id'].contiguous()
            color = torch.cat([color, sem_ids], dim=-1)
        else:
            color = torch.cat([color, masked_params['semantic_colors'].contiguous()], dim=-1)

    batch_size = len(curr_data_batch)
    quat = quat.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    scale = scale.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    opac = opac.unsqueeze(0).expand(batch_size, -1).contiguous()
    color = color.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    H, W, _, _ = curr_data_batch[0]['gscam']
    for data in curr_data_batch[1:]:
        h_i, w_i, _, _ = data['gscam']
        if h_i != H or w_i != W:
            raise ValueError("Batch rendering requires frames with identical resolution.")

    Ks = torch.stack([data['gscam'][2] for data in curr_data_batch], dim=0).contiguous()
    views = torch.stack([data['gscam'][3] for data in curr_data_batch], dim=0).contiguous()

    RGBDS, alpha, meta = rasterization(
        mean, quat, scale, opac, color, views, Ks, W, H, render_mode="RGB+D")

    B = RGBDS.shape[0]
    # Predictions: squeeze view dimension and permute to BCHW
    # From [B, 1, H, W, 3] -> [B, 3, H, W]
    rgb_pred = RGBDS[:, 0, ..., :3].permute(0, 3, 1, 2)
    depth_pred = RGBDS[:, 0, ..., -1]
    # Alpha may be [B, 1, H, W, 1] or [B, H, W, 1]
    if alpha.ndim == 5:
        alpha_pred = alpha[:, 0]
    else:
        alpha_pred = alpha
    if alpha_pred.ndim == 4 and alpha_pred.shape[-1] == 1:
        alpha_pred = alpha_pred.squeeze(-1)

    # GT stacks
    im_gt = torch.stack([d['im'] for d in curr_data_batch], dim=0).contiguous()
    depth_gt = torch.stack([d['depth'].squeeze(0) for d in curr_data_batch], dim=0).contiguous()

    # Optional semantic GT stacks
    if load_semantics and not use_one_hot_semantics:
        seg_gt = torch.stack([d['semantic_color'] for d in curr_data_batch], dim=0).contiguous()
    if load_semantics and use_one_hot_semantics:
        sem_id_gt = torch.stack([d['semantic_id'].squeeze(0) for d in curr_data_batch], dim=0).contiguous()

    # Optional semantic predictions
    if load_semantics and not use_one_hot_semantics:
        # From [B, 1, H, W, 3] -> [B, 3, H, W]
        seg_pred = RGBDS[:, 0, ..., 3:6].permute(0, 3, 1, 2)
    if load_semantics and use_one_hot_semantics:
        # From [B, 1, H, W, K] -> [B, K, H, W], where K = num_semantic_classes
        seg_logits = RGBDS[:, 0, ..., 3:-1].permute(0, 3, 1, 2)

    # Masks
    presence_sil_mask = (alpha_pred > sil_thres)
    nan_mask = (~torch.isnan(depth_pred))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(depth_gt - depth_pred) * (depth_gt > 0)
        med = depth_error.view(B, -1).median(dim=1).values
        th = (10.0 * med)[:, None, None]
        mask = (depth_error < th) & (depth_gt > 0)
    else:
        mask = (depth_gt > 0)
    mask = mask & nan_mask
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    losses = {}

    # Depth loss
    if use_l1:
        if tracking:
            losses['depth'] = (torch.abs(depth_gt - depth_pred) * mask.float()).sum()
        else:
            diff = torch.abs(depth_gt - depth_pred) * mask.float()
            sum_per = diff.view(B, -1).sum(1)
            cnt_per = mask.view(B, -1).sum(1).clamp_min(1)
            losses['depth'] = (sum_per / cnt_per).mean()

    # RGB/seg losses
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = mask.unsqueeze(1).expand(-1, 3, -1, -1).float()
        losses['im'] = (torch.abs(im_gt - rgb_pred) * color_mask).sum()
        if load_semantics and not use_one_hot_semantics:
            losses['seg'] = (torch.abs(seg_gt - seg_pred) * color_mask).sum()
    elif tracking:
        losses['im'] = torch.abs(im_gt - rgb_pred).sum()
        if load_semantics and not use_one_hot_semantics:
            losses['seg'] = torch.abs(seg_gt - seg_pred).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(rgb_pred, im_gt) + 0.2 * (1.0 - calc_ssim(rgb_pred, im_gt))
        if load_semantics:
            if not use_one_hot_semantics:
                losses['seg'] = 0.8 * l1_loss_v1(seg_pred, seg_gt) + 0.2 * (1.0 - calc_ssim(seg_pred, seg_gt))
            else:
                losses['seg'] = F.cross_entropy(seg_logits, sem_id_gt.long(), reduction='mean')

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, device, load_semantics=False,
                          params_opt_exclude=None, use_rgb_Triplane=False,
                          use_one_hot_semantics=False, use_sem_Triplane=False,
                          feature_3dgs=False, semantic_feature_dim=32,
                          num_semantic_classes=None):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device=device)
    if use_rgb_Triplane:
        params = {
        'means3D': means3D,
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    else:
        params = {
            'means3D': means3D,
            'rgb_colors': new_pt_cld[:, 3:6],
            'unnorm_rotations': unnorm_rots,
            'logit_opacities': logit_opacities,
            'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        }

    if load_semantics:
        if use_one_hot_semantics:
            if feature_3dgs:
                if num_semantic_classes is not None and num_semantic_classes > 0:
                    sem_dim = num_semantic_classes
                else:
                    sem_dim = new_pt_cld.shape[1] - 9
                params['semantic_features'] = torch.empty(
                    (num_pts, semantic_feature_dim), dtype=torch.float, device=device
                ).normal_(mean=0.0, std=0.01)
            elif not use_sem_Triplane:
                if num_semantic_classes is not None and num_semantic_classes > 0:
                    sem_dim = num_semantic_classes
                else:
                    sem_dim = new_pt_cld.shape[1] - 9
                params['semantic_id'] = new_pt_cld[:, 6:6 + sem_dim]
            # Always keep semantic colors if present in point cloud
            if 'semantic_id' in params and new_pt_cld.shape[1] >= 6 + params['semantic_id'].shape[1] + 3:
                start = 6 + params['semantic_id'].shape[1]
                params['semantic_colors'] = new_pt_cld[:, start:start + 3]
            elif feature_3dgs:
                start = 6 + sem_dim
                params['semantic_colors'] = new_pt_cld[:, start:start + 3]
            else:
                params['semantic_colors'] = new_pt_cld[:, 7:10]
        else:
            params['semantic_id'] = new_pt_cld[:, 6]
            params['semantic_colors'] = new_pt_cld[:, 7:10]

    for k, v in params.items():
        if k not in params_opt_exclude:
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(device).float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, params_opt_exclude, variables, curr_data, sil_thres, time_idx,
                      mean_sq_dist_method, device="cuda", load_semantics=False,
                      use_Triplane=False,use_one_hot_semantics=False,
                      use_sem_Triplane=False,use_rgb_Triplane=False,
                      use_init_plane = False,
                      feature_3dgs=False, semantic_feature_dim=32,
                      num_semantic_classes=None):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False,
                                         camera_grad=False, device=device)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts, device=device)
    
    
    
    # depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    
    
    
    mean, quat, scale, opac, renders = params_to_gsplat_inputs(params, transformed_pts,use_semantics=False)
    H,W,K,view=curr_data['gscam']
    RGBDS, alpha, meta = rasterization(
            mean, quat, scale, opac, renders, view, K, W, H,
            render_mode="RGB+D")


    render_depth = RGBDS[..., -1]  # 取深度部分，[1,1200,680]
    # rastered_depth= depth.squeeze(0)
    silhouette = alpha[0] if alpha.ndim==4 else alpha
    silhouette = silhouette.squeeze(0).permute(2,0,1).squeeze(0)           # 去掉第一维的1，变成 [1200, 680]
   

    
    
    
    
    
    
    
    
    
    # silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres)
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    
    # render_depth = depth_sil[0, :, :]
    render_depth = RGBDS[..., -1] 
    
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).to(device).float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

        if load_semantics:
            semantic_id = curr_data['semantic_id']
            semantic_color = curr_data['semantic_color']
        else:
            semantic_id = None
            semantic_color = None

        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'],
                                                   curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                                   mean_sq_dist_method=mean_sq_dist_method, device=device,
                                                   load_semantics=load_semantics, semantic_id=semantic_id,
                                                   semantic_color=semantic_color,
                                                   num_semantic_classes=num_semantic_classes)
        new_params = initialize_new_params(
            new_pt_cld, mean3_sq_dist, device, load_semantics=load_semantics,
            params_opt_exclude=params_opt_exclude,
            use_rgb_Triplane=use_rgb_Triplane,
            use_one_hot_semantics=use_one_hot_semantics,
            use_sem_Triplane=use_sem_Triplane,
            feature_3dgs=feature_3dgs, semantic_feature_dim=semantic_feature_dim,
            num_semantic_classes=num_semantic_classes)
        
        
        if use_init_plane:
            init_plane_semantic = []
            if num_semantic_classes is not None and num_semantic_classes > 0:
                sem_dim = num_semantic_classes
            else:
                sem_dim = new_pt_cld.shape[1] - 9
            init_plane_semantic = new_pt_cld[:, :6 + sem_dim].clone()

        for k, v in new_params.items():
            if k not in params_opt_exclude:
                params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            else:
                params[k] = torch.cat((params[k], v), dim=0)
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device=device).float()
        variables['denom'] = torch.zeros(num_pts, device=device).float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device=device).float()
        new_timestep = time_idx * torch.ones(new_pt_cld.shape[0], device=device).float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep), dim=0)

    return params, variables,init_plane_semantic if use_init_plane else None


def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Initialize the camera pose for the current frame based on a constant velocity model
            # Rotation
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
            # Translation
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def build_w2c_from_params(params, frame_idx):
    """Build a 4x4 world-to-camera matrix from stored quaternion and translation."""
    rot = F.normalize(params['cam_unnorm_rots'][..., frame_idx].detach())
    tran = params['cam_trans'][..., frame_idx].detach()
    w2c = torch.eye(4, device=rot.device, dtype=rot.dtype)
    w2c[:3, :3] = build_rotation(rot)
    w2c[:3, 3] = tran
    return w2c


def estimate_pose_with_orb_pnp(curr_frame, prev_frame, intrinsics, prev_w2c, device, max_matches=500):
    """
    Estimate the current pose using ORB feature matching + PnP against the previous frame.
    Inputs are expected as torch tensors: curr_frame (C,H,W) in [0,1], prev_frame dict with keys im/depth.
    Returns a tuple (quat, tran) if successful, otherwise None.
    """
    if prev_frame is None:
        return None
    prev_color = prev_frame.get('im', None)
    prev_depth = prev_frame.get('depth', None)
    if prev_color is None or prev_depth is None:
        return None

    # Convert tensors to CPU numpy for OpenCV
    curr_img = (curr_frame.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    prev_img = (prev_color.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    prev_depth_np = prev_depth.squeeze(0).detach().cpu().numpy()

    # Guard invalid depth shapes
    if prev_depth_np.ndim != 2:
        return None
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY) if prev_img.ndim == 3 else prev_img
    curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY) if curr_img.ndim == 3 else curr_img

    # Prepare intrinsics
    if torch.is_tensor(intrinsics):
        intr_np = intrinsics.detach().cpu().numpy()
    else:
        intr_np = np.array(intrinsics)
    fx, fy = float(intr_np[0, 0]), float(intr_np[1, 1])
    cx, cy = float(intr_np[0, 2]), float(intr_np[1, 2])

    # Detect & match ORB features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_img_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_img_gray, None)
    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 6:
        return None
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    obj_pts = []
    img_pts = []
    h, w = prev_depth_np.shape
    for m in matches:
        u_prev, v_prev = kp1[m.queryIdx].pt
        u_cur, v_cur = kp2[m.trainIdx].pt
        u_prev_i, v_prev_i = int(round(u_prev)), int(round(v_prev))
        if u_prev_i < 0 or u_prev_i >= w or v_prev_i < 0 or v_prev_i >= h:
            continue
        z = float(prev_depth_np[v_prev_i, u_prev_i])
        if not np.isfinite(z) or z <= 0:
            continue
        x = (u_prev - cx) / fx * z
        y = (v_prev - cy) / fy * z
        obj_pts.append([x, y, z])
        img_pts.append([u_cur, v_cur])

    if len(obj_pts) < 6:
        return None

    obj_pts = np.asarray(obj_pts, dtype=np.float32)
    img_pts = np.asarray(img_pts, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, intr_np, None, flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=200, reprojectionError=8.0
    )
    if not success or inliers is None or len(inliers) < 6:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T_rel = np.eye(4, dtype=np.float32)
    T_rel[:3, :3] = R.astype(np.float32)
    T_rel[:3, 3] = tvec.reshape(3).astype(np.float32)

    prev_w2c = prev_w2c.detach().to(device=device, dtype=torch.float32)
    curr_w2c = torch.from_numpy(T_rel).to(device=device) @ prev_w2c
    quat = matrix_to_quaternion(curr_w2c[:3, :3].unsqueeze(0))[0]
    quat = F.normalize(quat, dim=-1)
    tran = curr_w2c[:3, 3]
    return quat, tran


def evaluate_pose_color_depth_loss(params, rot, tran, curr_data, variables, iter_time_idx, tracking_config,
                                   load_semantics=False, use_one_hot_semantics=False,
                                   use_rgb_Triplane=False, use_sem_Triplane=False,
                                   feature_3dgs=False, decoder=None, all_planes=None, use_rgb_cond=False, device="cuda"):
    """Render once with the provided pose and return weighted color+depth loss (no grad)."""
    with torch.no_grad():
        orig_rot = params['cam_unnorm_rots'][..., iter_time_idx].detach().clone()
        orig_tran = params['cam_trans'][..., iter_time_idx].detach().clone()
        params['cam_unnorm_rots'][..., iter_time_idx] = rot
        params['cam_trans'][..., iter_time_idx] = tran
        _, _, weighted_losses = get_loss(
            params, curr_data, variables, iter_time_idx, tracking_config['loss_weights'],
            tracking_config['use_sil_for_loss'], tracking_config['sil_thres'], tracking_config['use_l1'],
            tracking_config['ignore_outlier_depth_loss'], tracking=True, device=device,
            load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics,
            use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
            feature_3dgs=feature_3dgs,
            decoder=decoder if (use_rgb_Triplane or use_sem_Triplane or feature_3dgs) else None,
            all_planes=all_planes if (use_rgb_Triplane or use_sem_Triplane) else None,
            use_rgb_cond=use_rgb_cond)
        params['cam_unnorm_rots'][..., iter_time_idx] = orig_rot
        params['cam_trans'][..., iter_time_idx] = orig_tran
    color_loss = float(weighted_losses.get('im', 0.0))
    depth_loss = float(weighted_losses.get('depth', 0.0))
    return color_loss + depth_loss


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store



def compute_decoder_visibility_mask(transformed_gaussians, curr_data, variables, margin_ratio=0.05, min_keep_ratio=0.05):
    if transformed_gaussians is None:
        return None
    if isinstance(transformed_gaussians, dict):
        pts_cam = transformed_gaussians.get('means3D', None)
    else:
        pts_cam = transformed_gaussians
    if pts_cam is None or pts_cam.shape[0] == 0:
        return None

    device = pts_cam.device
    mask = torch.isfinite(pts_cam).all(dim=1)
    if not mask.any():
        return mask

    z = pts_cam[:, 2]
    mask = mask & (z > 0)

    intrinsics = curr_data.get('intrinsics') if curr_data is not None else None
    if intrinsics is None:
        return mask

    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, device=device, dtype=pts_cam.dtype)
    else:
        intrinsics = intrinsics.to(device=device, dtype=pts_cam.dtype)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Avoid division instabilities
    z_safe = torch.where(torch.abs(z) < 1e-6, torch.full_like(z, 1e-6), z)
    x_proj = fx * (pts_cam[:, 0] / z_safe) + cx
    y_proj = fy * (pts_cam[:, 1] / z_safe) + cy

    im = curr_data.get('im') if curr_data is not None else None
    if im is None:
        return mask
    if im.dim() == 3:
        _, img_h, img_w = im.shape
    else:
        img_h, img_w = im.shape[-2:]

    margin_x = margin_ratio * img_w
    margin_y = margin_ratio * img_h

    radius_buffer = None
    if variables is not None:
        radius_buffer = variables.get('max_2D_radius', None)
        if radius_buffer is not None:
            if not torch.is_tensor(radius_buffer):
                radius_buffer = torch.tensor(radius_buffer, device=device, dtype=pts_cam.dtype)
            else:
                radius_buffer = radius_buffer.to(device=device, dtype=pts_cam.dtype)
            if radius_buffer.shape[0] != pts_cam.shape[0]:
                radius_buffer = None

    if radius_buffer is None:
        expand_x = margin_x
        expand_y = margin_y
    else:
        expand_x = radius_buffer + margin_x
        expand_y = radius_buffer + margin_y

    mask = mask & (x_proj >= -expand_x) & (x_proj <= img_w - 1 + expand_x)
    mask = mask & (y_proj >= -expand_y) & (y_proj <= img_h - 1 + expand_y)

    # Ensure we keep at least a minimum ratio to remain conservative
    if mask.sum() < min_keep_ratio * mask.shape[0]:
        return torch.ones_like(mask, dtype=torch.bool, device=device)

    return mask


def apply_visibility_mask(params, transformed_pts, visibility_mask):
    if visibility_mask is None:
        return params, transformed_pts, None
    if not torch.is_tensor(visibility_mask):
        return params, transformed_pts, None
    if visibility_mask.dtype != torch.bool:
        visibility_mask = visibility_mask.bool()
    if visibility_mask.ndim != 1:
        visibility_mask = visibility_mask.view(-1)
    if visibility_mask.shape[0] != transformed_pts.shape[0]:
        return params, transformed_pts, None
    if visibility_mask.sum() == 0 or visibility_mask.all():
        return params, transformed_pts, None

    visible_indices = torch.nonzero(visibility_mask, as_tuple=False).squeeze(1)
    base_count = params['means3D'].shape[0]
    masked_params = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == base_count:
            masked_params[k] = v[visible_indices]
        else:
            masked_params[k] = v
    masked_transformed_pts = transformed_pts[visible_indices]
    return masked_params, masked_transformed_pts, visible_indices






def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    use_one_hot_semantics = config.get('use_one_hot_semantics', False)
    use_orb_pnp = config.get('use_orb_pnp', False)
    orb_pnp_loss_scale = float(config.get('orb_pnp_loss_scale', 1.05))
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])
    if config["primary_device"].startswith("cuda:"):
        device_id = int(config["primary_device"].split(':')[1])
        torch.cuda.set_device(device_id)


      #初始化三平面 / 语义特征解码器
    use_rgb_Triplane = config.get("use_rgb_Triplane", False)
    use_sem_Triplane = config.get("use_sem_Triplane", False)
    feature_3dgs = bool(config.get("feature_3dgs", False)) and (not use_sem_Triplane)
    semantic_feature_dim = 32
    use_Triplane = use_rgb_Triplane or use_sem_Triplane
    use_semantic_decoder = use_sem_Triplane or feature_3dgs
    use_decoder = use_rgb_Triplane or use_semantic_decoder
    decoder = None
    all_planes = None
    planes_lr_config = {}
    plane_lr_factor = 1.0
    plane_lr_first_factor = 1.0
    if use_decoder or use_Triplane:
        use_proj_decoder = config.get("use_proj_decoder",False)
        if feature_3dgs and use_proj_decoder:
            raise ValueError("feature_3dgs=True requires the MLP decoder; set use_proj_decoder=False.")
        Planes_cfg = load_dataset_config(config["TriplaneConfigs"])
        if use_Triplane:
            print("Initialized Tri-Planes for Splatting Densification")
        c_dim = Planes_cfg['model']['c_dim']
        
        #加载边界
        bound = Tri_Plane_help.load_bound(Planes_cfg)
       
        #初始化平面
        if use_Triplane:
            raw_planes = Tri_Plane_help.init_planes(Planes_cfg, bound)
            all_planes = tuple([[plane.to(device) for plane in plane_list] for plane_list in raw_planes])
        if use_proj_decoder:
            decoder = None
        else:
            sem_dim_cfg = config.get("data", {}).get("num_semantic_classes", 0)
            decoder = Decoders(
                bound,
                c_dim=c_dim,
                sem_dim=sem_dim_cfg if sem_dim_cfg > 0 else 52,
                sem_feat_dim=semantic_feature_dim,
            ).to(device)
        planes_lr_config = Planes_cfg.get('lr', {})
        plane_lr_factor = Planes_cfg.get('lr_factor', 1.0)
        plane_lr_first_factor = Planes_cfg.get('lr_first_factor', plane_lr_factor)




    # Load Dataset
    if True:
        print("Loading Dataset ...")
        dataset_config = config["data"]
        if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        else:
            gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        if "ignore_bad" not in dataset_config:
            dataset_config["ignore_bad"] = False
        if "use_train_split" not in dataset_config:
            dataset_config["use_train_split"] = True
        if "densification_image_height" not in dataset_config:
            dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
            dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
            seperate_densification_res = False
        else:
            if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
                seperate_densification_res = True
            else:
                seperate_densification_res = False
        if "tracking_image_height" not in dataset_config:
            dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
            dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
            seperate_tracking_res = False
        else:
            if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
                seperate_tracking_res = True
            else:
                seperate_tracking_res = False
        if "load_semantics" not in dataset_config:
            load_semantics = False
            num_semantic_classes = 0
        else:
            load_semantics = dataset_config["load_semantics"]
            num_semantic_classes = dataset_config["num_semantic_classes"]
        # Poses are relative to the first frame
        dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            load_semantics=load_semantics,
            num_semantic_classes=num_semantic_classes,
        )
        num_frames = dataset_config["num_frames"]
        if num_frames == -1:
            num_frames = len(dataset)
        print("num_frames", num_frames)
    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, params_opt_exclude, \
            gs_cam, densify_intrinsics, densify_cam, densify_gs_cam = initialize_first_timestep(
                dataset, num_frames, config['scene_radius_depth_ratio'],
                config['mean_sq_dist_method'],
                device=device,
                densify_dataset=densify_dataset,
                load_semantics=load_semantics,
                use_one_hot_semantics=use_one_hot_semantics,
                use_rgb_Triplane=use_rgb_Triplane,
                use_sem_Triplane=use_sem_Triplane,
                feature_3dgs=feature_3dgs,
                semantic_feature_dim=semantic_feature_dim,
                num_semantic_classes=num_semantic_classes)
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            params_opt_exclude, gs_cam = initialize_first_timestep(
                dataset, num_frames, config['scene_radius_depth_ratio'],
                config['mean_sq_dist_method'], device=device,
                load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics,
                use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
                feature_3dgs=feature_3dgs, semantic_feature_dim=semantic_feature_dim,
                num_semantic_classes=num_semantic_classes)
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1],
                                    tracking_intrinsics.cpu().numpy(),
                                    first_frame_w2c.detach().cpu().numpy(), device=device)
        tracking_gs_cam = to_gsplat_camera(tracking_color.shape[2], tracking_color.shape[1],
                                           tracking_intrinsics.cpu().numpy(),
                                           first_frame_w2c.detach().cpu().numpy(), device=device)
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    timestamp_keyframes = []
    prev_frame_for_pnp = None
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        for k in params:
            if k not in params_opt_exclude:
                params[k] = torch.tensor(params[k]).to(device).float().requires_grad_(True)
            else:
                params[k] = torch.tensor(params[k]).to(device).float()

        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).to(device).float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).to(device).float()

        # Load decoder / tri-planes for this checkpoint when enabled
        if use_decoder or use_Triplane:
            ckpt_output_dir = os.path.join(config['workdir'], config['run_name'])
            # Prefer checkpoint-specific weights, fall back to final ones if necessary
            dec_ckpt_path = os.path.join(ckpt_output_dir, f"{checkpoint_time_idx}_decoder.pth")
            planes_ckpt_path = os.path.join(ckpt_output_dir, f"{checkpoint_time_idx}_planes.pth")
            dec_final_path = os.path.join(ckpt_output_dir, "decoder.pth")
            planes_final_path = os.path.join(ckpt_output_dir, "planes.pth")

            # Load decoder weights if available and decoder exists
            dec_to_load = dec_ckpt_path if os.path.exists(dec_ckpt_path) else dec_final_path
            if decoder is not None and os.path.exists(dec_to_load):
                try:
                    state_dict = torch.load(dec_to_load, map_location=device)
                    decoder.load_state_dict(state_dict)
                    print(f"Loaded decoder weights from: {dec_to_load}")
                except Exception as e:
                    print(f"Warning: failed to load decoder weights from {dec_to_load}: {e}")

            # Load tri-plane tensors if available
            planes_to_load = planes_ckpt_path if os.path.exists(planes_ckpt_path) else planes_final_path
            if use_Triplane and os.path.exists(planes_to_load):
                try:
                    planes_saved = torch.load(planes_to_load, map_location=device)
                    loaded_all_planes = []
                    # keys are like 'group_0', 'group_1', ...
                    for gi_key in sorted(planes_saved.keys(), key=lambda x: int(x.split('_')[1])):
                        plane_list = planes_saved[gi_key]
                        loaded_all_planes.append(
                            [p.to(device) if isinstance(p, torch.Tensor) else p for p in plane_list]
                        )
                    all_planes = tuple(loaded_all_planes)
                    print(f"Loaded tri-plane tensors from: {planes_to_load}")
                except Exception as e:
                    print(f"Warning: failed to load tri-planes from {planes_to_load}: {e}")

        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            if load_semantics:
                color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
            else:
                color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}

                if load_semantics:
                    semantic_id = semantic_id.permute(2, 0, 1)
                    semantic_color = semantic_color.permute(2, 0, 1) / 255
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
        # Load RGBD frames incrementally instead of all frames
        if load_semantics:
            color, depth, _, gt_pose, semantic_id, semantic_color = dataset[time_idx]
        else:
            color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx
        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,'gscam':gs_cam}
        
        if load_semantics:
            semantic_id = semantic_id.permute(2, 0, 1)
            semantic_color = semantic_color.permute(2, 0, 1) / 255
            curr_data['semantic_id'] = semantic_id
            curr_data['semantic_color'] = semantic_color
        
        # Initialize Data for Tracking
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth,
                                  'id': iter_time_idx, 'intrinsics': tracking_intrinsics,
                                  'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c,
                                  'gscam': tracking_gs_cam}
        else:
            tracking_curr_data = curr_data
        pose_eval_data = tracking_curr_data if 'gscam' in tracking_curr_data else curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        use_batch_mapping = bool(config.get('use_batch', False))
        batch_keyframe_count = max(1, int(config.get('batch', 1)))
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
            if use_orb_pnp and prev_frame_for_pnp is not None:
                const_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                const_tran = params['cam_trans'][..., time_idx].detach().clone()
                prev_w2c = build_w2c_from_params(params, time_idx - 1)
                orb_pose = estimate_pose_with_orb_pnp(
                    pose_eval_data['im'], prev_frame_for_pnp, pose_eval_data['intrinsics'], prev_w2c, device=device
                )
                if orb_pose is not None:
                    orb_rot, orb_tran = orb_pose
                    const_loss = evaluate_pose_color_depth_loss(
                        params, const_rot, const_tran, pose_eval_data, variables, iter_time_idx, config['tracking'],
                        load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics,
                        use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
                        feature_3dgs=feature_3dgs,
                        decoder=decoder if use_decoder else None, all_planes=all_planes if use_Triplane else None,
                        use_rgb_cond=config.get('use_rgb_cond', False), device=device)
                    orb_loss = evaluate_pose_color_depth_loss(
                        params, orb_rot, orb_tran, pose_eval_data, variables, iter_time_idx, config['tracking'],
                        load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics,
                        use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
                        feature_3dgs=feature_3dgs,
                        decoder=decoder if use_decoder else None, all_planes=all_planes if use_Triplane else None,
                        use_rgb_cond=config.get('use_rgb_cond', False), device=device)
                    scaled_orb_loss = orb_loss * orb_pnp_loss_scale
                    print(f"[ORB+PnP init] frame {time_idx}: const_loss={const_loss:.4f}, "
                          f"orb_loss_scaled={scaled_orb_loss:.4f} (raw={orb_loss:.4f}, scale={orb_pnp_loss_scale})")
                    if orb_loss * orb_pnp_loss_scale < const_loss:
                        with torch.no_grad():
                            params['cam_unnorm_rots'][..., time_idx].copy_(orb_rot.detach())
                            params['cam_trans'][..., time_idx].copy_(orb_tran.detach())
                        print(f"[ORB+PnP init] frame {time_idx}: chose ORB+PnP pose")
                    else:
                        print(f"[ORB+PnP init] frame {time_idx}: kept constant-velocity pose")


        # Step 1: Tracking
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(
                params,
                config['tracking']['lrs'],
                tracking=True,
                use_planes=use_Triplane,
                decoder=decoder if use_decoder else None,
                all_planes=all_planes if use_Triplane else None,
                plane_lr_config=planes_lr_config if use_decoder else None,
                plane_lr_scale=plane_lr_factor if use_decoder else 1.0,
            )
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            continue_slam_iter = 0
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'],
                                                   tracking=True, device=device, plot_dir=eval_dir,
                                                   visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter, load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics,
                                                   use_rgb_Triplane=use_rgb_Triplane, use_sem_Triplane=use_sem_Triplane,
                                                   feature_3dgs=feature_3dgs,
                                                   decoder=decoder if use_decoder else None, all_planes=all_planes if use_Triplane else None,
                                                   use_rgb_cond=config.get('use_rgb_cond', False))
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_tracking_step,
                                            wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                            tracking=True, device=device, load_semantics=load_semantics)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                

                # print("loss:",losses['depth'])
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        continue_slam_iter = continue_slam_iter+1
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*config['tracking']['num_iters']
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            # try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    
                    if use_decoder:
                        params_report = params.copy()
                        params_report = decode_gaussian_attributes(
                            params_report, decoder, all_planes,
                            use_rgb_Triplane=use_rgb_Triplane,
                            decode_semantics=load_semantics and use_one_hot_semantics and use_semantic_decoder,
                        )
                    else:
                        params_report = params
                    if config['use_wandb']:
                        report_progress(params_report, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics, wandb_run=wandb_run, wandb_step=wandb_time_step,
                                        wandb_save_qual=config['wandb']['save_qual'], global_logging=True,use_one_hot_semantics=use_one_hot_semantics)
                    else:
                        report_progress(params_report, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'],
                                        tracking=True, device=device, load_semantics=load_semantics, use_one_hot_semantics=use_one_hot_semantics)
                progress_bar.close()
            # except:
            #     ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            #     # 保存当前高斯参数
            #     save_params_ckpt(params, ckpt_output_dir, time_idx)
            #     # 如果使用三平面，同步保存 decoder 和 planes
            #     if use_Triplane:
            #         try:
            #             save_decoder_and_planes(
            #                 decoder if use_Triplane else None,
            #                 all_planes if use_Triplane else None,
            #                 ckpt_output_dir,
            #                 prefix=f"{time_idx}_",
            #             )
            #         except Exception as e:
            #             print(f"Warning: failed to save decoder/planes at tracking time {time_idx}: {e}")
            #     print('Failed to evaluate trajectory.')

        # Step 2: Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c,
                                 'iter_gt_w2c_list': curr_gt_w2c, 'gscam': densify_gs_cam}
                else:
                    densify_curr_data = curr_data

                use_init_plane = config.get('use_init_plane',False)


                # Add new Gaussians to the scene based on the Silhouette
                params, variables,init_plane_semantic = add_new_gaussians(
                    params, params_opt_exclude, variables, densify_curr_data,
                    config['mapping']['sil_thres'], time_idx, config['mean_sq_dist_method'],
                    device, load_semantics=load_semantics, use_Triplane=use_Triplane,
                    use_one_hot_semantics=use_one_hot_semantics, use_sem_Triplane=use_sem_Triplane,
                    use_rgb_Triplane=use_rgb_Triplane, use_init_plane=use_init_plane,
                    feature_3dgs=feature_3dgs, semantic_feature_dim=semantic_feature_dim,
                    num_semantic_classes=num_semantic_classes)
                      
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            # Update keyframes for gaussian mapping
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping
                if config["keyframe_random"]:
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                    num_keyframes, device=device)
                    # 将一半的关键帧替换为从所有已有关键帧中随机选取
                    
                    
                    if len(keyframe_list) > 0 and len(selected_keyframes) > 1:
                        try:
                            half_n = max(1, len(selected_keyframes) // 2)
                            total_kf = len(keyframe_list)
                            # 在所有已存在关键帧索引范围 [0, total_kf-1] 内随机采样（不放回）
                            # 避免与当前 overlap 选取完全重复，使用集合去重
                            rand_pool = np.arange(total_kf)
                            rand_pick = np.random.choice(rand_pool, size=min(half_n, total_kf), replace=False).tolist()
                            # 保留一半 overlap 结果，其余用随机关键帧替换
                            keep_overlap = selected_keyframes[:len(selected_keyframes) - len(rand_pick)]
                            mixed = list(dict.fromkeys(keep_overlap + rand_pick))  # 去重并保持顺序
                            # 若去重后数量不足，追加更多随机索引填满
                            if len(mixed) < len(selected_keyframes):
                                need = len(selected_keyframes) - len(mixed)
                                more_pool = [i for i in rand_pool.tolist() if i not in mixed]
                                if len(more_pool) > 0:
                                    more = np.random.choice(more_pool, size=min(need, len(more_pool)), replace=False).tolist()
                                    mixed.extend(more)
                            # 截断到原始数量
                            selected_keyframes = mixed[:len(selected_keyframes)]
                        except Exception:
                            pass
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    # print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    timestamp_keyframes.append(selected_time_idx)
                else:
                    num_keyframes = config['mapping_window_size']-2
                    selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1],
                                                                    num_keyframes, device=device)
                    selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                    if len(keyframe_list) > 0:
                        # Add last keyframe to the selected keyframes
                        selected_time_idx.append(keyframe_list[-1]['id'])
                        selected_keyframes.append(len(keyframe_list)-1)
                    # Add current frame to the selected keyframes
                    selected_time_idx.append(time_idx)
                    selected_keyframes.append(-1)
                    # Print the selected keyframes
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    timestamp_keyframes.append(selected_time_idx)



            # Reset Optimizer & Learning Rates for Full Map Optimization
            plane_lr_scale = plane_lr_factor
            if use_Triplane:
                plane_lr_scale = plane_lr_first_factor if (use_Triplane and time_idx == 0) else plane_lr_factor
            optimizer = initialize_optimizer(
                params,
                config['mapping']['lrs'],
                tracking=False,
                use_planes=use_Triplane,
                decoder=decoder if use_decoder else None,
                all_planes=all_planes if use_Triplane else None,
                plane_lr_config=planes_lr_config if use_decoder else None,
                plane_lr_scale=plane_lr_scale if use_decoder else 1.0,
            )

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")

            def prepare_mapping_iter_data(selected_idx):
                if selected_idx == -1:
                    frame_time_idx = time_idx
                    frame_color = color
                    frame_depth = depth
                    if load_semantics:
                        frame_sem_id = semantic_id
                        frame_sem_color = semantic_color
                else:
                    keyframe = keyframe_list[selected_idx]
                    frame_time_idx = keyframe['id']
                    frame_color = keyframe['color']
                    frame_depth = keyframe['depth']
                    if load_semantics:
                        frame_sem_id = keyframe['semantic_id']
                        frame_sem_color = keyframe['semantic_color']
                frame_gt_w2c = gt_w2c_all_frames[:frame_time_idx+1]
                frame_iter_data = {'cam': cam, 'im': frame_color, 'depth': frame_depth, 'id': frame_time_idx,
                                   'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': frame_gt_w2c,'gscam':gs_cam}
                if load_semantics:
                    frame_iter_data['semantic_id'] = frame_sem_id
                    frame_iter_data['semantic_color'] = frame_sem_color
                return frame_time_idx, frame_iter_data




            # warmup: 前 warmup_ratio * num_iters_mapping 轮不解码/不渲染语义
            warmup_ratio = float(config.get('warmup_ratio', 0.0))
            warmup_iters = int(num_iters_mapping * warmup_ratio) if warmup_ratio > 0 else 0
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # 语义开关：仅在过了 warmup_iters 之后才启用
                sem_enabled = (iter >= warmup_iters)

                if use_batch_mapping:
                    effective_batch = min(batch_keyframe_count, len(selected_keyframes))
                    
                    
                    
                    # sampled_indices = np.random.choice(len(selected_keyframes), size=effective_batch, replace=False).tolist()
                    
                    N = len(selected_keyframes)
                    if effective_batch > 2:
                        last_idx = N - 1
                        # 从 0..N-2 中随机抽取 effective_batch-1 个
                        sampled_rest = np.random.choice(N - 1, size=effective_batch - 1, replace=False)
                        sampled_indices = np.concatenate([sampled_rest, [last_idx]]).tolist()

                        # 如果你希望抽到的索引顺序也是随机的（而不是最后一个总在最后），再打乱一次：
                        # sampled_indices = np.random.permutation(sampled_indices).tolist()
                    else:
                        # 原逻辑：从全部索引中随机不放回抽取
                        sampled_indices = np.random.choice(N, size=effective_batch, replace=False).tolist()
                    
                    batch_iter_data = []
                    batch_time_indices = []
                    for sample_idx in sampled_indices:
                        selected_idx = selected_keyframes[int(sample_idx)]
                        frame_time_idx, frame_iter_data = prepare_mapping_iter_data(selected_idx)
                        batch_iter_data.append(frame_iter_data)
                        batch_time_indices.append(frame_time_idx)
                    iter_data = batch_iter_data[0]
                    iter_time_idx = batch_time_indices[0]
                    loss, variables, losses = get_loss_batch(
                        params, batch_iter_data, variables, batch_time_indices, config['mapping']['loss_weights'],
                        config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                        config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                        mapping=True, device=device, load_semantics=(load_semantics and sem_enabled),
                        use_one_hot_semantics=use_one_hot_semantics,
                        use_rgb_Triplane=use_rgb_Triplane,
                        use_sem_Triplane=(use_sem_Triplane and sem_enabled),
                        feature_3dgs=(feature_3dgs and sem_enabled),
                        decoder=decoder if use_decoder else None, all_planes=all_planes if use_Triplane else None,
                        use_rgb_cond=(config.get('use_rgb_cond', False) and sem_enabled))
                else:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    iter_time_idx, iter_data = prepare_mapping_iter_data(selected_rand_keyframe_idx)
                    loss, variables, losses = get_loss(
                        params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                        config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                        config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'],
                        mapping=True, device=device, load_semantics=(load_semantics and sem_enabled),
                        use_one_hot_semantics=use_one_hot_semantics,
                        use_rgb_Triplane=use_rgb_Triplane,
                        use_sem_Triplane=(use_sem_Triplane and sem_enabled),
                        feature_3dgs=(feature_3dgs and sem_enabled),
                        decoder=decoder if use_decoder else None, all_planes=all_planes if use_Triplane else None,
                        use_rgb_cond=(config.get('use_rgb_cond', False) and sem_enabled))
                



                
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True, load_semantics=load_semantics)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, params_opt_exclude, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'], params_opt_exclude, device=device)
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                # try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        
                        if use_decoder:
                            params_report = params.copy()
                            params_report = decode_gaussian_attributes(
                                params_report, decoder, all_planes,
                                use_rgb_Triplane=use_rgb_Triplane,
                                decode_semantics=load_semantics and use_one_hot_semantics and use_semantic_decoder,
                            )
                        else:
                            params_report = params
                        
                        if config['use_wandb']:
                            report_progress(params_report, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx, global_logging=True,use_one_hot_semantics=use_one_hot_semantics)
                        else:
                            report_progress(params_report, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, device=device, load_semantics=load_semantics, online_time_idx=time_idx,use_one_hot_semantics=use_one_hot_semantics)
                    progress_bar.close()
                # except:
                #     ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                #     # save_params_ckpt(params, ckpt_output_dir, time_idx)
                #     print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).to(device).float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                if load_semantics:
                    curr_keyframe['semantic_id'] = semantic_id
                    curr_keyframe['semantic_color'] = semantic_color
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            # 保存当前高斯参数
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
            # 如果使用 decoder / 三平面，同步保存 decoder 和 planes
            if use_decoder or use_Triplane:
                try:
                    save_decoder_and_planes(
                        decoder if use_decoder else None,
                        all_planes if use_Triplane else None,
                        ckpt_output_dir,
                        prefix=f"{time_idx}_",
                    )
                except Exception as e:
                    print(f"Warning: failed to save decoder/planes at checkpoint {time_idx}: {e}")
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        # Cache current frame for ORB+PnP init of the next timestep
        prev_frame_for_pnp = {
            'im': pose_eval_data['im'].detach().cpu(),
            'depth': pose_eval_data['depth'].detach().cpu()
        }

        torch.cuda.empty_cache()

    if config['save_timestamp_keyframes']:
        # Save keyframes selected at each timestamp
        max_length = max(len(inner) for inner in timestamp_keyframes)
        # Insert -1 for placeholder
        timestamp_keyframes_df = pd.DataFrame([inner + [-1 for _ in range(max_length - len(inner))] \
                                               for inner in timestamp_keyframes])
        timestamp_keyframes_df.to_csv(os.path.join(eval_dir, f"timestamp_keyframes.csv"), \
                                      index=False, header=False, na_rep='-1')

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if use_decoder:
            params = decode_gaussian_attributes(
                params, decoder, all_planes,
                use_rgb_Triplane=use_rgb_Triplane,
                decode_semantics=load_semantics and use_one_hot_semantics and use_semantic_decoder,
            )
        if config['use_wandb']:

            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'],
                 save_frames=True, use_one_hot_semantics=use_one_hot_semantics,
                 num_semantic_classes=num_semantic_classes)
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 device=device, load_semantics=load_semantics, eval_every=config['eval_every'],
                 save_frames=True, use_one_hot_semantics=use_one_hot_semantics,
                 num_semantic_classes=num_semantic_classes)

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    # if load_semantics:
    #     params['semantic_id'] = params['semantic_id'].type(torch.uint8)
    
    # Save Parameters (Gaussian params)
    save_params(params, output_dir, save_ply=False)
    if config.get("generate_pointcloud", True):
        torch.cuda.empty_cache()
        run_pointcloud_subprocess(config, output_dir, num_frames, device)
    # If decoder / tri-plane is used, also save decoder and planes
    if use_decoder or use_Triplane:
        try:
            save_decoder_and_planes(
                decoder if use_decoder else None,
                all_planes if use_Triplane else None,
                output_dir,
            )
        except Exception as e:
            print(f"Warning: failed to save decoder/planes at final save: {e}")

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))
    experiment.config["_experiment_path"] = os.path.abspath(args.experiment)
    experiment.config["_saved_experiment_path"] = os.path.join(results_dir, "config.py")

    rgbd_slam(experiment.config)
