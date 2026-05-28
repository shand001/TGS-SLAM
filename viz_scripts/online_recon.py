from utils.slam_external import build_rotation
from utils.slam_helpers import get_depth_and_silhouette, params_to_gsplat_inputs
from utils.recon_helpers import to_gsplat_camera
from utils.common_utils import seed_everything
from gsplat import rasterization
import torch.nn.functional as F
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
import os
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)


def semantic_id_to_colors(semantic_labels, num_classes):
    cmap_name = "tab20" if num_classes <= 20 else "gist_rainbow"
    cmap = plt.get_cmap(cmap_name, num_classes)
    palette = torch.tensor(cmap(np.arange(num_classes))[
                           :, :3], device=semantic_labels.device, dtype=torch.float32)
    labels = semantic_labels.long().view(-1).clamp(min=0, max=num_classes - 1)
    return palette[labels].view(*semantic_labels.shape, 3)


def transform_points_to_camera(pts_world, w2c, device="cuda"):
    pts_ones = torch.ones(
        pts_world.shape[0], 1, device=device, dtype=torch.float32)
    pts4 = torch.cat((pts_world, pts_ones), dim=1)
    w2c_t = torch.as_tensor(w2c, device=device, dtype=torch.float32)
    cam_pts = (w2c_t @ pts4.T).T[:, :3]
    return cam_pts


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(
        all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    return params, all_w2cs


def get_rendervars(params, w2c, curr_timestep, load_semantics=False):
    params_timesteps = params['timestep']
    selected_params_idx = params_timesteps <= curr_timestep
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]
    transformed_pts = selected_params['means3D']
    w2c = torch.tensor(w2c).cuda().float()
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': selected_params['rgb_colors'],
        'rgb_colors': selected_params['rgb_colors'],
        'unnorm_rotations': selected_params['unnorm_rotations'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'logit_opacities': selected_params['logit_opacities'],
        'log_scales': selected_params['log_scales'],
        'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rgb_colors': get_depth_and_silhouette(transformed_pts, w2c),
        'unnorm_rotations': selected_params['unnorm_rotations'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'logit_opacities': selected_params['logit_opacities'],
        'log_scales': selected_params['log_scales'],
        'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }

    if load_semantics:
        semantic_colors = None
        semantic_ids = None
        if 'semantic_id' in selected_params:
            semantic_logits = selected_params['semantic_id'].float()
            if semantic_logits.dim() == 1:
                semantic_logits = semantic_logits.unsqueeze(-1)
            semantic_probs = torch.softmax(semantic_logits, dim=-1)
            semantic_ids = torch.argmax(semantic_probs, dim=-1, keepdim=True)
            num_sem_classes = semantic_probs.shape[-1]
            semantic_colors = semantic_id_to_colors(
                semantic_ids.squeeze(-1), num_sem_classes).squeeze(1)
        elif 'semantic_colors' in selected_params:
            semantic_colors = selected_params['semantic_colors']
        if semantic_colors is not None:
            semantic_rendervar = {
                'means3D': transformed_pts,
                'colors_precomp': semantic_colors,
                'rgb_colors': semantic_colors,
                'semantic_colors': semantic_colors,
                'semantic_id': semantic_ids,
                'unnorm_rotations': selected_params['unnorm_rotations'],
                'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
                'opacities': torch.sigmoid(selected_params['logit_opacities']),
                'logit_opacities': selected_params['logit_opacities'],
                'log_scales': selected_params['log_scales'],
                'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
                'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
            }
        else:
            semantic_rendervar = None
    else:
        semantic_rendervar = None

    return rendervar, depth_rendervar, semantic_rendervar


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack(
            (pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(
            np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, cfg, device="cuda", use_semantics=False):
    with torch.no_grad():
        cam_pts = transform_points_to_camera(
            timestep_data['means3D'], w2c, device=device)
        mean, quat, scale, opac, color = params_to_gsplat_inputs(
            timestep_data,
            cam_pts,
            use_semantics=use_semantics,
            use_one_hot_semantics=cfg.get('use_one_hot_semantics', False),
            use_rgb_Triplane=False,
            use_sem_Triplane=False,
        )
        H, W, K, view = to_gsplat_camera(
            cfg['viz_w'], cfg['viz_h'], k, w2c, device=device)
        RGBDS, alpha, _ = rasterization(
            mean, quat, scale, opac, color, view, K, W, H,
            render_mode="RGB+D"
        )
        im = RGBDS[..., :3].squeeze(0).permute(2, 0, 1)
        depth = RGBDS[..., -1].squeeze(0).unsqueeze(0)
        sil = alpha[0] if alpha.ndim == 4 else alpha
        sil = sil.squeeze(0).permute(2, 0, 1)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + \
            (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(
            cols.contiguous().double().cpu().numpy())

    return pts, cols


def visualize(scene_path, cfg):
    # Load Scene Data
    first_frame_w2c, k = load_camera(cfg, scene_path)

    params, all_w2cs = load_scene_data(scene_path)
    print(params['means3D'].shape)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    load_semantics = cfg['load_semantics']
    scene_data, scene_depth_data, scene_semantic_data = get_rendervars(params, first_frame_w2c,
                                                                       curr_timestep=0,
                                                                       load_semantics=load_semantics)
    im, depth, sil = render(
        first_frame_w2c, k, scene_data, cfg,
        use_semantics=cfg.get('use_one_hot_semantics', False))
    init_pts, init_cols = rgbd2pcd(im, depth, first_frame_w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('autumn')
    norm_factor = 0.5
    total_num_lines = num_t - 1
    line_colormap = plt.get_cmap('cool')

    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c
    first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(
        cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    while curr_timestep < (num_timesteps-1) or not cfg['enter_interactive_post_online']:
        passed_time = time.time() - start_time
        passed_frames = passed_time * cfg['viz_fps']
        curr_timestep = int(passed_frames % num_timesteps)
        if not viz_start:
            if curr_timestep == prev_timestep:
                continue

        # Update Camera Frustum
        if curr_timestep == 0:
            cam_centers = []
            if not viz_start:
                vis.remove_geometry(prev_lines)
        if not viz_start:
            vis.remove_geometry(prev_frustum)
        new_frustum = o3d.geometry.LineSet.create_camera_visualization(
            w, h, k, all_w2cs[curr_timestep], frustum_size)
        new_frustum.paint_uniform_color(
            np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3]))
        vis.add_geometry(new_frustum)
        prev_frustum = new_frustum
        cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])

        # Update Camera Trajectory
        if len(cam_centers) > 1 and curr_timestep > 0:
            num_lines = [1]
            cols = []
            for line_t in range(curr_timestep):
                cols.append(np.array(line_colormap(
                    (line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)
            prev_lines = lines
        elif not viz_start:
            vis.remove_geometry(prev_lines)

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        view_w2c = cam_params.extrinsic
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        view_w2c[0:3, 3] += [0, 0, 0.5]

        cam_params.extrinsic = view_w2c
        view_control.convert_from_pinhole_camera_parameters(
            cam_params, allow_arbitrary=True)

        scene_data, scene_depth_data, scene_semantic_data = get_rendervars(params, view_w2c,
                                                                           curr_timestep=curr_timestep,
                                                                           load_semantics=load_semantics)
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(
                scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        elif cfg['render_mode'] == 'semantic_color':
            if scene_semantic_data is None:
                seg, depth, sil = render(
                    view_w2c, k, scene_data, cfg,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
            else:
                seg, depth, sil = render(
                    view_w2c, k, scene_semantic_data, cfg,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
            if cfg['show_sil']:
                seg = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(seg, depth, view_w2c, k, cfg)
        else:
            im, depth, sil = render(
                view_w2c, k, scene_data, cfg,
                use_semantics=cfg.get('use_one_hot_semantics', False))
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
        prev_timestep = curr_timestep
        viz_start = False

    # Enter Interactive Mode once all frames have been visualized
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(
                scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        elif cfg['render_mode'] == 'semantic_color':
            if scene_semantic_data is None:
                seg, depth, sil = render(
                    w2c, k, scene_data, cfg,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
            else:
                seg, depth, sil = render(
                    w2c, k, scene_semantic_data, cfg,
                    use_semantics=cfg.get('use_one_hot_semantics', False))
            if cfg['show_sil']:
                seg = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(seg, depth, w2c, k, cfg)
        else:
            im, depth, sil = render(
                w2c, k, scene_data, cfg,
                use_semantics=cfg.get('use_one_hot_semantics', False))
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    # Visualize Final Reconstruction
    visualize(scene_path, viz_cfg)
