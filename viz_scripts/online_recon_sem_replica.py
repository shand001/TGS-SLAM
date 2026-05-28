import os
import sys
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)
import cv2
import imageio
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
import time
from utils.slam_external import build_rotation
from utils.common_utils import seed_everything
from gsplat import rasterization as rasterization
import torch.nn.functional as F
import torch
import open3d as o3d
import numpy as np
from copy import deepcopy
import argparse



LABEL_COLORS = np.array(
    [
        (191, 202, 230),
        (174, 199, 232),
        (152, 223, 138),
        (31, 119, 180),
        (255, 187, 120),
        (188, 189, 34),
        (140, 86, 75),
        (255, 152, 150),
        (214, 39, 40),
        (197, 176, 213),
        (148, 103, 189),
        (196, 156, 148),
        (23, 190, 207),
        (178, 76, 76),
        (247, 182, 210),
        (66, 188, 102),
        (219, 219, 141),
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),
        (112, 128, 144),
        (96, 207, 209),
        (227, 119, 194),
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),
        (100, 85, 144),
        (100, 218, 200),
        (255, 179, 0),
        (144, 238, 144),
        (135, 206, 235),
        (255, 105, 180),
        (106, 90, 205),
        (255, 165, 0),
        (72, 209, 204),
        (199, 21, 133),
        (70, 130, 180),
        (255, 99, 71),
        (147, 112, 219),
        (60, 179, 113),
        (220, 20, 60),
    ],
    dtype=np.uint8,
)


def decode_segmap(image, nc=52):
    max_index = min(nc, LABEL_COLORS.shape[0])
    image = np.asarray(image, dtype=np.int64)
    image = np.clip(image, 0, max_index - 1)
    rgb = LABEL_COLORS[image]
    return rgb.astype(np.float32) / 255.0


def semantic_labelmapvis(im_semantic_in):
    # im_semantic_in: [H,W] int labels
    im_semantic_show = im_semantic_in.detach().cpu().numpy().copy()
    # [H,W,3] in [0,1]
    rgb = decode_segmap(im_semantic_show, nc=LABEL_COLORS.shape[0])
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    rgb_uint8 = np.squeeze(rgb_uint8)
    im_semantic_labelmap = torch.from_numpy(rgb_uint8).cuda()
    im_semantic_labelmap = im_semantic_labelmap.permute(2, 0, 1)
    return im_semantic_labelmap


def process_semantic(im_semantic_in):
    # im_semantic_in: [C, H, W] logits / scores
    im_semantic_out = torch.argmax(im_semantic_in, dim=0)
    return im_semantic_out


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
    print("load parans from: ", scene_path)
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


def get_rendervars_semantic(params, w2c, curr_timestep):
    params_timesteps = params['timestep']
    selected_params_idx = params_timesteps <= curr_timestep
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]

    if selected_params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(selected_params['log_scales'], (1, 3))
    else:
        log_scales = selected_params['log_scales']

    rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': selected_params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda"),
    }

    # Attach per-Gaussian semantic features if available
    if 'semantic' in selected_params:
        rendervar['semantics_precomp'] = selected_params['semantic']
    elif 'semantic_id' in selected_params:
        rendervar['semantics_precomp'] = selected_params['semantic_id']
    elif 'semantic_colors' in selected_params:
        rendervar['semantics_precomp'] = selected_params['semantic_colors']

    return rendervar


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


def render_semantic(w2c, k, timestep_data, cfg):
    with torch.no_grad():
        means = timestep_data['means3D']
        rots = timestep_data['rotations']
        scales = timestep_data['scales']
        opacities = timestep_data['opacities'].reshape(-1)
        colors_rgb = timestep_data['colors_precomp']

        semantics = timestep_data.get('semantics_precomp', None)
        if semantics is not None:
            colors = torch.cat([colors_rgb, semantics], dim=-1)
        else:
            colors = colors_rgb

        device = means.device
        dtype = means.dtype

        view = torch.as_tensor(w2c, device=device, dtype=dtype).unsqueeze(0)
        K = torch.as_tensor(k, device=device, dtype=dtype).unsqueeze(0)
        H, W = int(cfg['viz_h']), int(cfg['viz_w'])

        RGBDS, alpha, _ = rasterization(
            means,
            rots,
            scales,
            opacities,
            colors,
            view,
            K,
            W,
            H,
            near_plane=cfg.get('viz_near', 0.01),
            far_plane=cfg.get('viz_far', 100.0),
            render_mode="RGB+D",
        )

        im = RGBDS[..., :3].squeeze(0).permute(2, 0, 1)
        depth = RGBDS[..., -1].squeeze(0).unsqueeze(0)

        if semantics is not None:
            im_semantic = RGBDS[..., 3:-1].squeeze(0).permute(2, 0, 1)
        else:
            im_semantic = None

        sil = alpha
        if sil.ndim == 4:
            sil = sil[0]
        sil = sil.squeeze(-1).unsqueeze(0)

        return im, depth, im_semantic, sil


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


def visualize(scene_path, cfg, mlp_func=None, flag_save=False,
              enable_vis=True, save_video=False, video_fps=10.0):

    source_save_path = os.path.dirname(scene_path)
    if flag_show_semantic:
        sem_save_path = "3d_vis/semantic/source/"
    else:
        sem_save_path = "3d_vis/rgb/source/"
    save_path = os.path.join(source_save_path, sem_save_path)
    print("save at: ===>> ", save_path)
    if flag_save:
        os.makedirs(save_path, exist_ok=True)

    # Load Scene Data
    first_frame_w2c, k = load_camera(cfg, scene_path)

    params, all_w2cs = load_scene_data(scene_path)
    print(params['means3D'].shape)

    vis = None
    view_control = None
    render_options = None
    if enable_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                          height=int(cfg['viz_h'] * cfg['view_scale']),
                          visible=True)

    scene_data = get_rendervars_semantic(
        params, first_frame_w2c, curr_timestep=0)
    im, depth, im_semantic, sil = render_semantic(
        first_frame_w2c, k, scene_data, cfg)
    if flag_show_semantic:
        im_semantic = process_semantic(im_semantic)
        im_semantic_labelmap = semantic_labelmapvis(im_semantic)
        im_semantic_labelmap = im_semantic_labelmap.to(im.dtype)
        im_semantic_labelmap = im_semantic_labelmap/255
        im = im_semantic_labelmap.clone()
    init_pts, init_cols = rgbd2pcd(
        im, depth, first_frame_w2c, k, cfg)      # Get 1st frame 3d pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    if enable_vis:
        vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('cool')
    norm_factor = 0.5
    total_num_lines = num_t - 1
    line_colormap = plt.get_cmap('cool')

    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    if enable_vis:
        view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c
    first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    if enable_vis:
        view_control.convert_from_pinhole_camera_parameters(
            cparams, allow_arbitrary=True)

        render_options = vis.get_render_option()
        render_options.point_size = cfg['view_scale']
        render_options.light_on = False

    # Optional video writer
    video_writer = None

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    while curr_timestep < (num_timesteps-1) or not cfg['enter_interactive_post_online']:

        if not viz_start:
            if curr_timestep == prev_timestep:
                continue
        print("curr_timestep is: ", curr_timestep)

        # Update Camera Frustum
        if enable_vis:
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
        if enable_vis:
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
        if enable_vis:
            cam_params = view_control.convert_to_pinhole_camera_parameters()
            view_k = cam_params.intrinsic.intrinsic_matrix
            k = view_k / cfg['view_scale']
            k[2, 2] = 1
            view_w2c = cam_params.extrinsic
            view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
            cam_params.extrinsic = view_w2c
            view_control.convert_from_pinhole_camera_parameters(
                cam_params, allow_arbitrary=True)
        else:
            view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])

        scene_data = get_rendervars_semantic(
            params, view_w2c, curr_timestep=curr_timestep)
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(
                scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(
                scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, im_semantic, sil = render_semantic(
                view_w2c, k, scene_data, cfg)
            if flag_show_semantic:
                im_semantic = process_semantic(im_semantic)
                im_semantic_labelmap = semantic_labelmapvis(im_semantic)
                im_semantic_labelmap = im_semantic_labelmap.to(im.dtype)
                im_semantic_labelmap = im_semantic_labelmap/255
                im = im_semantic_labelmap.clone()

            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        if enable_vis:
            vis.update_geometry(pcd)

        if enable_vis:
            if not vis.poll_events():
                break
            vis.update_renderer()
        prev_timestep = curr_timestep
        viz_start = False

        if flag_save:
            # Base frame from current rendered image
            frame = im.detach().cpu().permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0.0, 1.0)
            frame_uint8 = (frame * 255.0).astype(np.uint8)

            # Overlay camera trajectory and current pose in image space
            if len(cam_centers) > 0:
                pts_world = np.asarray(cam_centers, dtype=np.float32)  # [N,3]
                ones = np.ones((pts_world.shape[0], 1), dtype=np.float32)
                pts4 = np.concatenate([pts_world, ones], axis=1).T  # [4,N]

                cam_mat = view_w2c @ pts4  # [4,N]
                cam_xyz = cam_mat[:3, :].T  # [N,3]
                z = cam_xyz[:, 2]
                valid = z > 0
                cam_xyz = cam_xyz[valid]
                if cam_xyz.shape[0] > 0:
                    uvw = (k @ cam_xyz.T).T  # [N,3]
                    u = (uvw[:, 0] / uvw[:, 2]).astype(np.int32)
                    v = (uvw[:, 1] / uvw[:, 2]).astype(np.int32)
                    H_img, W_img = frame_uint8.shape[:2]
                    pts_2d = [
                        (ui, vi)
                        for ui, vi in zip(u, v)
                        if 0 <= ui < W_img and 0 <= vi < H_img
                    ]

                    # Draw trajectory as connected line (purple), skip large jumps
                    max_jump = int(0.1 * min(H_img, W_img))
                    for p0, p1 in zip(pts_2d[:-1], pts_2d[1:]):
                        dx = p0[0] - p1[0]
                        dy = p0[1] - p1[1]
                        if dx * dx + dy * dy > max_jump * max_jump:
                            continue
                        cv2.line(frame_uint8, p0, p1, (255, 0, 255), 1)

                    # Draw current camera as a rectangle with diagonals (sky blue)
                    if pts_2d:
                        cx, cy = pts_2d[-1]
                        # Rectangle size relative to image (slightly larger)
                        base_size = int(min(H_img, W_img) * 0.06)
                        half_w = base_size
                        half_h = int(base_size * 0.6)

                        left = max(0, cx - half_w)
                        right = min(W_img - 1, cx + half_w)
                        top = max(0, cy - half_h)
                        bottom = min(H_img - 1, cy + half_h)

                        p_tl = (left, top)
                        p_tr = (right, top)
                        p_br = (right, bottom)
                        p_bl = (left, bottom)

                        rect_color = (18, 237, 255)  # sky blue in BGR

                        # Outer rectangle with thin line
                        cv2.rectangle(frame_uint8, p_tl, p_br, rect_color, 1)
                        # Diagonals
                        cv2.line(frame_uint8, p_tl, p_br, rect_color, 1)
                        cv2.line(frame_uint8, p_tr, p_bl, rect_color, 1)

                        # Ensure trajectory line meets rectangle center
                        center = (cx, cy)
                        if len(pts_2d) >= 2:
                            prev_pt = pts_2d[-2]
                            dx = prev_pt[0] - center[0]
                            dy = prev_pt[1] - center[1]
                            if dx * dx + dy * dy <= max_jump * max_jump:
                                cv2.line(frame_uint8, prev_pt,
                                         center, (146, 109, 255), 1)

                        # Small center point
                        cv2.circle(frame_uint8, center, 2, rect_color, -1)

            # Turn true background (no depth) to white,
            # keep valid label-0 (black) pixels unchanged
            depth_np = depth.detach().cpu().numpy()[0]  # [H,W]
            bg_mask = depth_np <= 0.0
            frame_uint8[bg_mask] = np.array([255, 255, 255], dtype=np.uint8)

            frame_path = os.path.join(save_path, f"{curr_timestep}.png")
            imageio.imwrite(frame_path, frame_uint8)

            # Append to video if requested
            if save_video:
                if video_writer is None:
                    H_img, W_img = frame_uint8.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_path = os.path.join(
                        source_save_path, sem_save_path, "mapping_video.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, video_fps, (W_img, H_img))
                bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr)
        curr_timestep = curr_timestep+1

    # Cleanup
    if enable_vis:
        vis.destroy_window()
        del view_control
        del vis
        del render_options

    if video_writer is not None:
        video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")
    parser.add_argument("--flag_semantic",
                        action="store_true",
                        help="visualize semantics instead of RGB")
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Do not open Open3D window, only render and save frames/video.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save an MP4 video (mapping_video.mp4) from rendered frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for saved video (default: use viz.viz_fps or 10).",
    )

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    flag_show_semantic = args.flag_semantic

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    enable_vis = not args.no_vis
    video_fps = args.fps if args.fps is not None else viz_cfg.get(
        "viz_fps", 10)

    MLP_func = None
    # Visualize Final Reconstruction
    visualize(
        scene_path,
        viz_cfg,
        flag_save=True,
        enable_vis=enable_vis,
        save_video=args.save_video,
        video_fps=video_fps,
    )
