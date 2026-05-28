import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from gsplat import rasterization as rasterization  # noqa: E402

from scripts.gen_pointcloud_tsdf import (  # noqa: E402
    build_dataset_from_config,
    load_final_params_from_npz,
)
from utils.common_utils import seed_everything  # noqa: E402
from utils.slam_helpers import params_to_gsplat_inputs  # noqa: E402
from utils.slam_external import build_rotation  # noqa: E402


def decode_segmap(image, nc=25):
    label_colors = np.array(
        [
            (0, 0, 0),
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

    max_index = min(nc, label_colors.shape[0])
    image = np.asarray(image, dtype=np.int64)
    image = np.clip(image, 0, max_index - 1)
    rgb = label_colors[image]
    return rgb.astype(np.float32) / 255.0


def backproject_to_world(depth, labels, intrinsics, w2c, sil_mask=None, conf_mask=None):
    """
    depth: [H,W] tensor (on device)
    labels: [H,W] tensor (int)
    intrinsics: [3,3] tensor (on device)
    w2c: [4,4] tensor (world -> cam, on device)
    sil_mask: [H,W] bool tensor or None
    """
    device = depth.device
    H, W = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x_grid, y_grid = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_flat = depth.reshape(-1)

    pts_cam = torch.stack((xx * depth_flat, yy * depth_flat, depth_flat), dim=-1)

    pix_ones = torch.ones(H * W, 1, device=device).float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(w2c)
    pts_world = (c2w @ pts4.T).T[:, :3]

    labels_flat = labels.reshape(-1)

    valid_mask = depth_flat > 0
    valid_mask &= ~torch.isnan(depth_flat)
    if sil_mask is not None:
        valid_mask &= sil_mask.reshape(-1)
    if conf_mask is not None:
        valid_mask &= conf_mask.reshape(-1)

    pts_world = pts_world[valid_mask]
    labels_flat = labels_flat[valid_mask]
    return pts_world, labels_flat


def generate_semantic_pointcloud_from_render(
    config,
    dataset,
    final_params,
    num_frames,
    out_dir,
    eval_every=1,
    sil_thres=0.5,
    min_conf=0.5,
    voxel_size=0.03,
    device=torch.device("cuda"),
):
    os.makedirs(out_dir, exist_ok=True)
    out_pcd_path = os.path.join(out_dir, "semantic_render_points.ply")

    dataset_config = config["data"]
    load_semantics = dataset_config.get("load_semantics", False)
    use_one_hot_semantics = config.get("use_one_hot_semantics", False)
    num_semantic_classes = dataset_config.get("num_semantic_classes", 0)

    if not load_semantics or not use_one_hot_semantics:
        print(
            "Warning: load_semantics / use_one_hot_semantics not enabled in config; "
            "rendered semantics may be invalid."
        )

    all_pts = []
    all_labels = []

    for time_idx in tqdm(range(num_frames), desc="Collecting semantic surfels"):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        sample = dataset[time_idx]
        if len(sample) == 6:
            color, depth_gt, intrinsics, pose, semantic_id, semantic_color = sample
        else:
            color, depth_gt, intrinsics, pose = sample
        intrinsics = intrinsics[:3, :3]

        color_chw = color.permute(2, 0, 1) / 255.0
        depth_gt_chw = depth_gt.permute(2, 0, 1)

        H = color_chw.shape[1]
        W = color_chw.shape[2]

        with torch.no_grad():
            curr_cam_rot = F.normalize(
                final_params["cam_unnorm_rots"][..., time_idx].detach()
            )
            curr_cam_tran = final_params["cam_trans"][..., time_idx].detach()
            curr_w2c = torch.eye(4, device=device).float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

        mean, quat, scale, opac, color_feat = params_to_gsplat_inputs(
            final_params,
            final_params["means3D"],
            use_semantics=load_semantics,
            use_one_hot_semantics=use_one_hot_semantics,
        )

        Ks = intrinsics.to(device=device, dtype=torch.float32)[None]
        views = curr_w2c[None]

        rgbds, alpha, _ = rasterization(
            mean,
            quat,
            scale,
            opac,
            color_feat,
            views,
            Ks,
            W,
            H,
            render_mode="RGB+D",
        )

        rgbds = rgbds.squeeze(0).squeeze(0)  # -> [H,W,C]
        depth_pred = rgbds[..., -1]          # [H,W]
        sem_logits = None
        if load_semantics:
            sem_logits = rgbds[..., 3:-1]    # [H,W,K]

        # Match silhouette handling in scripts/slam.py
        if alpha.ndim == 5:
            silhouette = alpha[0, 0]  # [H,W,1] or [H,W]
        elif alpha.ndim == 4:
            silhouette = alpha[0]     # [H,W,1] or [H,W]
        else:
            silhouette = alpha
        if silhouette.ndim == 3 and silhouette.shape[-1] == 1:
            silhouette = silhouette[..., 0]
        sil_mask = silhouette > sil_thres   # [H,W]

        if sem_logits is None:
            continue

        # semantic predictions with confidence
        probs = torch.softmax(sem_logits, dim=-1)
        conf, labels = torch.max(probs, dim=-1)  # [H,W]
        conf_mask = conf > min_conf

        pts_world, labels_world = backproject_to_world(
            depth_pred,
            labels,
            intrinsics.to(device=device),
            curr_w2c,
            sil_mask=sil_mask,
            conf_mask=conf_mask,
        )

        if pts_world.numel() == 0:
            continue

        all_pts.append(pts_world.cpu().numpy())
        all_labels.append(labels_world.cpu().numpy())

    if not all_pts:
        print("No valid semantic surfels collected; nothing to save.")
        return

    pts = np.concatenate(all_pts, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Collected {pts.shape[0]} semantic surfels before voxel voting.")

    if num_semantic_classes <= 0:
        num_semantic_classes = int(labels.max()) + 1

    # --- 3D majority vote in voxel grid ---
    voxel_size = float(voxel_size)
    coords = np.floor(pts / voxel_size).astype(np.int32)
    from collections import defaultdict

    sum_pos = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    count_pos = defaultdict(int)
    count_labels = defaultdict(lambda: np.zeros(num_semantic_classes, dtype=np.int64))

    for p, l, c in zip(pts, labels, coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        sum_pos[key] += p
        count_pos[key] += 1
        if 0 <= l < num_semantic_classes:
            count_labels[key][int(l)] += 1

    voted_pts = []
    voted_labels = []
    for key in sum_pos.keys():
        cnt = count_pos[key]
        if cnt == 0:
            continue
        pos = sum_pos[key] / float(cnt)
        label_counts = count_labels[key]
        if label_counts.sum() == 0:
            continue
        lbl = int(label_counts.argmax())
        voted_pts.append(pos)
        voted_labels.append(lbl)

    if not voted_pts:
        print("No voxels after voting; nothing to save.")
        return

    voted_pts = np.stack(voted_pts, axis=0)
    voted_labels = np.asarray(voted_labels, dtype=np.int32)
    print(f"After voxel voting: {voted_pts.shape[0]} voxels.")

    colors = decode_segmap(voted_labels, nc=num_semantic_classes)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voted_pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    o3d.io.write_point_cloud(out_pcd_path, pcd)
    print(f"Saved rendered semantic point cloud to: {out_pcd_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate 3D semantic point cloud by backprojecting "
            "rendered semantic maps and depths (gsplat)."
        )
    )
    parser.add_argument(
        "experiment", type=str, help="Path to TGS-SLAM experiment config (.py)."
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to params.npz (default: <workdir>/<run_name>/params.npz).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory "
            "(default: <workdir>/<run_name>/semantic_render_pointcloud)."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to integrate (default: config/data setting).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Use every N-th frame (default: 1).",
    )
    parser.add_argument(
        "--sil_thres",
        type=float,
        default=0.5,
        help="Silhouette threshold for valid surfels (default: 0.5).",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.5,
        help="Minimum semantic confidence per pixel (softmax) to keep (default: 0.5).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel size for 3D majority voting in meters (default: 0.03).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (default: config['primary_device']).",
    )

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    config = experiment.config
    seed_everything(seed=config["seed"])

    device_str = args.device or config.get("primary_device", "cuda:0")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    dataset, default_num_frames = build_dataset_from_config(config, device)
    num_frames = args.num_frames or default_num_frames

    output_root = os.path.join(config["workdir"], config["run_name"])
    params_path = args.params or os.path.join(output_root, "params.npz")
    out_dir = args.out_dir or os.path.join(
        output_root, "semantic_render_pointcloud"
    )
    final_params = load_final_params_from_npz(params_path, device)

    generate_semantic_pointcloud_from_render(
        config=config,
        dataset=dataset,
        final_params=final_params,
        num_frames=num_frames,
        out_dir=out_dir,
        eval_every=args.eval_every,
        sil_thres=args.sil_thres,
        min_conf=args.min_conf,
        voxel_size=args.voxel_size,
        device=device,
    )
