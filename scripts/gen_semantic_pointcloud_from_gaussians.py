import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import open3d as o3d
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from utils.common_utils import seed_everything  # noqa: E402


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
    # image: [N] int
    image = np.asarray(image, dtype=np.int64)
    image = np.clip(image, 0, max_index - 1)
    rgb = label_colors[image]
    return rgb.astype(np.float32) / 255.0


def load_gaussian_semantics(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "means3D" not in data or "semantic_id" not in data:
        raise KeyError("params.npz must contain 'means3D' and 'semantic_id'")

    means3d = data["means3D"].astype(np.float32)  # [N,3]
    sem_id = data["semantic_id"]  # [N,K] one-hot or logits

    if sem_id.ndim != 2:
        raise ValueError(f"semantic_id must be 2D [N,K], got shape {sem_id.shape}")
    num_classes = sem_id.shape[1]

    # 若是 uint8 的 one-hot，直接 argmax；若是 float/logit，同样 argmax
    if sem_id.dtype != np.float32 and sem_id.dtype != np.float64:
        sem_id_np = sem_id.astype(np.float32)
    else:
        sem_id_np = sem_id

    labels = np.argmax(sem_id_np, axis=1).astype(np.int32)  # [N]
    return means3d, labels, num_classes


def build_semantic_pointcloud(means3d, labels, num_classes, use_palette=True):
    if use_palette:
        colors = decode_segmap(labels, nc=num_classes)  # [N,3] in [0,1]
    else:
        # 简单按类别 id 归一化成灰度色
        max_label = max(1, labels.max())
        g = labels.astype(np.float32) / max_label
        colors = np.stack([g, g, g], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means3d.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D semantic point cloud directly from Gaussian semantics (no TSDF)."
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
        help="Output directory (default: <workdir>/<run_name>/gaussian_semantic_pointcloud).",
    )
    parser.add_argument(
        "--no_palette",
        action="store_true",
        help="If set, use grayscale by class id instead of predefined color palette.",
    )

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    config = experiment.config
    seed_everything(seed=config["seed"])

    output_root = os.path.join(config["workdir"], config["run_name"])
    params_path = args.params or os.path.join(output_root, "params.npz")
    out_dir = args.out_dir or os.path.join(
        output_root, "gaussian_semantic_pointcloud"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading params from: {params_path}")
    means3d, labels, num_classes = load_gaussian_semantics(params_path)
    print(f"Loaded {means3d.shape[0]} gaussians, {num_classes} semantic classes.")

    pcd = build_semantic_pointcloud(
        means3d, labels, num_classes, use_palette=not args.no_palette
    )

    out_ply = os.path.join(out_dir, "semantic_gaussians.ply")
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved semantic Gaussian point cloud to: {out_ply}")

