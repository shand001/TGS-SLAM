import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
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


def render_semantic_images(
    config,
    dataset,
    final_params,
    num_frames,
    out_dir,
    eval_every=1,
    sil_thres=0.5,
    min_conf=0.0,
    device=torch.device("cuda"),
):
    os.makedirs(out_dir, exist_ok=True)

    dataset_config = config["data"]
    load_semantics = dataset_config.get("load_semantics", False)
    use_one_hot_semantics = config.get("use_one_hot_semantics", False)
    num_semantic_classes = dataset_config.get("num_semantic_classes", 0)

    if not load_semantics or not use_one_hot_semantics:
        print(
            "Warning: load_semantics / use_one_hot_semantics not enabled in config; "
            "rendered semantics may be invalid."
        )

    mean, quat, scale, opac, color_feat = params_to_gsplat_inputs(
        final_params,
        final_params["means3D"],
        use_semantics=load_semantics,
        use_one_hot_semantics=use_one_hot_semantics,
    )

    for time_idx in tqdm(range(num_frames), desc="Rendering semantic images"):
        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        sample = dataset[time_idx]
        if len(sample) == 6:
            color, depth_gt, intrinsics, pose, semantic_id, semantic_color = sample
        else:
            color, depth_gt, intrinsics, pose = sample
        intrinsics = intrinsics[:3, :3]

        color_chw = color.permute(2, 0, 1) / 255.0
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

        rgbds = rgbds.squeeze(0).squeeze(0)  # [H,W,C]
        sem_logits = None
        if load_semantics:
            sem_logits = rgbds[..., 3:-1]  # [H,W,K]
            print("sem_logits",sem_logits.shape)
        if sem_logits is None or sem_logits.shape[-1] == 0:
            print(f"[Frame {time_idx}] No semantic channels in rendered output.")
            continue

        probs = torch.softmax(sem_logits, dim=-1)
        conf, labels = torch.max(probs, dim=-1)  # [H,W]
        
        # Optional confidence + silhouette masking
        if alpha.ndim == 5:
            silhouette = alpha[0, 0]
        elif alpha.ndim == 4:
            silhouette = alpha[0]
        else:
            silhouette = alpha
        if silhouette.ndim == 3 and silhouette.shape[-1] == 1:
            silhouette = silhouette[..., 0]
        sil_mask = silhouette > sil_thres

        if min_conf > 0.0:
            conf_mask = conf > min_conf
            valid_mask = sil_mask & conf_mask
        else:
            valid_mask = sil_mask

        labels_np = labels.cpu().numpy().astype(np.int32)
        labels_np[~valid_mask.cpu().numpy()] = 0

        if num_semantic_classes <= 0:
            num_semantic_classes = int(labels_np.max()) + 1

        color_img = decode_segmap(labels_np, nc=num_semantic_classes)  # [H,W,3] in [0,1]
        color_img = (color_img * 255.0).astype(np.uint8)
        color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(out_dir, f"semantic_{time_idx:06d}.png")
        cv2.imwrite(out_path, color_img_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Render per-frame semantic images using gsplat and camera poses "
            "stored in params.npz."
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
            "Output directory for semantic images "
            "(default: <workdir>/<run_name>/semantic_renders)."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to render (default: config/data setting).",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Render every N-th frame (default: 1).",
    )
    parser.add_argument(
        "--sil_thres",
        type=float,
        default=0.5,
        help="Silhouette threshold to mask background (default: 0.5).",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.0,
        help="Minimum semantic confidence per pixel (softmax) to keep (default: 0.0).",
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
    out_dir = args.out_dir or os.path.join(output_root, "semantic_renders")
    final_params = load_final_params_from_npz(params_path, device)

    render_semantic_images(
        config=config,
        dataset=dataset,
        final_params=final_params,
        num_frames=num_frames,
        out_dir=out_dir,
        eval_every=args.eval_every,
        sil_thres=args.sil_thres,
        min_conf=args.min_conf,
        device=device,
    )

