import argparse
import os
import re
import sys
from importlib import util as importlib_util
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets import (  # noqa: E402
    Ai2thorDataset,
    AzureKinectDataset,
    ICLDataset,
    NeRFCaptureDataset,
    Record3DDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    RealsenseDataset,
    ScannetDataset,
    ScannetPPDataset,
    TUMDataset,
    load_dataset_config,
)
from utils.eval_helpers import evaluate_ate  # noqa: E402

DATASET_REGISTRY = {
    "icl": ICLDataset,
    "replica": ReplicaDataset,
    "replicav2": ReplicaV2Dataset,
    "azure": AzureKinectDataset,
    "azurekinect": AzureKinectDataset,
    "scannet": ScannetDataset,
    "ai2thor": Ai2thorDataset,
    "record3d": Record3DDataset,
    "realsense": RealsenseDataset,
    "tum": TUMDataset,
    "scannetpp": ScannetPPDataset,
    "nerfcapture": NeRFCaptureDataset,
}


def infer_valid_pred_length(params_path: Path, checkpoint_arg: Optional[int], max_len: int) -> int:
    """
    Determine how many predicted frames are valid for a checkpoint.
    """
    if max_len == 0:
        return 0
    if checkpoint_arg is not None:
        return max(1, min(checkpoint_arg + 1, max_len))
    stem = params_path.stem
    match = re.match(r"params(\d+)$", stem)
    if match:
        idx = int(match.group(1))
        return max(1, min(idx + 1, max_len))
    return max_len


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize camera trajectory on the X-Z plane.")
    parser.add_argument("--config", required=True,
                        help="Path to the experiment config file.")
    parser.add_argument(
        "--params", default=None, help="Optional explicit path to params npz file. Overrides --checkpoint."
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint index. "
        "For example, 0 -> params0.npz. If omitted the final params.npz is used unless --params is set.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. If relative, it will be placed inside the run directory determined by the config.",
    )
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for the saved figure.")
    parser.add_argument(
        "-sfp",
        "--show_frame_points",
        action="store_true",
        help="If set, overlay a scatter point for every ground-truth and estimated frame.",
    )
    return parser.parse_args()


def load_config(config_path: Path):
    spec = importlib_util.spec_from_file_location(
        "tgs_slam_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config module from {config_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise AttributeError(
            f"{config_path} does not define a 'config' dictionary")
    return module.config


def resolve_path(path_str: str, repo_root: Path, reference_dir: Path, description: str, must_exist: bool = True) -> Path:
    expanded = os.path.expanduser(path_str)
    candidate = Path(expanded)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = None
        for base in (reference_dir, repo_root, Path.cwd()):
            tentative = (base / expanded).resolve()
            if tentative.exists():
                resolved = tentative
                break
        if resolved is None:
            resolved = (repo_root / expanded).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Could not find {description} at {resolved}")
    return resolved


def configure_dataset(config: dict, repo_root: Path, config_dir: Path):
    data_cfg = config.get("data", {})
    if not data_cfg:
        raise ValueError("Config file does not define the 'data' section.")
    if "basedir" not in data_cfg or "sequence" not in data_cfg:
        raise ValueError(
            "The 'data' section must provide both 'basedir' and 'sequence'.")

    gradslam_cfg = data_cfg.get("gradslam_data_cfg")
    if gradslam_cfg:
        gradslam_cfg_path = resolve_path(
            gradslam_cfg, repo_root, config_dir, "dataset config file")
        dataset_cfg = load_dataset_config(str(gradslam_cfg_path))
    else:
        if "dataset_name" not in data_cfg:
            raise ValueError(
                "Either 'gradslam_data_cfg' or 'dataset_name' must be provided in the data config.")
        dataset_cfg = data_cfg

    dataset_name = dataset_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError(
            "Dataset configuration does not specify 'dataset_name'.")
    dataset_cls = DATASET_REGISTRY.get(dataset_name.lower())
    if dataset_cls is None:
        known = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Known datasets: {known}")

    basedir = resolve_path(
        data_cfg["basedir"], repo_root, config_dir, "dataset root directory")
    sequence = os.path.basename(data_cfg["sequence"].rstrip("/"))

    cam_params = dataset_cfg.get("camera_params", {})
    desired_height = (
        data_cfg.get("desired_image_height")
        or dataset_cfg.get("desired_image_height")
        or cam_params.get("image_height")
    )
    desired_width = (
        data_cfg.get("desired_image_width")
        or dataset_cfg.get("desired_image_width")
        or cam_params.get("image_width")
    )
    if desired_height is None or desired_width is None:
        raise ValueError(
            "Could not determine desired image resolution. Please specify desired_image_height/width in the config."
        )

    kwargs = {
        "stride": data_cfg.get("stride", 1),
        "start": data_cfg.get("start", 0),
        "end": data_cfg.get("end", -1),
        "desired_height": desired_height,
        "desired_width": desired_width,
        "device": "cpu",
        "relative_pose": True,
        "use_train_split": data_cfg.get("use_train_split", True),
        "load_semantics": data_cfg.get("load_semantics", False),
        "num_semantic_classes": data_cfg.get("num_semantic_classes", 0),
    }

    if "load_embeddings" in data_cfg:
        kwargs["load_embeddings"] = data_cfg["load_embeddings"]
    if "embedding_dir" in data_cfg:
        kwargs["embedding_dir"] = data_cfg["embedding_dir"]
    if "embedding_dim" in data_cfg:
        kwargs["embedding_dim"] = data_cfg["embedding_dim"]

    if dataset_name.lower() == "scannetpp":
        dataset = dataset_cls(
            str(basedir),
            sequence,
            ignore_bad=data_cfg.get("ignore_bad", False),
            use_train_split=data_cfg.get("use_train_split", True),
            stride=kwargs["stride"],
            start=kwargs["start"],
            end=kwargs["end"],
            desired_height=kwargs["desired_height"],
            desired_width=kwargs["desired_width"],
            load_semantics=kwargs["load_semantics"],
            num_semantic_classes=kwargs["num_semantic_classes"],
        )
    else:
        dataset = dataset_cls(dataset_cfg, str(basedir), sequence, **kwargs)
    requested_frames = data_cfg.get("num_frames", -1)
    if requested_frames == -1:
        requested_frames = len(dataset)
    else:
        requested_frames = min(requested_frames, len(dataset))
    return dataset, requested_frames, sequence


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = quat.astype(np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError(
            "Encountered near-zero quaternion while building rotation matrix.")
    w, x, y, z = q / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
             2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rot


def _reshape_pose_array(arr: np.ndarray, feature_dim: int) -> np.ndarray:
    data = np.asarray(arr)
    axes = [idx for idx, size in enumerate(data.shape) if size == feature_dim]
    if not axes:
        raise ValueError(
            f"Could not find feature dimension {feature_dim} in array shape {data.shape}")
    data = np.moveaxis(data, axes[0], -1)
    data = data.reshape(-1, feature_dim)
    return data


def load_predicted_w2c(params_path: Path):
    params = dict(np.load(params_path))
    if "cam_unnorm_rots" not in params or "cam_trans" not in params:
        raise KeyError(
            f"{params_path} does not contain camera pose parameters.")
    rots_seq = _reshape_pose_array(params["cam_unnorm_rots"], 4)
    trans_seq = _reshape_pose_array(params["cam_trans"], 3)
    if rots_seq.shape[0] != trans_seq.shape[0]:
        raise ValueError(
            "Mismatched number of predicted camera poses between rotations and translations.")

    w2c_list = []
    cam_positions = []
    for quat, tran in zip(rots_seq, trans_seq):
        rot = quat_to_rotmat(quat)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = rot
        w2c[:3, 3] = tran
        w2c_list.append(w2c)
        c2w = np.linalg.inv(w2c)
        cam_positions.append(c2w[:3, 3])
    return w2c_list, np.asarray(cam_positions, dtype=np.float32)


def determine_params_path(args, config: dict, repo_root: Path, config_dir: Path):
    if args.params:
        return resolve_path(args.params, repo_root, config_dir, "params file")
    workdir = config.get("workdir", "")
    run_name = config.get("run_name", "")
    if not workdir or not run_name:
        raise ValueError(
            "Config must define both 'workdir' and 'run_name' to locate params.")
    workdir_path = resolve_path(workdir, repo_root, config_dir, "workdir")
    run_dir = workdir_path / run_name
    params_name = "params.npz" if args.checkpoint is None else f"params{args.checkpoint}.npz"
    params_path = run_dir / params_name
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find params file at {params_path}")
    return params_path


def resolve_output_path(args, run_dir: Path, checkpoint: Optional[int]) -> Path:
    suffix = "final" if checkpoint is None else f"ckpt{checkpoint}"
    default_name = f"camera_traj_xz_{suffix}.png"
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = run_dir / output_path
    else:
        output_path = run_dir / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_trajectory(
    gt_pos: np.ndarray,
    est_pos: np.ndarray,
    ate_cm: float,
    sequence: str,
    output_path: Path,
    dpi: int,
    show_frame_points: bool = False,
):
    plt.figure(figsize=(8, 6))
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], label="Ground Truth",
             linewidth=2.2, color="#1f77b4")
    plt.plot(est_pos[:, 0], est_pos[:, 2], label="Estimated",
             linewidth=2.2, linestyle="--", color="#d62728")
    if show_frame_points:
        plt.scatter(
            gt_pos[:, 0],
            gt_pos[:, 2],
            s=12,
            color="#1f77b4",
            alpha=0.6,
            edgecolors="none",
            label="GT Frames",
        )
        if len(est_pos) > 0:
            plt.scatter(
                est_pos[:, 0],
                est_pos[:, 2],
                s=18,
                color="#d62728",
                alpha=0.7,
                edgecolors="none",
                marker="o",
                label="Estimated Frames",
            )
    plt.scatter(gt_pos[0, 0], gt_pos[0, 2], c="#1f77b4",
                edgecolors="k", s=40, zorder=3)
    plt.scatter(est_pos[0, 0], est_pos[0, 2], c="#d62728",
                edgecolors="k", s=40, zorder=3)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"{sequence} Camera Trajectory (ATE RMSE: {ate_cm:.2f} cm)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    repo_root = Path(_BASE_DIR)
    config_dir = config_path.parent

    params_path = determine_params_path(args, config, repo_root, config_dir)
    run_dir = params_path.parent

    dataset, requested_frames, sequence_name = configure_dataset(
        config, repo_root, config_dir)
    pred_w2c, pred_positions = load_predicted_w2c(params_path)

    if not pred_w2c:
        raise RuntimeError(f"No camera poses were found in {params_path}")

    gt_c2w_all = dataset.transformed_poses.detach().cpu()
    gt_positions_full = gt_c2w_all[:, :3, 3].numpy()
    available_gt = min(requested_frames, gt_c2w_all.shape[0])

    pred_valid_len = infer_valid_pred_length(
        params_path, args.checkpoint, len(pred_w2c))
    frames_to_use = min(pred_valid_len, available_gt)
    if frames_to_use == 0:
        raise RuntimeError(
            "No overlapping frames between predicted poses and dataset ground truth.")

    gt_c2w_subset = gt_c2w_all[:frames_to_use]
    gt_w2c_list = [torch.linalg.inv(gt_c2w_subset[i]).cpu()
                   for i in range(frames_to_use)]
    est_w2c_list = [torch.from_numpy(pred_w2c[i]).float()
                    for i in range(frames_to_use)]

    ate_rmse_m = float(evaluate_ate(gt_w2c_list, est_w2c_list))
    ate_rmse_cm = ate_rmse_m * 100.0

    output_path = resolve_output_path(args, run_dir, args.checkpoint)
    plot_trajectory(
        gt_positions_full,
        pred_positions[:pred_valid_len],
        ate_rmse_cm,
        sequence_name,
        output_path,
        args.dpi,
        args.show_frame_points,
    )

    print(f"Saved trajectory plot for {frames_to_use} frames to {output_path}")
    print(f"ATE RMSE: {ate_rmse_cm:.2f} cm")


if __name__ == "__main__":
    main()
