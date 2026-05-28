#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt


def build_rotation_from_quat(quat):
    """
    将四元数转为旋转矩阵。
    quat: (N, 4)  假定顺序为 [w, x, y, z]
    返回: (N, 3, 3)
    """
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)  # 归一化

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # 3DGS / SplaTAM 常用的 wxyz -> R 公式
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = np.zeros((quat.shape[0], 3, 3), dtype=np.float64)

    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)

    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)

    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    return R


def umeyama_alignment(src, dst, with_scale=True):
    """
    Umeyama 相似变换对齐: src -> dst
    src, dst: (N, 3)
    返回: s, R, t  使得 dst ≈ s * R @ src^T + t
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)

    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = (dst_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0

    t = mean_dst - scale * (R @ mean_src)

    return scale, R, t


def extract_cam_centers_from_w2c(w2c):
    """
    从 world-to-camera 矩阵中提取相机中心在世界坐标下的位置
    w2c: (N, 4, 4)
    返回: (N, 3) 相机中心 C = -R^T t
    """
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    C = -np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), t)
    return C


def extract_gt_positions(gt_w2c):
    """GT 相机中心"""
    return extract_cam_centers_from_w2c(gt_w2c)


def extract_pred_positions(cam_quat, cam_trans):
    """
    从 cam_unnorm_rots 和 cam_trans 中提取预测相机位置。
    cam_quat: (1, 4, N)
    cam_trans: (1, 3, N)
    返回: (N, 3) 相机中心
    """
    # -> (N,4) and (N,3)
    quat = np.transpose(cam_quat[0], (1, 0))    # (N,4)
    t = np.transpose(cam_trans[0], (1, 0))      # (N,3)

    R = build_rotation_from_quat(quat)          # (N,3,3)

    # w2c: R, t  => 相机中心
    C = -np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), t)
    return C


def compute_ate_rmse(gt_pos, pred_pos):
    """
    计算 ATE RMSE，并对 pred_pos 做 Umeyama 对齐到 gt_pos。
    输入:
        gt_pos:   (N,3)
        pred_pos: (N,3)
    返回:
        rmse, pred_aligned
    """
    assert gt_pos.shape == pred_pos.shape

    scale, R, t = umeyama_alignment(pred_pos, gt_pos, with_scale=True)
    pred_aligned = (scale * (R @ pred_pos.T)).T + t

    errors = gt_pos - pred_aligned
    rmse = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))

    return rmse, pred_aligned


def plot_trajectory_2d(gt_pos, pred_pos, rmse, out_path="trajectory_2d.png"):
    """
    画 GT 和预测轨迹的 2D 投影 (X-Z 平面)。
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(gt_pos[:, 0], gt_pos[:, 2], label="Ground Truth", linewidth=2)
    ax.plot(pred_pos[:, 0], pred_pos[:, 2],
            label="Prediction (aligned)", linestyle="--")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(f"Camera Trajectory (ATE RMSE = {rmse:.4f} m)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ATE RMSE and visualize GT & predicted trajectory from params.npz"
    )
    parser.add_argument("params_path", type=str,
                        help="Path to params.npz file")
    parser.add_argument(
        "--out",
        type=str,
        default="trajectory_2d.png",
        help="Output image path for 2D trajectory",
    )
    args = parser.parse_args()

    data = np.load(args.params_path)

    if "gt_w2c_all_frames" not in data:
        raise KeyError("gt_w2c_all_frames not found in the npz file.")
    if "cam_trans" not in data or "cam_unnorm_rots" not in data:
        raise KeyError(
            "cam_trans or cam_unnorm_rots not found in the npz file.")

    gt_w2c = data["gt_w2c_all_frames"]       # (N,4,4)
    cam_trans = data["cam_trans"]            # (1,3,N)
    cam_quat = data["cam_unnorm_rots"]       # (1,4,N)


    gt_pos = extract_gt_positions(gt_w2c)                    # (Ng,3)
    pred_pos = extract_pred_positions(cam_quat, cam_trans)   # (Np,3)

    # 只在两边都存在的帧上做评估
    N = min(gt_pos.shape[0], pred_pos.shape[0])
    if gt_pos.shape[0] != pred_pos.shape[0]:
        print(f"[WARN] GT frames: {gt_pos.shape[0]}, Pred frames: {pred_pos.shape[0]} "
            f"=> only using first {N} frames for ATE.")

    gt_pos = gt_pos[:N]
    pred_pos = pred_pos[:N]
    rmse, pred_aligned = compute_ate_rmse(gt_pos, pred_pos)

    print(f"Number of frames: {gt_pos.shape[0]}")
    print(f"ATE RMSE (after similarity alignment): {rmse:.6f} m")

    plot_trajectory_2d(gt_pos, pred_aligned, rmse, out_path=args.out)
    print(f"2D trajectory image saved to: {args.out}")


if __name__ == "__main__":
    main()
