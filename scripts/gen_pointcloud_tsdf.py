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

from datasets.gradslam_datasets import (  # noqa: E402
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
    NeRFCaptureDataset,
)
from utils.common_utils import seed_everything  # noqa: E402
from utils.recon_helpers import to_gsplat_camera  # noqa: E402
from utils.slam_helpers import (  # noqa: E402
    transform_to_frame,
    params_to_gsplat_inputs,
)
from utils.slam_external import build_rotation  # noqa: E402


def get_dataset(config_dict, basedir, sequence, **kwargs):
    name = config_dict["dataset_name"].lower()
    if name in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    if name in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    if name in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    if name in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def decode_segmap(image, nc=25):
    label_colors = np.array([
        (0, 0, 0),
        (174, 199, 232), (152, 223, 138), (31, 119,
                                           180), (255, 187, 120), (188, 189, 34),
        (140, 86, 75), (255, 152, 150), (214, 39,
                                         40), (197, 176, 213), (148, 103, 189),
        (196, 156, 148), (23, 190, 207), (178,
                                          76, 76), (247, 182, 210), (66, 188, 102),
        (219, 219, 141), (140, 57, 197), (202,
                                          185, 52), (51, 176, 203), (200, 54, 131),
        (92, 193, 61), (78, 71, 183), (172, 114,
                                       82), (255, 127, 14), (91, 163, 138),
        (153, 98, 156), (140, 153, 101), (158, 218,
                                          229), (100, 125, 154), (178, 127, 135),
        (120, 185, 128), (146, 111, 194), (44,
                                           160, 44), (112, 128, 144), (96, 207, 209),
        (227, 119, 194), (213, 92, 176), (94,
                                          106, 211), (82, 84, 163), (100, 85, 144),
        (100, 218, 200), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
        (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
        (192, 128, 0), (64, 0, 128)
    ], dtype=np.uint8)

    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    max_index = min(nc, label_colors.shape[0])
    for label in range(0, max_index):
        mask = image == label
        r[mask] = label_colors[label, 0]
        g[mask] = label_colors[label, 1]
        b[mask] = label_colors[label, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb.astype(np.float32) / 255.0


def load_final_params_from_npz(npz_path, device):
    data = np.load(npz_path, allow_pickle=True)

    required_keys = [
        "means3D",
        "unnorm_rotations",
        "log_scales",
        "logit_opacities",
        "rgb_colors",
        "cam_unnorm_rots",
        "cam_trans",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}")

    params = {}
    for key in required_keys:
        arr = data[key]
        tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        params[key] = tensor

    if "semantic_colors" in data:
        arr = data["semantic_colors"]
        params["semantic_colors"] = torch.from_numpy(arr).to(
            device=device, dtype=torch.float32
        )
    if "semantic_id" in data:
        arr = data["semantic_id"]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        params["semantic_id"] = torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    return params


def generate_pointcloud(
    config,
    dataset,
    final_params,
    num_frames,
    out_dir,
    sil_thres=0.5,
    eval_every=1,
    scale=1.0,
    depth_trunc=30.0,
    depth_clip_max=10.0,
    device=torch.device("cuda"),
):
    os.makedirs(out_dir, exist_ok=True)
    pcl_dir = os.path.join(out_dir, "pointcloud")
    os.makedirs(pcl_dir, exist_ok=True)

    print("Start generating TSDF point clouds (gsplat)...")

    dataset_config = config["data"]
    # cam_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    # cam_params = cam_cfg["camera_params"]
    # width = cam_params["image_width"] - 2 * cam_params["crop_edge"]
    # height = cam_params["image_height"] - 2 * cam_params["crop_edge"]
    # fx = cam_params["fx"]
    # fy = cam_params["fy"]
    # cx = cam_params["cx"]
    # cy = cam_params["cy"]

    volume_rgb = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    volume_sem = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    compensate_vector = (
        -0.0 * scale / 512.0,
        2.5 * scale / 512.0,
        -2.5 * scale / 512.0,
    )

    load_semantics = dataset_config.get("load_semantics", False)
    use_one_hot_semantics = config.get("use_one_hot_semantics", False)
    num_semantic_classes = dataset_config.get("num_semantic_classes", 0)

    for time_idx in tqdm(range(num_frames), desc="Integrating frames"):
        if load_semantics:
            color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[
                time_idx
            ]
        else:
            color, depth, intrinsics, pose = dataset[time_idx]
        intrinsics = intrinsics[:3, :3]

        color_chw = color.permute(2, 0, 1) / 255.0
        depth_chw = depth.permute(2, 0, 1)

        if time_idx != 0 and (time_idx + 1) % eval_every != 0:
            continue

        # Estimated camera pose for this frame (world -> camera)
        with torch.no_grad():
            curr_cam_rot = F.normalize(
                final_params["cam_unnorm_rots"][..., time_idx].detach()
            )
            curr_cam_tran = final_params["cam_trans"][..., time_idx].detach()
            curr_w2c = torch.eye(4, device=device).float()
            curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
            curr_w2c[:3, 3] = curr_cam_tran

        # Build per-frame gsplat camera using the estimated pose
        img_height, img_width = color_chw.shape[1], color_chw.shape[2]
        _, _, k_gs, view = to_gsplat_camera(
            img_width,
            img_height,
            intrinsics.cpu().numpy(),
            curr_w2c.detach().cpu().numpy(),
            device=device,
        )

        # Render using static world Gaussians and per-frame camera
        mean, quat, scale_tensor, opac, color_feat = params_to_gsplat_inputs(
            final_params,
            final_params["means3D"],
            use_semantics=load_semantics,
            use_one_hot_semantics=use_one_hot_semantics,
        )

        rgbds, alpha, _ = rasterization(
            mean,
            quat,
            scale_tensor,
            opac,
            color_feat,
            view,
            k_gs,
            img_width,
            img_height,
            render_mode="RGB+D",
        )

        rgb_tensor = (
            rgbds[..., :3].squeeze(0).squeeze(0).permute(2, 0, 1).contiguous()
        )
        depth_tensor = rgbds[..., -1].squeeze(0).squeeze(0).contiguous()

        depth_np = depth_tensor.detach().cpu().numpy()
        depth_np = np.clip(depth_np, 0.0, depth_clip_max).astype(np.float32)
        depth_img = o3d.geometry.Image(depth_np)

        rgb_np = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb_img = o3d.geometry.Image(np.ascontiguousarray(rgb_np))

        sem_img = None
        if load_semantics:
            if not use_one_hot_semantics:
                seg_tensor = (
                    rgbds[..., 3:6].squeeze(0).squeeze(0).permute(2, 0, 1).contiguous()
                )
                seg_np = seg_tensor.detach().cpu().permute(1, 2, 0).numpy()
                seg_np = (np.clip(seg_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                seg_logits = rgbds[..., 3:-1].squeeze(0).squeeze(0).contiguous()
                label_map = torch.argmax(seg_logits, dim=-1).cpu().numpy().astype(
                    np.uint8
                )
                class_count = (
                    num_semantic_classes
                    if num_semantic_classes > 0
                    else seg_logits.shape[-1]
                )
                seg_color = decode_segmap(label_map, nc=class_count)
                seg_np = (seg_color * 255.0).astype(np.uint8)
            sem_img = o3d.geometry.Image(np.ascontiguousarray(seg_np))

        # Use the same estimated pose for TSDF integration
        w2c_np = curr_w2c.detach().cpu().numpy()

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            img_width,
            img_height,
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img,
            depth_img,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume_rgb.integrate(rgbd, intrinsic_o3d, w2c_np)

        if sem_img is not None:
            rgbd_sem = o3d.geometry.RGBDImage.create_from_color_and_depth(
                sem_img,
                depth_img,
                depth_scale=1.0,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
            )
            volume_sem.integrate(rgbd_sem, intrinsic_o3d, w2c_np)

    print("TSDF integration finished, extracting meshes ...")
    mesh_rgb = volume_rgb.extract_triangle_mesh()
    mesh_sem = volume_sem.extract_triangle_mesh()

    mesh_rgb = mesh_rgb.translate(compensate_vector)
    mesh_sem = mesh_sem.translate(compensate_vector)

    print("Converting meshes to point clouds ...")
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = mesh_rgb.vertices
    if mesh_rgb.has_vertex_colors():
        pcd_rgb.colors = mesh_rgb.vertex_colors

    pcd_sem = o3d.geometry.PointCloud()
    pcd_sem.points = mesh_sem.vertices
    if mesh_sem.has_vertex_colors():
        pcd_sem.colors = mesh_sem.vertex_colors

    rgb_pcd_path = os.path.join(pcl_dir, "pred_pointcloud_rgb.ply")
    sem_pcd_path = os.path.join(pcl_dir, "pred_pointcloud_se.ply")
    o3d.io.write_point_cloud(rgb_pcd_path, pcd_rgb)
    o3d.io.write_point_cloud(sem_pcd_path, pcd_sem)
    print(f"Saved RGB point cloud to: {rgb_pcd_path}")
    print(f"Saved semantic point cloud to: {sem_pcd_path}")

    mesh_dir = os.path.join(out_dir, "mesh_for_debug")
    os.makedirs(mesh_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, "pred_mesh_rgb.ply"), mesh_rgb)
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, "pred_mesh_se.ply"), mesh_sem)
    print("Done.")


def build_dataset_from_config(config, device):
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True

    if "load_semantics" not in dataset_config:
        load_semantics = False
        num_semantic_classes = 0
    else:
        load_semantics = dataset_config["load_semantics"]
        num_semantic_classes = dataset_config["num_semantic_classes"]

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

    return dataset, num_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TSDF-fused RGB and semantic point clouds using gsplat."
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
        help="Output directory (default: <workdir>/<run_name>/tsdf_pointcloud).",
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
        help="Integrate every N-th frame (default: 1).",
    )
    parser.add_argument(
        "--sil_thres",
        type=float,
        default=0.5,
        help="Silhouette threshold (currently unused, for compatibility).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scene scale for TSDF voxels (default: 1.0).",
    )
    parser.add_argument(
        "--depth_trunc",
        type=float,
        default=30.0,
        help="Depth truncation for TSDF integration (Open3D).",
    )
    parser.add_argument(
        "--depth_clip_max",
        type=float,
        default=10.0,
        help="Clip rendered depth before TSDF (default: 10.0m).",
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

    output_root = os.path.join(config["workdir"], config["run_name"])
    params_path = args.params or os.path.join(output_root, "params.npz")
    out_dir = args.out_dir or os.path.join(output_root, "tsdf_pointcloud")

    dataset, default_num_frames = build_dataset_from_config(config, device)
    num_frames = args.num_frames or default_num_frames

    final_params = load_final_params_from_npz(params_path, device)

    generate_pointcloud(
        config=config,
        dataset=dataset,
        final_params=final_params,
        num_frames=num_frames,
        out_dir=out_dir,
        sil_thres=args.sil_thres,
        eval_every=args.eval_every,
        scale=args.scale,
        depth_trunc=args.depth_trunc,
        depth_clip_max=args.depth_clip_max,
        device=device,
    )
