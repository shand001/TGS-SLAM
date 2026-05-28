# TGS-SLAM

<p align="center">
  <img src="https://img.shields.io/badge/Paper-RA--L%202026-1f77b4?style=for-the-badge" alt="Paper badge">
  <img src="https://img.shields.io/badge/Task-Semantic%20RGB--D%20SLAM-2ca02c?style=for-the-badge" alt="Task badge">
  <img src="https://img.shields.io/badge/Backend-3DGS%20%2B%20TriDS-ff7f0e?style=for-the-badge" alt="Backend badge">
</p>

<p align="center">
  Accepted for publication in <b>IEEE Robotics and Automation Letters</b> (April 2026).
</p>

<p align="center">
  <b>TGS-SLAM</b> is a semantic RGB-D SLAM system that anchors geometry and appearance with 3D Gaussians while encoding semantics in TriDS, a shared coarse-to-fine tri-plane field decoupled from Gaussian primitives.
</p>

<p align="center">
  <a href="https://doi.org/10.1109/LRA.2026.3692078"><b>Paper</b></a> ·
  <a href="#installation">Installation</a> ·
  <a href="#data-preparation">Data Preparation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#demo">Demo</a> ·
  <a href="#citation">Citation</a>
</p>

---

## Overview

Semantic SLAM aims to build 3D maps that are both geometrically accurate and semantically consistent. TGS-SLAM combines 3D Gaussian Splatting for efficient scene representation with TriDS for shared semantic decoding, so semantic features are queried at Gaussian centers and splatted together with color and depth in a single rendering pass.

### Highlights

- Geometry and appearance are anchored by 3D Gaussians, while semantics live in a shared TriDS field
- Semantic storage is decoupled from the number of Gaussian primitives, reducing memory overhead and label fragmentation
- Geometry-first, two-stage optimization delays semantic learning until the Gaussian map is stable
- Hybrid tracking combines constant-velocity prediction with ORB+PnP proposals selected by render-based scoring

## Key Results

| Setting | Result |
| --- | --- |
| Replica semantic mIoU | 97.02% |
| Replica ATE / Depth L1 | 0.33 cm / 0.47 cm |
| Replica PSNR / SSIM | 35.82 dB / 0.983 |
| Semantic parameter size on Replica | 10 MB |
| One-hot semantic baseline | 1059 MB |

## Demo

<p align="center">
  <b>Replica</b><br>
  <img src="video/Replica_TGS_SLAM_github.gif" width="90%" alt="TGS-SLAM on Replica">
</p>

<p align="center">
  <b>ScanNet</b><br>
  <img src="video/Scannet_TGS_SLAM_github.gif" width="90%" alt="TGS-SLAM on ScanNet">
</p>

<p align="center">
  <b>ScanNet++</b><br>
  <img src="video/Scannetpp_TGS_SLAM_github.gif" width="90%" alt="TGS-SLAM on ScanNet++">
</p>

## Installation

### Prerequisites

- Linux with an NVIDIA GPU
- Conda or Miniconda
- CUDA-compatible NVIDIA driver
- `git`, `gcc/g++`, and `ninja` for CUDA extensions

We recommend Python 3.9, CUDA 12.8, and PyTorch 2.7.1. Python 3.9, CUDA 11.8, and PyTorch 2.3.1 have also been tested.

```bash
conda create -n tgs-slam python=3.9 -y
conda activate tgs-slam
cd TGS-SLAM
python -m pip install --upgrade pip setuptools wheel ninja
conda install -y -c nvidia cuda-toolkit=12.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# Optional: set a target architecture if you want to build for a specific GPU
# RTX 30 series: export TORCH_CUDA_ARCH_LIST="8.6"
# RTX 40 series: export TORCH_CUDA_ARCH_LIST="8.9"
# RTX 50 series: export TORCH_CUDA_ARCH_LIST="12.0"

python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"

pip install -r requirements.txt
```

## Data Preparation

The default configs expect datasets under `../data` relative to the repository root. You can either follow the layouts below or edit `config["data"]["basedir"]` in the corresponding `configs/*/slam.py`.

### Replica

Default path:

```text
../data/Replica/<scene_name>/
  frames/frame*.jpg
  depths/depth*.png
  semantic_ids/semantic_id*.png
  semantic_colors/semantic_color*.png
  traj.txt
```

Provided scene configs include:

```text
room0 room1 room2 office0 office1 office2 office3 office4
```

Run one scene by editing `scene_name`, `primary_device`, and `num_frames` in `configs/replica/slam.py`, then:

```bash
python scripts/slam.py configs/replica/slam.py
```

### ScanNet

Default path:

```text
../data/scans/<scene_name>/
  color/*.jpg
  depth/*.png
  pose/*.txt
  label_40/*.png
```

The code expects `label_40` semantic labels in NYU40 id space. If your ScanNet folder is in raw `.sens` format, export RGB, depth, pose, and intrinsics with:

```bash
python preprocess/scannet/run.py \
  --input_folder /path/to/scannet_raw/scene0000_00 \
  --output_folder ../data/scans/scene0000 \
  --export_depth_images \
  --export_color_images \
  --export_poses \
  --export_intrinsics
```

Then convert the ScanNet `label-filt` annotations to NYU40 ids and place them in:

```text
../data/scans/scene0000/label_40/
```

Make sure the frame indices in `color`, `depth`, `pose`, and `label_40` are aligned.

Provided scene configs include:

```text
scene0000 scene0059 scene0106 scene0169 scene0181 scene0207
```

Run one scene by editing `scene_name`, `primary_device`, and `data.basedir` in `configs/ScanNet/slam.py`, then:

```bash
python scripts/slam.py configs/ScanNet/slam.py
```

### ScanNet++

Default path:

```text
../data/scannetpp/<scene_name>/dslr/
  train_test_lists.json
  nerfstudio/transforms_undistorted.json
  undistorted_images/*.JPG
  undistorted_depths/*.png
  undistorted_semantic_id/*.png
  undistorted_semantic_color/*.png
```

Provided scene configs include:

```text
8b5caf3398 b20a261fdf
```

Run one scene by editing `scene_name`, `primary_device`, `use_train_split`, and `data.basedir` in `configs/scannetpp/slam.py`, then:

```bash
python scripts/slam.py configs/scannetpp/slam.py
```

## Quick Start

If you just want to launch a scene, start from one of the config files above and run:

```bash
python scripts/slam.py configs/replica/slam.py
```

Other utility scripts are available under `scripts/` for rendering, evaluation, and point cloud generation.

After a successful run, results are written to an experiment directory such as:

```text
experiments/Replica/room0_0_20260528_1149/
```

Typical outputs in that folder are:

- `config.py`: the resolved run configuration
- `params.npz`, `planes.pth`, `decoder.pth`: the main saved checkpoints
- `eval/`: the evaluation summary, rendered RGB/depth/segmentation outputs, and metric files
- `tsdf_pointcloud/`: the exported point cloud and mesh results

The timestamped suffix in the experiment folder name changes from run to run.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tgsslam,
  title={TGS-SLAM: Tri-plane Gaussian Splatting for Semantic SLAM},
  author={Sun, Guoxi and Shen, Handong and Liu, Xiaohao and Li, Xinchao and Liang, Lingyu and Liu, Beibei and Huang, Shuangping},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  pages={1--8},
  doi={10.1109/LRA.2026.3692078}
}
```
