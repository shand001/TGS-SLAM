import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

METRIC_PATTERNS = {
    # Allow extra chars around, no anchors; tolerate spaces/units
    "tracking_iter_ms": re.compile(r"Average\s+Tracking/Iteration\s+Time:\s*([\d\.eE+-]+)\s*ms"),
    "tracking_frame_s": re.compile(r"Average\s+Tracking/Frame\s+Time:\s*([\d\.eE+-]+)\s*s"),
    "mapping_iter_ms": re.compile(r"Average\s+Mapping/Iteration\s+Time:\s*([\d\.eE+-]+)\s*ms"),
    "mapping_frame_s": re.compile(r"Average\s+Mapping/Frame\s+Time:\s*([\d\.eE+-]+)\s*s"),
    "final_ate_cm": re.compile(r"Final\s+Average\s+ATE\s+RMSE:\s*([\d\.eE+-]+)\s*cm"),
    "avg_psnr": re.compile(r"Average\s+PSNR:\s*([\d\.eE+-]+)\b"),
    "avg_depth_rmse_cm": re.compile(r"Average\s+Depth\s+RMSE:\s*([\d\.eE+-]+)(?:\s*cm)?\b"),
    "avg_depth_l1_cm": re.compile(r"Average\s+Depth\s+L1:\s*([\d\.eE+-]+)(?:\s*cm)?\b"),
    "avg_msssim": re.compile(r"Average\s+MS-SSIM:\s*([\d\.eE+-]+)\b"),
    "avg_lpips": re.compile(r"Average\s+LPIPS:\s*([\d\.eE+-]+)\b"),
    "avg_miou": re.compile(r"Average\s+mIoU:\s*([\d\.eE+-]+)\b"),
}


def parse_args():
    """CLI for batch-running TGS-SLAM on multiple scenes.

    Required:
      --config  Path to a single base config .py
      --scenes  Scene / sequence names (space separated)

    Everything else has sensible defaults.
    """
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir / "configs" / "replica" / "slam.py"
    default_slam_script = script_dir / "scripts" / "slam.py"
    default_csv = script_dir / "experiments" / "exp_report.csv"

    p = argparse.ArgumentParser(
        description="Batch runner for TGS-SLAM/scripts/slam.py using config overrides"
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        default=str(default_config),
        help="Base config .py file path",
    )
    p.add_argument(
        "-s",
        "--scenes",
        required=True,
        nargs="+",
        help="Scene / sequence names to run, e.g., room0 office0",
    )

    p.add_argument(
        "-d",
        "--primary_device",
        default="cuda:0",
        help="Primary device, e.g., cuda:0 (default: cuda:0)",
    )
    p.add_argument(
        "-t",
        "--run_times",
        type=int,
        default=1,
        help="Number of runs per scene; seed increments each run (default: 1)",
    )
    p.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base seed for the first run of each scene (default: 0)",
    )

    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current python)",
    )
    p.add_argument(
        "--slam_script",
        default=str(default_slam_script),
        help="Path to scripts/slam.py (default: TGS-SLAM/scripts/slam.py)",
    )
    p.add_argument(
        "-p",
        "--path_csv",
        default=str(default_csv),
        help="Global CSV path to append results (default: experiments/exp_report.csv)",
    )
    return p.parse_args()


def make_temp_config(
    base_config_path: str,
    out_dir: Path,
    scene: str,
    seed: int,
    primary_device: str,
    run_name: str,
) -> Path:
    """Create a full config file by copying the base and appending overrides.

    This avoids inheritance-style wrappers so that the main SLAM code
    saves a self-contained config.py in the experiments directory.
    """
    base_path = Path(base_config_path).resolve()
    base_text = base_path.read_text(encoding="utf-8")

    override_block = f"""

# ==== Auto-generated overrides by run_slam_batch.py (do not edit) ====
primary_device = "{primary_device}"
seed = {seed}
scene_name = "{scene}"

try:
    # Keep config dict in sync if present
    if "config" in globals() and isinstance(config, dict):
        config["primary_device"] = primary_device
        config["seed"] = seed
        data_cfg = config.get("data")
        if isinstance(data_cfg, dict):
            if "sequence" in data_cfg:
                data_cfg["sequence"] = scene_name
            elif "scene_name" in data_cfg:
                data_cfg["scene_name"] = scene_name
        viz_cfg = config.get("viz")
        if isinstance(viz_cfg, dict) and "scene_name" in viz_cfg:
            viz_cfg["scene_name"] = scene_name

        config["run_name"] = "{run_name}"
        if "wandb" in config and isinstance(config["wandb"], dict):
            config["wandb"]["name"] = config["run_name"]

        # Replica-style Triplane configs (best-effort)
        tri_cfg = config.get("TriplaneConfigs")
        if isinstance(tri_cfg, str) and "replica" in tri_cfg.lower():
            config["TriplaneConfigs"] = f"./configs/replica/{{scene_name}}.yaml"
except Exception:
    # Don't let override issues crash the run
    pass
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"config_{scene}_seed{seed}.py"
    out_path.write_text(base_text.rstrip() + override_block, encoding="utf-8")
    return out_path


def run_one(slam_script: str, config_path: Path, env=None, python_exe: str = None):
    py = python_exe or sys.executable
    proc = subprocess.Popen(
        [py, slam_script, str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        print(line, end="")  # also stream to console
    proc.wait()
    return proc.returncode, "".join(lines)


def parse_metrics(log_text: str) -> dict:
    # Strip ANSI to reduce tqdm/control-code interference
    clean = ANSI_RE.sub("", log_text)
    results = {}
    for key, pat in METRIC_PATTERNS.items():
        m = None
        # find last occurrence to capture final values if printed multiple times
        for match in pat.finditer(clean):
            m = match
        if m:
            results[key] = float(m.group(1))
        else:
            results[key] = None
    return results


def main():
    args = parse_args()

    base_config = Path(args.config).resolve()
    slam_script = Path(args.slam_script).resolve()

    # Validate target slam script looks like the expected one (takes a single experiment file)
    if not slam_script.exists():
        raise FileNotFoundError(f"slam_script not found: {slam_script}")
    try:
        head = slam_script.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        head = ""
    # Heuristic: the correct script defines a single positional 'experiment' arg and uses SourceFileLoader
    if "parser.add_argument(\"experiment\"" not in head or "SourceFileLoader" not in head:
        raise RuntimeError(
            f"The provided --slam_script does not look like TGS-SLAM/scripts/slam.py.\n"
            f"Got: {slam_script}\n"
            f"Tip: point --slam_script to TGS-SLAM/scripts/slam.py (absolute path recommended)."
        )

    # Temp configs live outside the repo to avoid cluttering experiments
    tmp_root = Path(tempfile.gettempdir()) / "tgs_slam_batch_configs"
    tmp_cfg_dir = tmp_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Global CSV (append mode)
    csv_path = Path(args.path_csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8")
    try:
        if not csv_exists:
            csv_file.write(
                "run_name,scene,seed,primary_device,tracking_iter_ms,tracking_frame_s,mapping_iter_ms,mapping_frame_s,"
                "final_ate_cm,avg_psnr,avg_depth_rmse_cm,avg_depth_l1_cm,avg_msssim,avg_lpips,avg_miou\n"
            )

        for scene in args.scenes:
            for i in range(args.run_times):
                seed = args.base_seed + i
                run_ts = datetime.now().strftime("%Y%m%d_%H%M")
                run_name = f"{scene}_{seed}_{run_ts}"

                cfg_path = make_temp_config(
                    str(base_config),
                    tmp_cfg_dir,
                    scene,
                    seed,
                    args.primary_device,
                    run_name,
                )

                print(
                    f"\n=== Running {run_name} (scene={scene}, seed={seed}) ===")
                print(f"Invoking: {args.python} {slam_script} {cfg_path}")
                code, out = run_one(
                    str(slam_script), cfg_path, python_exe=args.python)

                # Parse metrics from stdout
                metrics = parse_metrics(out)

                csv_file.write(
                    ",".join(
                        [
                            run_name,
                            scene,
                            str(seed),
                            args.primary_device,
                            str(metrics.get("tracking_iter_ms")),
                            str(metrics.get("tracking_frame_s")),
                            str(metrics.get("mapping_iter_ms")),
                            str(metrics.get("mapping_frame_s")),
                            str(metrics.get("final_ate_cm")),
                            str(metrics.get("avg_psnr")),
                            str(metrics.get("avg_depth_rmse_cm")),
                            str(metrics.get("avg_depth_l1_cm")),
                            str(metrics.get("avg_msssim")),
                            str(metrics.get("avg_lpips")),
                            str(metrics.get("avg_miou")),
                        ]
                    )
                    + "\n"
                )
                csv_file.flush()

                if code != 0:
                    print(
                        f"Run failed for scene={scene}, seed={seed} (exit={code})."
                    )
    finally:
        csv_file.close()

    print(f"\nAll done. Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
