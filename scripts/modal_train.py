#!/usr/bin/env python3
"""Serverless GPU training via Modal.

Train on an A100 without managing servers. Pay per second.
$30/month free credits ≈ 25 hours of A100 time.

Setup (one time):
    pip install modal
    modal setup              # links your Modal account

Usage:
    # Default experiment
    python scripts/modal_train.py

    # Specific experiment
    python scripts/modal_train.py --experiment 02_locomotion

    # Override config
    python scripts/modal_train.py --timesteps 1000000 --creature coral
"""

import modal
import sys

app = modal.App("yeppoh")

# Docker image with all deps pre-installed
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "genesis-world",
        "torchrl>=0.4",
        "tensordict>=0.4",
        "pettingzoo>=1.24",
        "trimesh>=4.0",
        "omegaconf>=2.3",
        "hydra-core>=1.3",
        "tensorboard>=2.15",
        "scipy>=1.12",
        "imageio>=2.33",
    )
    .run_commands("git clone https://github.com/nicholascpark/yeppoh.git /workspace/yeppoh")
)

# Persistent volume for saving results across runs
results_volume = modal.Volume.from_name("yeppoh-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,  # 4 hour max
    volumes={"/results": results_volume},
)
def train(experiment: str = "01_basic_blob", overrides: dict | None = None):
    """Run training on a cloud A100."""
    import subprocess
    import shutil
    from pathlib import Path

    workdir = Path("/workspace/yeppoh")

    # Pull latest code
    subprocess.run(["git", "pull"], cwd=workdir, check=True)

    # Run experiment
    exp_script = workdir / "experiments" / f"{experiment}.py"
    if exp_script.exists():
        print(f"Running experiment: {experiment}")
        subprocess.run(
            [sys.executable, str(exp_script)],
            cwd=workdir,
            check=True,
        )
    else:
        print(f"Experiment {experiment} not found, running default training")
        cmd = [sys.executable, "scripts/train.py"]
        if overrides:
            for k, v in overrides.items():
                cmd.append(f"{k}={v}")
        subprocess.run(cmd, cwd=workdir, check=True)

    # Copy results to persistent volume
    runs_dir = workdir / "runs"
    if runs_dir.exists():
        dest = Path("/results") / experiment
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(runs_dir, dest, dirs_exist_ok=True)
        print(f"Results saved to volume: /results/{experiment}")

    results_volume.commit()


@app.function(
    image=image,
    volumes={"/results": results_volume},
)
def download_results(experiment: str = "01_basic_blob"):
    """List available results."""
    from pathlib import Path

    results_dir = Path("/results") / experiment
    if results_dir.exists():
        files = list(results_dir.rglob("*"))
        for f in sorted(files):
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f.relative_to('/results')} ({size_mb:.1f} MB)")
    else:
        print(f"No results found for {experiment}")
        print("Available:")
        for d in Path("/results").iterdir():
            print(f"  {d.name}")


@app.local_entrypoint()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="01_basic_blob")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--creature", default=None)
    parser.add_argument("--list-results", action="store_true")
    args = parser.parse_args()

    if args.list_results:
        download_results.remote(args.experiment)
        return

    overrides = {}
    if args.timesteps:
        overrides["training.timesteps"] = args.timesteps
    if args.creature:
        overrides["creature.plan"] = args.creature

    train.remote(
        experiment=args.experiment,
        overrides=overrides if overrides else None,
    )
