#!/usr/bin/env python3
"""Read-only environment/resource checks. No driver changes or training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from _bootstrap import bootstrap
bootstrap()
import psutil
import torch
from cxr_steganalysis.lab import load_lab_config, atomic_json
from cxr_steganalysis.data.portable import check_data, read_pairs
from cxr_steganalysis.config import resolve_config_path


def resources(frame):
    # float32 cache; solver materializes float64 slices/scaled matrices/liblinear.
    estimates = {}
    for view in ("AP", "PA"):
        ntrain = 2 * int(((frame.split == "train") & (frame.view_position == view)).sum())
        nval = 2 * int(((frame.split == "validation") & (frame.view_position == view)).sum())
        peak = (5*ntrain + 2*nval) * 34671 * 8
        estimates[view] = dict(train_rows=ntrain, validation_rows=nval, conservative_solver_gib=peak/2**30,
                               required_available_ram_gib_at_70pct_guard=peak/.7/2**30)
    return dict(srm_feature_dimension=34671, float32_cache_gib=len(frame)*2*34671*4/2**30,
                solver=estimates, note="Estimates, not measurements; copies can exceed the memmap size. No subsampling fallback.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--require-srm", action="store_true")
    p.add_argument("--config", type=Path)
    p.add_argument("--local", type=Path)
    p.add_argument("--data", action="store_true")
    p.add_argument("--verify-hashes", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = []
    warnings = []
    packages = {}
    for name in ("torch", "numpy", "pandas", "Pillow", "PyYAML", "matplotlib", "scikit-learn", "scipy", "psutil", "sealwatch", "pytest"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
            if name not in {"sealwatch", "pytest"} or (name == "sealwatch" and a.require_srm): errors.append(f"Missing {name}")
    if os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        warnings.append("PYTHONPATH/PYTHONHOME is set; verify it refers to the intended lab environment")
    # Existing lab/Conda/system-managed runtimes are valid: readiness concerns
    # actual dependencies and CUDA, not whether a project-specific venv exists.
    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        try:
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(i)
                devices.append(dict(index=i, name=device.name, vram_gib=device.total_memory/2**30))
            torch.ones(2, device="cuda").sum().item()
        except RuntimeError as error: errors.append(str(error))
    if a.require_cuda and not cuda: errors.append("CUDA unavailable; NIH CNN queues will refuse CPU fallback")
    check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    if check.returncode: errors.append(check.stdout + check.stderr)
    result = dict(python=sys.version, executable=sys.executable, isolated_venv=sys.prefix != sys.base_prefix,
                  torch_runtime=torch.__version__, torch_cuda_build=torch.version.cuda, cuda_available=cuda, gpus=devices,
                  ram_total_gib=psutil.virtual_memory().total/2**30, ram_available_gib=psutil.virtual_memory().available/2**30,
                  project_disk_free_gib=shutil.disk_usage(root).free/2**30, packages=packages, pip_check=check.stdout.strip())
    if shutil.which("nvidia-smi"):
        driver = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"], text=True, capture_output=True, timeout=20)
        result["nvidia_driver_query"] = dict(returncode=driver.returncode, output=driver.stdout.strip(), note="Driver visibility does not imply this Torch build supports CUDA")
    if a.config:
        config = load_lab_config(a.config, a.local)
        result["dataset_profile"] = config.get("dataset_profile")
        result["resolved_paths"] = {k: str(resolve_config_path(config, k)) for k in config["paths"]}
        if a.data:
            try:
                result["data"] = check_data(config, a.verify_hashes)
                result["resource_estimates"] = resources(read_pairs(resolve_config_path(config, "pair_manifest")))
            except (ValueError, OSError, AssertionError) as error: errors.append(str(error))
    elif a.data: errors.append("--data requires --config")
    result.update(status="blocked" if errors else "ready", errors=errors, warnings=warnings)
    if a.output: atomic_json(a.output, result)
    print(json.dumps(result, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
