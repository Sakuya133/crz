"""Research run bindings and read-only environment/source provenance."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml
from cxr_steganalysis.config import serializable_config
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.reproducibility import runtime_environment, stable_seed


def effective_seeds(config):
    return {name: int(config.get("seeds", {}).get(name, config["seed"])) for name in ("split", "sampling", "embedding", "crop", "training")}


def manifest_identity(frame):
    """Path-independent identity; image checksums remain part of the binding."""
    cols = sorted(c for c in frame.columns if not c.endswith("_path"))
    text = frame.sort_values("pair_id")[cols].astype(str).to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


def select_mixed_training(frame, per_view, seed):
    """Sample only within locked training assignments; keep validation/test untouched."""
    pieces = [frame[frame.split != "train"]]
    for view in ("AP", "PA"):
        pool = frame[(frame.split == "train") & (frame.view_position == view)].copy()
        if len(pool) < per_view:
            raise ValueError(f"Mixed quota shortage {view}: {len(pool)} < {per_view}")
        pool["_order"] = [stable_seed(seed, p, "mixed-training") for p in pool.pair_id]
        pieces.append(pool.sort_values(["_order", "pair_id"]).head(per_view).drop(columns="_order"))
    return pd.concat(pieces, ignore_index=True)


def select_size_matched(frame, seed):
    """Equalize AP/PA pairs per train/validation split at min(AP, PA).

    Pairs are drawn inside the locked splits (assignments untouched); test is kept
    whole so every model is evaluated on the same target cohort. Equal pairs do not
    imply equal patients.
    """
    pieces = [frame[frame.split == "test"]]
    for split in ("train", "validation"):
        pools = {view: frame[(frame.split == split) & (frame.view_position == view)].copy() for view in ("AP", "PA")}
        quota = min(len(pool) for pool in pools.values())
        if quota == 0:
            raise ValueError(f"Size control impossible: empty {split} view")
        for view, pool in pools.items():
            pool["_order"] = [stable_seed(seed, p, split, "size-matched") for p in pool.pair_id]
            pieces.append(pool.sort_values(["_order", "pair_id"]).head(quota).drop(columns="_order"))
    return pd.concat(pieces, ignore_index=True)


def enrollment_counts(frame):
    group = ["split", "view_position"] + (["test_cohort"] if "test_cohort" in frame else [])
    return frame.groupby(group).agg(pairs=("pair_id", "size"), patients=("patient_id", "nunique")).reset_index().to_dict("records")


def save_run_provenance(output, config, binding):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config["_project_root"])
    effective = serializable_config(config)
    effective["project_root"] = str(root)
    content = yaml.safe_dump(effective, sort_keys=True)
    effective_path = output / "effective_config.yaml"
    resuming = effective_path.exists()
    if resuming:
        # Epoch budget may increase on resume; immutable fields are checked by Trainer.
        previous = yaml.safe_load(effective_path.read_text())
        if previous != effective:
            (output / "resume_effective_config.yaml").write_text(content)
    else:
        effective_path.write_text(content)
    sources = sorted(p for folder in ("src", "scripts", "configs") for p in (root / folder).rglob("*") if p.suffix in {".py", ".yaml"})
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sources}
    git = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    metadata = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
                "binding": binding, "seeds": effective_seeds(config), "environment": runtime_environment(),
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "git_status": git.stdout if git.returncode == 0 else "not_a_git_repository; source snapshot used",
                "source_sha256": hashes, "source_identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    destination = output
    if resuming:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = output / "resume_provenance" / stamp
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "effective_config.yaml").write_text(content)
        metadata["scope"] = "resume_attempt; original training provenance retained unchanged"
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with tarfile.open(destination / "source_snapshot.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(root)))
