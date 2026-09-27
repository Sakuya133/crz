#!/usr/bin/env python3
"""Inventory or materialize full NIH; never download, extract archives, or train."""
import argparse
import json
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.lab import load_lab_config, atomic_json
from cxr_steganalysis.config import resolve_config_path
from cxr_steganalysis.data.full_dataset import full_preflight
from cxr_steganalysis.data.portable import read_pairs
from cxr_steganalysis.provenance import sha256_file
from doctor import resources
from prepare_full import prepare


def ready(preflight):
    summary = json.loads((preflight / "summary.json").read_text())
    if summary["missing_images"] or summary["exclusions"].get("unreadable_requires_retrieval",0):
        raise ValueError(f"Full dataset incomplete: see {preflight}/missing_images.csv and inventory.csv; no silent available-only subset")
    for path, expected in summary["bindings"].items():
        if sha256_file(path) != expected:
            raise ValueError("Metadata/official lists/pilot binding changed after preflight; make a new inventory")
    import pandas as pd
    frame = pd.read_csv(preflight / "splits.csv", dtype={"patient_id":str})
    for r in frame.itertuples():
        stat = Path(r.cover_path).stat()
        if stat.st_size != r.file_bytes or stat.st_mtime_ns != r.mtime_ns:
            raise ValueError(f"Raw image changed after inventory: {r.image_id}")
    for view in ("AP","PA"):
        for split in ("train","validation","test"):
            if not ((frame.view_position == view) & (frame.split == split)).any():
                raise ValueError(f"Empty full eligible {split}/{view}")
        if not ((frame.view_position == view) & (frame.test_cohort == "confirmatory_unseen_patient")).any():
            raise ValueError(f"No unseen confirmatory patients for {view}")
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight","prepare"])
    p.add_argument("--config", type=Path, default=Path("configs/full_all_eligible.yaml"))
    p.add_argument("--local", type=Path)
    p.add_argument("--preflight", type=Path, help="Default: configured split_manifest parent")
    a = p.parse_args()
    config = load_lab_config(a.config,a.local)
    if config.get("dataset_profile") != "full_all_eligible": raise ValueError("Full command requires full_all_eligible preset")
    preflight = a.preflight or resolve_config_path(config,"split_manifest").parent
    if a.action == "preflight":
        summary = full_preflight(config,preflight)
        print(json.dumps(summary,indent=2))
        frame = ready(preflight)
        # Pair count is one per eligible cover; no files are embedded here.
        atomic_json(preflight / "resource_estimates.json",resources(frame))
    else:
        ready(preflight)
        output = resolve_config_path(config,"pair_manifest").parent
        if resolve_config_path(config,"stego_dir") != output / "stego":
            raise ValueError("Full prepare stores stego in <pair_manifest parent>/stego; set both paths consistently")
        prepare(config,preflight,output)


if __name__ == "__main__":
    main()
