#!/usr/bin/env python3
"""Freeze the whole full-NIH plan (all presets, models, seeds) before any confirmatory test."""
import argparse
import json
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.config import resolve_config_path
from cxr_steganalysis.lab import load_lab_config
from cxr_steganalysis.protocol_lock import freeze_plan
from cxr_steganalysis.queue import make_plan

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, nargs="+", required=True,
                   help="Full presets; each declares lab.analysis_role, lab.models and lab.training_seeds")
    p.add_argument("--local", type=Path, help="Paths-only overlay applied to every preset")
    a = p.parse_args()
    entries, outputs = [], set()
    for path in a.plan:
        config = load_lab_config(path, a.local)
        # Identical run configs to what lab.py generates for this preset.
        plan, configs = make_plan(config, resolve_config_path(config, "output_dir"))
        runs = [(r["id"], configs[Path(r["config"])]) for r in plan["runs"]]
        entries.append(dict(role=config["lab"]["analysis_role"], config=config,
                            manifest_path=resolve_config_path(config, "pair_manifest"), runs=runs))
        outputs.add(resolve_config_path(config, "protocol_lock"))
    if len(outputs) != 1:
        raise ValueError(f"Presets must share one paths.protocol_lock, got {sorted(map(str, outputs))}")
    lock = outputs.pop()
    document = freeze_plan(entries, lock)
    print(json.dumps(dict(lock=str(lock), presets=[(e["role"], e["protocol_id"]) for e in document["presets"]],
                          configurations=len(document["configurations"])), indent=2))
