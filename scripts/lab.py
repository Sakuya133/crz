#!/usr/bin/env python3
"""Explicit lab queue (one GPU job), reproducible dry-run, status and resume."""
import argparse
import json
from pathlib import Path
import sys
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.lab import load_lab_config
from cxr_steganalysis.queue import make_plan, execute, status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/pilot_existing.yaml"))
    p.add_argument("--local", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=["highpass", "srnet", "srm_svm"])
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--sources", nargs="+", choices=["AP", "PA"])
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--budget-hours", type=float, default=8, help="Per invocation; resume renews the budget, never the scientific epoch budget")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.status: status(a.output); return 0
    config = load_lab_config(a.config, a.local)
    plan, configs = make_plan(config, a.output, a.models, a.seeds, a.sources, a.feature_cache)
    if a.dry_run: print(json.dumps(plan, indent=2)); return 0
    return execute(plan, configs, a.output, resume=a.resume, budget_hours=a.budget_hours)


if __name__ == "__main__":
    try: sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr); sys.exit(2)
