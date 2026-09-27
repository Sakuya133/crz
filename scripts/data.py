#!/usr/bin/env python3
"""Private audited-pilot transport/import and strict dataset checks; never download NIH."""
import argparse
import json
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.lab import load_lab_config, atomic_json
from cxr_steganalysis.data.portable import export_pilot, import_pilot, check_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export-pilot", help="Export small private CSV+hash bundle, no images")
    export.add_argument("--source-manifests", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("import-pilot", "check"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", type=Path, default=Path("configs/pilot_existing.yaml"))
        cmd.add_argument("--local", type=Path)
        if name == "import-pilot":
            cmd.add_argument("--bundle", type=Path, required=True)
            cmd.add_argument("--generate-stego", action="store_true")
            cmd.add_argument("--stego-root", type=Path)
            cmd.add_argument("--resume", action="store_true")
        else:
            cmd.add_argument("--verify-hashes", action="store_true")
            cmd.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.action == "export-pilot":
        print(export_pilot(a.source_manifests, a.output)); return
    config = load_lab_config(a.config, a.local)
    result = import_pilot(a.bundle, config, a.generate_stego, a.stego_root, a.resume) if a.action == "import-pilot" else check_data(config, a.verify_hashes)
    if getattr(a, "output", None): atomic_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
