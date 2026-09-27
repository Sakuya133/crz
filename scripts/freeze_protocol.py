#!/usr/bin/env python3
"""Freeze effective run configurations before opening additional patient test results."""
import argparse
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
import pandas as pd
from cxr_steganalysis.config import load_config
from cxr_steganalysis.protocol_lock import freeze_protocol

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-configs", type=Path, nargs="+", required=True,
                   help="Effective configs saved by training, or equivalent prespecified configs")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frame = pd.read_csv(a.manifest, dtype={"patient_id": str, "seed": str})
    result = freeze_protocol([load_config(p) for p in a.run_configs], frame, a.output)
    print(f"Frozen {len(result['configurations'])} scientific configurations: {a.output}")
