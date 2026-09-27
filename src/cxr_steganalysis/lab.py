"""Portable settings and atomic receipts shared by the lab CLIs."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
from cxr_steganalysis.config import load_config, serializable_config


def load_lab_config(path, local=None):
    config = load_config(path)
    if local:
        override = yaml.safe_load(Path(local).read_text()) or {}
        if set(override) - {"paths"} or not isinstance(override.get("paths", {}), dict):
            raise ValueError("Local configuration is paths-only; scientific changes need a versioned preset")
        unknown = set(override.get("paths", {})) - set(config["paths"])
        if unknown:
            raise ValueError(f"Unknown local path keys: {sorted(unknown)}")
        config["paths"].update(override.get("paths", {}))
    return config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_config(path, config):
    value = serializable_config(config)
    value["project_root"] = config["_project_root"]
    path = Path(path)
    if path.exists():
        if yaml.safe_load(path.read_text()) != value:
            raise ValueError(f"Resolved config changed: {path}. Use a new queue directory.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=True))


def require_empty(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
