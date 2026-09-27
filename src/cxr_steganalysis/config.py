"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1337,
    "patch_size": 256,
    "patches_per_image": 1,
    "payload_bpp": 0.2,
    "algorithm": "lsb_matching",
    "batch_size": 8,
    "epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "mixed_precision": True,
    "early_stopping_patience": 3,
    "paths": {
        "metadata": "data/metadata/Data_Entry_2017.csv",
        "official_train_list": "data/metadata/train_val_list.txt",
        "official_test_list": "data/metadata/test_list.txt",
        "raw_dir": "data/raw",
        "stego_dir": "data/stego",
        "split_manifest": "data/manifests/splits.csv",
        "patient_assignments": "data/manifests/patient_assignments.csv",
        "generation_manifest": "data/manifests/stego_generation.csv",
        "pair_manifest": "data/manifests/cover_stego.csv",
        "output_dir": "outputs",
    },
    "split": {
        "train_fraction": 0.70,
        "validation_fraction": 0.10,
        "test_fraction": 0.20,
        "use_official_test": True,
    },
    # Pilot filtering is opt-in so existing full-dataset configurations retain
    # their previous behavior.
    "pilot": {
        "available_only": False,
        "allowed_views": ["AP", "PA"],
        "required_image_mode": None,
        "sampling_seed": 1337,
        "quotas_per_view": None,
    },
    "data": {"allow_grayscale_conversion": False, "normalize": "unit_range"},
    "model": {"name": "highpass", "base_channels": 16},
    "training": {"train_view": "ALL", "device": "auto", "resume": None},
    "evaluation": {"test_views": ["PA", "AP"]},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config, apply defaults, and record its project root."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    parent = loaded.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / str(parent)).resolve()
        if parent_path == config_path:
            raise ValueError("Config cannot extend itself")
        inherited = load_config(parent_path)
        base = serializable_config(inherited)
        loaded.setdefault("project_root", inherited["_project_root"])
    else:
        base = deepcopy(DEFAULT_CONFIG)
    config = _deep_update(base, loaded)
    project_root_value = config.pop("project_root", None)
    if project_root_value:
        root = Path(project_root_value).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
        root = root.resolve()
    else:
        root = config_path.parent.parent.resolve()
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(root)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    payload = float(config["payload_bpp"])
    if not 0.0 <= payload <= 1.0:
        raise ValueError(f"payload_bpp must be in [0, 1], got {payload}")
    if int(config["patch_size"]) <= 0:
        raise ValueError("patch_size must be positive")
    if int(config["patches_per_image"]) <= 0:
        raise ValueError("patches_per_image must be positive")
    fractions = config["split"]
    total = sum(float(fractions[name]) for name in (
        "train_fraction", "validation_fraction", "test_fraction"
    ))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    if config["algorithm"] != "lsb_matching":
        raise ValueError("This pilot currently supports algorithm=lsb_matching only")
    if str(config["model"].get("name", "highpass")).lower() not in {"highpass", "srnet", "hsmnet"}:
        raise ValueError("model.name must be highpass, srnet or hsmnet")
    config["model"]["name"] = str(config["model"].get("name", "highpass")).lower()
    selection = config["training"].get("selection_metric", "source_validation_auc")
    if selection not in {"source_validation_auc", "macro_view_auc"}:
        raise ValueError("Unknown validation selection_metric; refusing a silent fallback")
    pilot = config.get("pilot", {})
    allowed_views = [str(view).upper() for view in pilot.get("allowed_views", [])]
    if len(allowed_views) != len(set(allowed_views)):
        raise ValueError("pilot.allowed_views must not contain duplicates")
    pilot["allowed_views"] = allowed_views
    if pilot.get("available_only") and not allowed_views:
        raise ValueError("pilot.allowed_views must be non-empty when available_only is enabled")
    required_mode = pilot.get("required_image_mode")
    if required_mode is not None and not str(required_mode).strip():
        raise ValueError("pilot.required_image_mode must be null or a non-empty string")
    quotas = pilot.get("quotas_per_view")
    if quotas is not None:
        expected_splits = {"train", "validation", "test"}
        if set(quotas) != expected_splits:
            raise ValueError(
                "pilot.quotas_per_view must contain exactly train, validation, and test"
            )
        for split, quota in quotas.items():
            if int(quota) <= 0:
                raise ValueError(f"pilot quota for {split} must be positive")
            quotas[split] = int(quota)
    pilot["sampling_seed"] = int(pilot.get("sampling_seed", config["seed"]))


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    """Resolve one entry in config.paths relative to the project root."""
    if key not in config["paths"]:
        raise KeyError(f"Unknown config path key: {key}")
    path = Path(config["paths"][key]).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
