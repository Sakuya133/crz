"""Whole-plan pre-test protocol freeze for the additional official-test patients.

One lock declares every preset, model, training seed, hyperparameter, metric and
comparison before the first confirmatory evaluation; staged queues must be members.
This is an audit guard, not proof that no human has ever inspected the data.
Paths/hardware can change; scientific settings and semantic enrollment cannot.
"""
from __future__ import annotations
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from cxr_steganalysis.config import resolve_config_path, serializable_config
from cxr_steganalysis.experiment import effective_seeds, enrollment_counts, manifest_identity
from cxr_steganalysis.provenance import sha256_file

CONFIRMATORY = "confirmatory_unseen_patient"
THRESHOLD_RULE = "maximum source-validation balanced accuracy; highest threshold on ties; fixed on both targets"


def scientific_settings(config):
    c = serializable_config(config)
    training = {k: v for k, v in c.get("training", {}).items() if k not in {"device", "resume"}}
    model = {k: v for k, v in c.get("model", {}).items() if k != "author_source"}
    return dict(protocol_id=c.get("protocol_id"), dataset_profile=c.get("dataset_profile"),
                algorithm=c["algorithm"], payload_bpp=c["payload_bpp"],
                patch_size=c["patch_size"], patches_per_image=c["patches_per_image"],
                batch_size=c["batch_size"], epochs=c["epochs"], learning_rate=c["learning_rate"],
                weight_decay=c["weight_decay"], mixed_precision=c["mixed_precision"],
                early_stopping_patience=c["early_stopping_patience"],
                seeds=effective_seeds(c), data=c["data"],
                model=model, training=training, evaluation=c.get("evaluation", {}),
                classifier=c.get("classifier"), mixed=c.get("mixed"))


def settings_identity(config):
    return hashlib.sha256(json.dumps(scientific_settings(config), sort_keys=True).encode()).hexdigest()


def read_manifest(path):
    # Same parsing as the evaluators, so identities agree exactly.
    return pd.read_csv(path, dtype={"patient_id": str, "seed": str})


def _comparisons(role, models):
    rows = []
    for model in models:
        for target, other in (("AP", "PA"), ("PA", "AP")):
            rows.append(dict(name=f"mismatch_{model}_{target}", role=role, test=f"{other}->{target} minus {target}->{target}",
                             primary=role == "primary"))
        if model != "highpass":
            for source in ("AP", "PA"):
                for target in ("AP", "PA"):
                    rows.append(dict(name=f"method_{model}_minus_highpass_{source}_{target}", role=role, primary=False))
    return rows


def freeze_plan(entries, output):
    """entries: [dict(role, config, manifest_path, runs=[(run_id, run_config)])]."""
    from cxr_steganalysis.evaluation.cluster_bootstrap import BOOTSTRAP_METRICS  # torch-heavy; keep lazy
    output = Path(output)
    if output.exists():
        raise FileExistsError("Protocol lock already exists; do not overwrite a pre-test declaration")
    if sorted(e["role"] for e in entries).count("primary") != 1:
        raise ValueError("Exactly one preset must be lab.analysis_role: primary")
    presets, records, comparisons = [], [], []
    for entry in entries:
        config, frame = entry["config"], read_manifest(entry["manifest_path"])
        if "test_cohort" not in frame or not (frame.test_cohort == CONFIRMATORY).any():
            raise ValueError("Freeze requires a manifest with additional confirmatory patients")
        if config.get("dataset_profile") == "full_size_matched":
            for split in ("train", "validation"):
                counts = frame[frame.split == split].view_position.value_counts()
                if counts.get("AP", 0) != counts.get("PA", 0):
                    raise ValueError(f"Size-matched manifest is unbalanced in {split}: {counts.to_dict()}")
        identity = manifest_identity(frame)
        # AP-vs-PA comparisons within one architecture/seed may differ only in source view.
        by_cell = {}
        for run_id, run_config in entry["runs"]:
            settings = scientific_settings(run_config)
            cell = deepcopy(settings)
            cell["training"].pop("train_view", None)
            by_cell.setdefault((settings["model"]["name"], settings["seeds"]["training"]), []).append(json.dumps(cell, sort_keys=True))
            records.append(dict(run_id=run_id, role=entry["role"], settings_identity=settings_identity(run_config),
                                manifest_identity=identity, settings=settings))
        for cell, variants in by_cell.items():
            if len(set(variants)) != 1:
                raise ValueError(f"AP/PA runs differ beyond source view for {cell}")
        lab = config["lab"]
        presets.append(dict(role=entry["role"], protocol_id=config["protocol_id"], dataset_profile=config["dataset_profile"],
                            manifest_sha256=sha256_file(entry["manifest_path"]), manifest_identity=identity,
                            sampling_seed=effective_seeds(config)["sampling"], models=lab["models"],
                            training_seeds=lab["training_seeds"], bootstrap_replicates=lab.get("bootstrap_replicates", 10000),
                            counts=enrollment_counts(frame)))
        comparisons += _comparisons(entry["role"], lab["models"])
    document = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    declaration="Whole plan frozen before any confirmatory evaluation; staged runs must be listed here. Not proof of prior non-exposure",
                    presets=presets, configurations=records, test_cohort=CONFIRMATORY, metrics=list(BOOTSTRAP_METRICS),
                    threshold_rule=THRESHOLD_RULE, comparisons=comparisons,
                    statistics="95% percentile patient-cluster paired bootstrap per training seed; seeds never pooled as patients; seed mean/SD descriptive")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against an accidental replacement/race.
    with output.open("x") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    return document


def _locked_pairs(document):
    return {(r["settings_identity"], r.get("manifest_identity", document.get("manifest_identity"))) for r in document["configurations"]}


def verify_plan_membership(lock_path, run_configs):
    """Refuse to start a full queue whose runs were not declared in the plan lock."""
    lock_path = Path(lock_path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"Whole-plan lock missing: {lock_path}. Run scripts/freeze_protocol.py --plan ... before training/evaluation")
    pairs = _locked_pairs(json.loads(lock_path.read_text()))
    identities = {}
    for config in run_configs:
        path = resolve_config_path(config, "pair_manifest")
        identities.setdefault(path, manifest_identity(read_manifest(path)))
        if (settings_identity(config), identities[path]) not in pairs:
            raise ValueError(f"Run {config['paths']['output_dir']} is not part of the frozen plan {lock_path}")
    return sha256_file(lock_path)


def validate_test_access(config, frame, cohort, lock_path=None):
    is_full = str(config.get("dataset_profile", "")).startswith("full_") or "test_cohort" in frame
    if not is_full:
        return None
    if cohort is None:
        raise ValueError("Full NIH requires explicit --test-cohort; do not pool exposed and confirmatory patients")
    if cohort != CONFIRMATORY:
        return None
    if lock_path is None:
        raise ValueError("Confirmatory test requires --protocol-lock, created before inspecting its scores")
    lock_path = Path(lock_path)
    pairs = _locked_pairs(json.loads(lock_path.read_text()))
    identity = manifest_identity(frame)
    if identity not in {m for _, m in pairs}:
        raise ValueError("Confirmatory enrollment differs from the frozen protocol")
    if (settings_identity(config), identity) not in pairs:
        raise ValueError("Scientific settings differ from the frozen protocol")
    return sha256_file(lock_path)
