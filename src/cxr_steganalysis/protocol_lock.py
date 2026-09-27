"""Explicit pre-test protocol freeze for the additional official-test patients.

This is an audit guard, not proof that no human has ever inspected the data.
Paths/hardware can change; scientific settings and semantic enrollment cannot.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from cxr_steganalysis.config import serializable_config
from cxr_steganalysis.experiment import effective_seeds, manifest_identity
from cxr_steganalysis.provenance import sha256_file


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


def freeze_protocol(configs, frame, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Protocol lock already exists; do not overwrite a pre-test declaration")
    if "test_cohort" not in frame or not (frame.test_cohort == "confirmatory_unseen_patient").any():
        raise ValueError("Freeze requires a manifest with additional confirmatory patients")
    records = [{"settings_identity": settings_identity(c), "settings": scientific_settings(c)} for c in configs]
    document = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    declaration="Scientific settings frozen before this confirmatory evaluation; not proof of prior non-exposure",
                    manifest_identity=manifest_identity(frame), configurations=records,
                    threshold_rule="maximum source-validation balanced accuracy; highest threshold on ties; fixed on both targets")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against an accidental replacement/race.
    with output.open("x") as f:
        json.dump(document, f, indent=2)
        f.write("\n")
    return document


def validate_test_access(config, frame, cohort, lock_path=None):
    is_full = config.get("dataset_profile") == "full_all_eligible" or "test_cohort" in frame
    if not is_full:
        return None
    if cohort is None:
        raise ValueError("Full NIH requires explicit --test-cohort; do not pool exposed and confirmatory patients")
    if cohort != "confirmatory_unseen_patient":
        return None
    if lock_path is None:
        raise ValueError("Confirmatory test requires --protocol-lock, created before inspecting its scores")
    lock_path = Path(lock_path)
    document = json.loads(lock_path.read_text())
    if document["manifest_identity"] != manifest_identity(frame):
        raise ValueError("Confirmatory enrollment differs from the frozen protocol")
    if settings_identity(config) not in {r["settings_identity"] for r in document["configurations"]}:
        raise ValueError("Scientific settings differ from the frozen protocol")
    return sha256_file(lock_path)
