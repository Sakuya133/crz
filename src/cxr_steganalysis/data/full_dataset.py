"""All-eligible NIH enrollment, preserving previously locked patient assignments."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from cxr_steganalysis.data.metadata import load_metadata, read_image_list
from cxr_steganalysis.data.split import assert_patient_disjoint, validate_official_patient_partition
from cxr_steganalysis.reproducibility import stable_seed
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.config import resolve_config_path


def assign_all_patients(metadata, locked, official_train, official_test, seed, exposed):
    train_patients, test_patients = validate_official_patient_partition(metadata, official_train, official_test)
    if set(metadata.image_id) != set(official_train) | set(official_test):
        raise ValueError("Official lists must cover the full metadata exactly")
    if locked.patient_id.duplicated().any():
        raise ValueError("Locked assignment has duplicate patients")
    old = locked.set_index("patient_id").split.to_dict()
    patients = set(metadata.patient_id)
    if not set(old) <= patients:
        raise ValueError("Locked patients absent from metadata")
    records = []
    for patient in sorted(patients):
        if patient in old:
            split = old[patient]
            if split not in {"train", "validation", "test"}:
                raise ValueError("Unknown locked split")
            if (patient in test_patients) != (split == "test"):
                raise ValueError(f"Locked assignment conflicts with official split: {patient}")
        elif patient in test_patients:
            split = "test"
        else:
            # 1/8 official-training patients -> validation; independent of image
            # availability/order. Existing pilot assignments are never redrawn.
            split = "validation" if stable_seed(seed, patient, "full-validation") % 8 == 0 else "train"
        cohort = ("pilot_exposed" if patient in exposed else "confirmatory_unseen_patient") if split == "test" else "not_test"
        records.append(dict(patient_id=patient, split=split, assignment_origin="pilot_locked" if patient in old else "full_hash_v1", test_cohort=cohort))
    return pd.DataFrame(records)


def full_preflight(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preflight output exists; use a new snapshot directory")
    metadata_path = resolve_config_path(config, "metadata")
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    expected_count = config.get("full_expected_metadata_images")
    if expected_count is not None and len(source) != int(expected_count):
        raise ValueError(f"Full NIH metadata expected {expected_count} images, got {len(source)}; do not call a truncated metadata subset full NIH")
    for column in ("Image Index", "Patient ID", "View Position"):
        if column not in source or source[column].str.strip().eq("").any():
            raise ValueError(f"Full NIH requires complete explicit {column}")
    metadata = load_metadata(metadata_path)
    if metadata.image_id.duplicated().any():
        raise ValueError("Duplicate metadata image_id")
    derived = metadata.image_id.str.extract(r"^(\d+)_\d+\.png$", expand=False)
    if derived.isna().any() or not derived.astype(int).astype(str).equals(metadata.patient_id.astype(int).astype(str)):
        raise ValueError("NIH filename patient ID does not match metadata")
    metadata.patient_id = metadata.patient_id.astype(int).astype(str)
    train_list = read_image_list(resolve_config_path(config, "official_train_list"))
    test_list = read_image_list(resolve_config_path(config, "official_test_list"))
    locked_path = resolve_config_path(config, "pilot_patient_assignments")
    pilot_path = resolve_config_path(config, "pilot_pair_manifest")
    locked = pd.read_csv(locked_path, dtype={"patient_id": str})
    pilot = pd.read_csv(pilot_path, dtype={"patient_id": str})
    locked.patient_id = locked.patient_id.astype(int).astype(str)
    pilot.patient_id = pilot.patient_id.astype(int).astype(str)
    exposed = set(pilot.loc[pilot.split == "test", "patient_id"])
    assignment = assign_all_patients(metadata, locked, train_list, test_list, int(config["seeds"]["split"]), exposed)
    local = {}
    raw = resolve_config_path(config, "raw_dir")
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.name in local:
                raise ValueError(f"Duplicate basename: {path.name}")
            local[path.name] = path
    if set(local) - set(metadata.image_id):
        raise ValueError("Local PNGs lack metadata")
    records = []
    for r in metadata.itertuples():
        path = local.get(r.image_id)
        record = dict(image_id=r.image_id, available=path is not None, reason="not_available", cover_path=str(path.resolve()) if path else "", file_bytes=None, mode=None, width=None, height=None)
        if path:
            record["file_bytes"] = path.stat().st_size
            record["mtime_ns"] = path.stat().st_mtime_ns
            try:
                with Image.open(path) as im:
                    im.load()
                    record.update(mode=im.mode, width=im.width, height=im.height)
                record["reason"] = "eligible" if r.view_position in {"AP", "PA"} and record["mode"] == "L" and min(record["width"], record["height"]) >= config["patch_size"] else "excluded_view" if r.view_position not in {"AP", "PA"} else "excluded_mode" if record["mode"] != "L" else "excluded_size"
            except (OSError, ValueError) as error:
                record["reason"] = "unreadable_requires_retrieval"
                record["error"] = str(error)
        records.append(record)
    inventory = metadata.merge(pd.DataFrame(records), on="image_id", validate="one_to_one").merge(assignment, on="patient_id", validate="many_to_one")
    eligible = inventory[inventory.reason == "eligible"].copy()
    assert_patient_disjoint(eligible)
    output.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output / "patient_assignments.csv", index=False)
    inventory.to_csv(output / "inventory.csv", index=False)
    eligible.to_csv(output / "splits.csv", index=False)
    inventory[inventory.reason == "not_available"][["image_id", "patient_id", "view_position", "split"]].to_csv(output / "missing_images.csv", index=False)
    counts = eligible.groupby(["split", "view_position", "test_cohort"]).agg(images=("image_id", "size"), patients=("patient_id", "nunique")).reset_index()
    counts.to_csv(output / "counts.csv", index=False)
    archives = sorted(p.name for p in Path(config["_project_root"]).glob("images_*.tar.gz"))
    expected = [f"images_{i:03d}.tar.gz" for i in range(1,13)]
    summary = dict(dataset_profile="full_all_eligible", status="ready_available_subset" if len(local) < len(metadata) else "complete_inventory",
                   metadata_images=len(metadata), available_pngs=len(local), eligible_images=len(eligible),
                   missing_images=int((inventory.reason == "not_available").sum()), exclusions=inventory.reason.value_counts().to_dict(),
                   patients_locked=len(locked), exposed_test_patients=len(exposed),
                   available_confirmatory_test_patients=eligible.loc[eligible.test_cohort == "confirmatory_unseen_patient", "patient_id"].nunique(),
                   archives_found=archives, archives_not_found_at_project_root=[n for n in expected if n not in archives],
                   archives_note="Inventory of archive filenames, not checksum verification; extracted images determine availability.",
                   bindings={str(p):sha256_file(p) for p in (metadata_path, locked_path, pilot_path, resolve_config_path(config,"official_train_list"), resolve_config_path(config,"official_test_list"))},
                   split_identity=sha256_file(output/"splits.csv"), assignment_identity=sha256_file(output/"patient_assignments.csv"),
                   estimated_srm_cache_all_metadata_gib=len(metadata)*2*34671*4/2**30,
                   estimated_one_payload_uncompressed_stego_gib=len(metadata)*1024*1024/2**30,
                   protocol_lock_status="preflight_is_not_a_protocol_freeze",
                   confirmatory_evaluation_requires_explicit_protocol_lock=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_size_control(config):
    """Write the size-matched manifest as a row subset of the prepared full manifest.

    Reuses the same cover/stego files; nothing is re-embedded or copied.
    """
    from cxr_steganalysis.data.portable import read_pairs
    from cxr_steganalysis.experiment import effective_seeds, enrollment_counts, manifest_identity, select_size_matched
    from cxr_steganalysis.lab import atomic_json
    source = resolve_config_path(config, "size_control_source_manifest")
    target = resolve_config_path(config, "pair_manifest")
    if source == target or target.parent == source.parent:
        raise ValueError("Size-control manifest needs its own directory next to, not over, the full manifest")
    complete = json.loads((source.parent / "complete.json").read_text())
    if complete["manifest_sha256"] != sha256_file(source):
        raise ValueError("Full manifest is incomplete or changed after prepare; finish full_data.py prepare first")
    frame = read_pairs(source)
    seed = effective_seeds(config)["sampling"]
    control = select_size_matched(frame, seed)
    text = control.to_csv(index=False)
    if target.exists() and target.read_text() != text:
        raise ValueError(f"Existing size-control manifest differs; refusing overwrite: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.csv")
        temporary.write_text(text)
        temporary.replace(target)
    quotas = {s: int((control.split == s).sum() // 2) for s in ("train", "validation")}
    receipt = dict(rule="train/validation: per split, min(AP, PA) pairs per view, ordered by stable_seed(sampling_seed, pair_id, split); test: all pairs kept",
                   sampling_seed=seed, seed_role="fixed for all models and training seeds; independent of training seed",
                   pairs_per_view=quotas, source_manifest=str(source), source_manifest_sha256=sha256_file(source),
                   source_identity=manifest_identity(frame), control_manifest_sha256=sha256_file(target),
                   control_identity=manifest_identity(control), counts_source=enrollment_counts(frame),
                   counts_control=enrollment_counts(control),
                   note="Equal pairs per view do not imply equal patients; see counts")
    receipt_path = target.parent / "size_control.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError(f"Existing size-control receipt differs: {receipt_path}")
    atomic_json(receipt_path, receipt)
    return receipt
