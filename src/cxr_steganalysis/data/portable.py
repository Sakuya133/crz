"""Import a private audited pilot without redrawing patient assignments.

The transport bundle contains CSVs and hashes, never images or checkpoints.
Pixel/byte checks on import bind the relocated paths to the same observations.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from cxr_steganalysis.config import resolve_config_path
from cxr_steganalysis.data.manifest import validate_pair_manifest, make_pair_id
from cxr_steganalysis.data.metadata import load_metadata, read_image_list
from cxr_steganalysis.data.split import assert_patient_disjoint, validate_official_patient_partition
from cxr_steganalysis.experiment import effective_seeds, manifest_identity, select_size_matched
from cxr_steganalysis.lab import atomic_json, require_empty
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.stego.lsb_matching import embed_lsb_matching_file


def read_pairs(path):
    return pd.read_csv(path, dtype={"patient_id": str, "seed": str}, keep_default_na=False)


def file_inventory(frame):
    records = {}
    for row in frame.itertuples():
        for kind in ("cover", "stego"):
            path = Path(getattr(row, kind + "_path"))
            stat = path.stat()
            records[str(path.resolve())] = [stat.st_size, stat.st_mtime_ns, str(getattr(row, kind + "_sha256"))]
    return records


def png_index(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PNG root is absent: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            if path.name in result:
                raise ValueError(f"Duplicate basename {path.name}: {result[path.name]} and {path}")
            result[path.name] = path.resolve()
    return result


def export_pilot(source, output):
    source, output = Path(source), Path(output)
    frame = read_pairs(source / "cover_stego.csv")
    assignment = pd.read_csv(source / "patient_assignments.csv", dtype={"patient_id": str})
    validate_pair_manifest(frame, check_files=False)
    check_assignments(frame, assignment)
    identity = manifest_identity(frame)
    require_empty(output)
    # No machine-specific paths need to leave the old workstation.
    for kind in ("cover", "stego"):
        frame[kind + "_path"] = frame[kind + "_path"].map(lambda x: kind + "/" + Path(x).name)
    frame.to_csv(output / "cover_stego.csv", index=False)
    assignment.to_csv(output / "patient_assignments.csv", index=False)
    atomic_json(output / "bundle.json", dict(
        format="audited-pilot-transport-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha256_file(source / "cover_stego.csv"),
        source_assignment_sha256=sha256_file(source / "patient_assignments.csv"),
        manifest_identity=identity, pairs=len(frame),
        files={n: sha256_file(output / n) for n in ("cover_stego.csv", "patient_assignments.csv")},
        note="Private patient-level transport; no images, stego or checkpoints. Export does not re-audit pixels."))
    return output / "bundle.json"


def check_assignments(frame, assignment):
    assert_patient_disjoint(frame)
    if assignment.patient_id.duplicated().any():
        raise ValueError("Duplicate patient assignment")
    if not set(assignment.split) <= {"train", "validation", "test"}:
        raise ValueError("Unknown assignment split")
    lookup = assignment.set_index("patient_id").split.to_dict()
    if any(lookup.get(r.patient_id) != r.split for r in frame.itertuples()):
        raise ValueError("Selected images do not match the locked patient assignment")


def validate_enrollment(frame, assignment, config):
    """Official-test protection is global, including patients with both AP and PA."""
    validate_pair_manifest(frame, check_files=False)
    check_assignments(frame, assignment)
    metadata_path = resolve_config_path(config, "metadata")
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    for column in ("Image Index", "Patient ID", "View Position"):
        if column not in source or source[column].str.strip().eq("").any():
            raise ValueError(f"Explicit metadata {column} is required; no inferred patient/view")
    metadata = load_metadata(metadata_path)
    if metadata.image_id.duplicated().any():
        raise ValueError("Duplicate metadata image ID")
    # NIH IDs are numeric: normalize for comparison, not for rewriting the bundle.
    derived = metadata.image_id.str.extract(r"^(\d+)_\d+\.png$", expand=False)
    if derived.isna().any() or not derived.astype(int).equals(metadata.patient_id.astype(int)):
        raise ValueError("NIH image ID and metadata patient ID disagree")
    metadata.patient_id = metadata.patient_id.astype(int).astype(str)
    reference = metadata.set_index("image_id")
    for row in frame.itertuples():
        if row.image_id not in reference.index:
            raise ValueError(f"Selected image missing metadata: {row.image_id}")
        record = reference.loc[row.image_id]
        if str(int(row.patient_id)) != record.patient_id or row.view_position != record.view_position:
            raise ValueError(f"Metadata identity/view mismatch: {row.image_id}")
        if row.view_position not in {"AP", "PA"}:
            raise ValueError("Only AP/PA are enrolled")
        if row.pair_id != make_pair_id(row.image_id, row.algorithm, row.payload_bpp):
            raise ValueError("Pair identity differs from image/algorithm/payload")
    if config["split"]["use_official_test"]:
        for key in ("official_train_list", "official_test_list"):
            if not resolve_config_path(config, key).is_file():
                raise FileNotFoundError(f"Official list is mandatory: {resolve_config_path(config, key)}")
        train = read_image_list(resolve_config_path(config, "official_train_list"))
        test = read_image_list(resolve_config_path(config, "official_test_list"))
        if set(metadata.image_id) != set(train) | set(test):
            raise ValueError("Official lists must cover metadata exactly")
        train_patients, test_patients = validate_official_patient_partition(metadata, train, test)
        for row in assignment.itertuples():
            patient = str(int(row.patient_id))
            if patient not in train_patients | test_patients or (patient in test_patients) != (row.split == "test"):
                raise ValueError(f"Locked assignment conflicts with official lists: patient {patient}")
    quotas = config.get("pilot", {}).get("quotas_per_view")
    if config.get("dataset_profile") == "pilot_existing" and quotas:
        for view in ("AP", "PA"):
            for split, expected in quotas.items():
                actual = int(((frame.view_position == view) & (frame.split == split)).sum())
                if actual != expected:
                    raise ValueError(f"Pilot quota {view}/{split}: {actual} != {expected}; no resampling")
    if set(frame.algorithm) != {config["algorithm"]} or not np.allclose(frame.payload_bpp, config["payload_bpp"], rtol=0, atol=0):
        raise ValueError("Manifest algorithm/payload differs from configured protocol")


def import_pilot(bundle, config, generate=False, stego_root=None, resume=False):
    bundle = Path(bundle)
    receipt = json.loads((bundle / "bundle.json").read_text())
    if config.get("dataset_profile") == "pilot_existing":
        for key, recorded in (("expected_pilot_source_sha256", "source_manifest_sha256"), ("expected_assignment_source_sha256", "source_assignment_sha256")):
            expected = config.get("lab", {}).get(key)
            if expected and receipt.get(recorded) != expected:
                raise ValueError("This is not the pinned audited pilot bundle; do not silently redraw the split")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or sha256_file(bundle / name) != digest:
            raise ValueError("Transport bundle has changed")
    frame = read_pairs(bundle / "cover_stego.csv")
    if manifest_identity(frame) != receipt["manifest_identity"]:
        raise ValueError("Bundle semantic identity changed")
    assignment = pd.read_csv(bundle / "patient_assignments.csv", dtype={"patient_id": str})
    validate_enrollment(frame, assignment, config)
    manifest = resolve_config_path(config, "pair_manifest")
    output = manifest.parent
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Import destination exists; use --resume for identical input or a new destination")
    raw = resolve_config_path(config, "raw_dir")
    covers = png_index(raw)
    stego_dir = Path(stego_root).resolve() if stego_root else resolve_config_path(config, "stego_dir")
    if generate and (stego_dir == raw or raw in stego_dir.parents):
        raise ValueError("Writable stego directory must not be inside raw images")
    stegos = png_index(stego_dir) if stego_dir.is_dir() else {}
    expected_binding = dict(bundle_sha256=sha256_file(bundle / "bundle.json"), raw_dir=str(raw), stego_dir=str(stego_dir), embedding_seed=effective_seeds(config)["embedding"])
    binding_path = output / "import_binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != expected_binding:
        raise ValueError("Import binding changed; select a new destination")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(binding_path, expected_binding)
    for i, row in enumerate(frame.itertuples()):
        if row.image_id not in covers:
            raise FileNotFoundError(f"Selected cover missing: {row.image_id}")
        cover = covers[row.image_id]
        if sha256_file(cover) != row.cover_sha256:
            raise ValueError(f"Cover checksum differs: {row.image_id}")
        with Image.open(cover) as im:
            im.load()
            if im.mode != "L" or min(im.size) < config["patch_size"]:
                raise ValueError(f"Pilot mode/dimensions differ: {row.image_id}")
        stego_name = Path(row.stego_path).name
        stego = stegos.get(stego_name, stego_dir / stego_name)
        if not stego.exists():
            if not generate:
                raise FileNotFoundError(f"Stego absent: {stego}; pass --generate-stego or supply existing PNGs")
            result = embed_lsb_matching_file(cover, float(row.payload_bpp), effective_seeds(config)["embedding"], row.image_id)
            if str(result.metadata.seed) != str(row.seed) or result.metadata.changed_pixels != row.changed_pixels:
                raise ValueError("Regenerated stego seed/count does not match pilot")
            stego.parent.mkdir(parents=True, exist_ok=True)
            temporary = stego.with_suffix(".tmp.png")
            Image.fromarray(result.image).save(temporary)
            if sha256_file(temporary) != row.stego_sha256:
                raise ValueError(f"Regenerated PNG bytes differ for {row.image_id}; possible encoder/version change. Supply the original stego files; no equivalence silently assumed.")
            temporary.replace(stego)
        if sha256_file(stego) != row.stego_sha256:
            raise ValueError(f"Stego checksum differs: {row.image_id}")
        frame.loc[i, ["cover_path", "stego_path"]] = [str(cover), str(stego.resolve())]
    validate_pair_manifest(frame, check_files=True)
    if manifest_identity(frame) != receipt["manifest_identity"]:
        raise AssertionError("Relocation changed scientific identity")
    csv_text = frame.to_csv(index=False)
    if manifest.exists() and manifest.read_text() != csv_text:
        raise ValueError("Existing relocated manifest differs; refusing overwrite")
    tables = [(resolve_config_path(config, "patient_assignments"), assignment),
              (resolve_config_path(config, "split_manifest"), frame[["image_id", "patient_id", "view_position", "finding_labels", "split", "cover_path"]]),
              (resolve_config_path(config, "generation_manifest"), frame[["image_id", "stego_path", "algorithm", "payload_bpp", "seed", "cover_sha256", "stego_sha256", "embedded_bits", "changed_pixels"]]),
              (manifest, frame)]
    for target, table in tables:
        text = table.to_csv(index=False)
        if target.exists() and target.read_text() != text:
            raise ValueError(f"Imported artifact differs; refusing overwrite: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.csv")
            temporary.write_text(text)
            temporary.replace(target)
    relocation = dict(status="verified_relocated", pairs=len(frame), patients=int(frame.patient_id.nunique()),
                      source_manifest_sha256=receipt["source_manifest_sha256"], transport_manifest_sha256=receipt["files"]["cover_stego.csv"],
                      relocated_manifest_sha256=sha256_file(manifest), semantic_identity_before=receipt["manifest_identity"], semantic_identity_after=manifest_identity(frame),
                      checksums="all cover/stego PNG bytes verified; mode/dimensions verified", changes="paths only; patient/image/pair/view/split/payload/seed/checksums unchanged")
    atomic_json(output / "relocation.json", relocation)
    atomic_json(output / "verified_files.json", dict(manifest_sha256=sha256_file(manifest), files=file_inventory(frame)))
    return relocation


def check_data(config, verify_hashes=False):
    manifest = resolve_config_path(config, "pair_manifest")
    frame = read_pairs(manifest)
    assignments = pd.read_csv(resolve_config_path(config, "patient_assignments"), dtype={"patient_id": str})
    png_index(resolve_config_path(config, "raw_dir"))
    validate_enrollment(frame, assignments, config)
    profile = config.get("dataset_profile")
    full_frame = frame
    if profile == "full_size_matched":
        # The control must be exactly the deterministic subset of the complete
        # full manifest; the full manifest itself gets the full-enrollment checks.
        source = resolve_config_path(config, "size_control_source_manifest")
        receipt = json.loads((manifest.parent / "size_control.json").read_text())
        full_frame = read_pairs(source)
        seed = effective_seeds(config)["sampling"]
        if sha256_file(source) != receipt["source_manifest_sha256"] or receipt["sampling_seed"] != seed:
            raise ValueError("Size control receipt no longer matches the full manifest/sampling seed")
        if manifest_identity(select_size_matched(full_frame, seed)) != manifest_identity(frame):
            raise ValueError("Size-matched manifest is not the deterministic subset of the full manifest")
    if profile in {"full_all_eligible", "full_size_matched"}:
        snapshot = resolve_config_path(config, "split_manifest")
        summary = json.loads((snapshot.parent / "summary.json").read_text())
        if summary["missing_images"] or summary["exclusions"].get("unreadable_requires_retrieval", 0):
            raise ValueError("Full inventory is incomplete; no silent available-only subset")
        if sha256_file(snapshot) != summary["split_identity"]:
            raise ValueError("Full enrollment snapshot changed")
        for path, digest in summary["bindings"].items():
            if sha256_file(path) != digest:
                raise ValueError("Full metadata/official/pilot binding changed")
        enrollment = pd.read_csv(snapshot, dtype={"patient_id":str})
        if set(full_frame.image_id) != set(enrollment.image_id) or len(full_frame) != len(enrollment):
            raise ValueError("Full manifest must retain every eligible image; no resampling")
    validate_pair_manifest(frame, verify_hashes=verify_hashes)
    # Avoid rehashing unchanged audited PNGs on every queue resume. Any changed
    # size/mtime or missing proof forces a byte check, never silent acceptance.
    receipt_path = manifest.parent / "verified_files.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    recorded = previous.get("files", {})
    current = file_inventory(frame)
    manifest_digest = sha256_file(manifest)
    rechecked = 0
    for path, info in current.items():
        if not verify_hashes and (previous.get("manifest_sha256") != manifest_digest or recorded.get(path) != info):
            if sha256_file(path) != info[2]:
                raise ValueError(f"Changed image checksum mismatch: {path}")
            rechecked += 1
    # Checksums prove byte identity, not decodability. Fully decode every file
    # not yet decoded under this receipt; files already decoded with unchanged
    # path/size/mtime/hash get a header-only size check (saves ~30 min/resume).
    decoded_before = previous.get("png_decode_verified", False)
    decoded = 0
    for row in frame.itertuples():
        for path in (row.cover_path, row.stego_path):
            key = str(Path(path).resolve())
            with Image.open(path) as im:
                if not (decoded_before and recorded.get(key) == current[key]):
                    im.load()
                    decoded += 1
                if min(im.size) < config["patch_size"]:
                    raise ValueError(f"Image smaller than crop: {path}")
    counts = frame.groupby(["split", "view_position"]).agg(pairs=("pair_id", "size"), patients=("patient_id", "nunique")).reset_index().to_dict("records")
    atomic_json(receipt_path, dict(manifest_sha256=sha256_file(manifest), files=current, png_decode_verified=True))
    return dict(status="ready", pairs=len(frame), patients=int(frame.patient_id.nunique()), counts=counts,
                manifest_sha256=sha256_file(manifest), manifest_identity=manifest_identity(frame),
                checksums_verified=verify_hashes, changed_files_rehashed=rechecked, files_fully_decoded=decoded,
                unchanged_byte_binding="Previously verified hashes reused only when path/size/mtime and manifest binding agree", seeds=effective_seeds(config))
