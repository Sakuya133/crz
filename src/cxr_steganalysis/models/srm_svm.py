"""Full spatial rich-model features with a train-only scaled Linear SVM.

The 34,671-dimensional SRM extractor is provided by the optional ``sealwatch``
dependency. The classifier is an explicit project adaptation: StandardScaler plus
LinearSVC, not the FLD ensemble from the SRM paper. SVM outputs are decision margins,
not probabilities.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
import warnings
import resource
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from cxr_steganalysis.data.dataset import deterministic_crop_coordinates
from cxr_steganalysis.data.manifest import validate_pair_manifest
from cxr_steganalysis.evaluation.cross_view import (
    _finalize_predictions,
    save_evaluation_results,
)
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.reproducibility import runtime_environment
from cxr_steganalysis.training.metrics import binary_metrics, select_threshold


SRM_DIMENSION = 34_671
SRM_MODEL_NAME = "srm_svm"
SRM_MODEL_VERSION = "sealwatch-full-srm-standardscaler-linearsvc-v1"


def _bounded_ordered_map(executor, function, jobs, max_pending):
    """Ordered results without eagerly submitting the entire full NIH inventory."""
    jobs = iter(jobs)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(jobs)))
        except StopIteration:
            break
    while pending:
        result = pending.popleft().result()
        yield result
        try:
            pending.append(executor.submit(function, next(jobs)))
        except StopIteration:
            pass


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh_directory(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def srm_vector(image: np.ndarray) -> tuple[np.ndarray, list[list[Any]]]:
    """Extract a deterministic full SRM vector from an unnormalized uint8 patch."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Full SRM requires a 2D uint8 image, not normalized input")
    try:
        import sealwatch
    except ImportError as error:
        raise ImportError(
            "Full SRM requires the optional sealwatch==2025.9 dependency; no "
            "approximate high-pass substitute is used"
        ) from error
    grouped = sealwatch.srm.extract(image)
    keys = sorted(grouped)
    schema = [[key, list(np.asarray(grouped[key]).shape)] for key in keys]
    vector = np.concatenate([
        np.asarray(grouped[key]).ravel(order="C") for key in keys
    ]).astype(np.float32)
    if vector.size != SRM_DIMENSION or not np.isfinite(vector).all():
        raise ValueError(
            f"Invalid full SRM vector: expected {SRM_DIMENSION}, got {vector.size}"
        )
    return vector, schema


def _load_uint8(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"SRM input must be mode L: {path}")
        return np.asarray(image, dtype=np.uint8).copy()


def _sample_index(
    frame: pd.DataFrame,
    patch_size: int,
    patches_per_image: int,
    crop_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    feature_row = 0
    for row in frame.itertuples(index=False):
        with Image.open(row.cover_path) as image:
            width, height = image.size
        for patch_index in range(patches_per_image):
            top, left = deterministic_crop_coordinates(
                height,
                width,
                patch_size,
                crop_seed,
                str(row.pair_id),
                patch_index,
            )
            for class_name, label in (("cover", 0), ("stego", 1)):
                records.append({
                    "feature_row": feature_row,
                    "pair_id": str(row.pair_id),
                    "image_id": str(row.image_id),
                    "patient_id": str(row.patient_id),
                    "split": str(row.split),
                    "view_position": str(row.view_position),
                    "class_name": class_name,
                    "label": label,
                    "patch_index": patch_index,
                    "crop_top": top,
                    "crop_left": left,
                    "payload_bpp": float(row.payload_bpp),
                    "algorithm": str(row.algorithm),
                })
                feature_row += 1
    result = pd.DataFrame(records)
    if "test_cohort" in frame:
        result["test_cohort"] = result.image_id.map(frame.set_index("image_id").test_cohort)
    return result


def extract_srm_features(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    patch_size: int,
    patches_per_image: int,
    crop_seed: int,
    workers: int = 1,
) -> Path:
    """Extract a resumable feature matrix aligned to the CNN's shared crops."""
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str})
    validate_pair_manifest(frame, check_files=True)
    if frame["payload_bpp"].nunique() != 1 or frame["algorithm"].nunique() != 1:
        raise ValueError("One SRM feature cache cannot mix payloads or algorithms")
    version = importlib.metadata.version("sealwatch")
    binding = {
        "manifest_sha256": sha256_file(manifest_path),
        "sealwatch_version": version,
        "dimension": SRM_DIMENSION,
        "patch_size": int(patch_size),
        "patches_per_image": int(patches_per_image),
        "crop_seed": int(crop_seed),
        "input": "shared uint8 cover/stego crop after full-image embedding",
    }
    output = Path(output_dir)
    complete_path = output / "complete.json"
    progress_path = output / "progress.json"
    feature_path = output / "features.npy"
    index_path = output / "samples.csv"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete["binding"] != binding:
            raise ValueError("Existing SRM feature cache has a different protocol binding")
        if sha256_file(feature_path) != complete["features_sha256"]:
            raise ValueError("Existing SRM feature cache failed its SHA-256 check")
        if sha256_file(index_path) != complete["samples_sha256"]:
            raise ValueError("Existing SRM sample index failed its SHA-256 check")
        return feature_path

    samples = _sample_index(frame, patch_size, patches_per_image, crop_seed)
    if progress_path.is_file():
        progress = _read_json(progress_path)
        if progress["binding"] != binding:
            raise ValueError("Partial SRM cache has a different protocol binding")
        if sha256_file(index_path) != progress["samples_sha256"]:
            raise ValueError("Partial SRM sample index failed its SHA-256 check")
        persisted = pd.read_csv(index_path, dtype={"patient_id": str})
        pd.testing.assert_frame_equal(samples, persisted, check_dtype=False)
        start_pair = int(progress["pairs_completed"])
        schema = progress["schema"]
        features = np.lib.format.open_memmap(feature_path, mode="r+")
    else:
        _fresh_directory(output)
        samples.to_csv(index_path, index=False)
        start_pair = 0
        schema = None
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), SRM_DIMENSION),
        )
        _write_json(progress_path, {
            "binding": binding,
            "pairs_completed": 0,
            "schema": None,
            "samples_sha256": sha256_file(index_path),
        })
    if features.shape != (len(samples), SRM_DIMENSION) or features.dtype != np.float32:
        raise ValueError("Malformed partial SRM feature matrix")

    started = time.perf_counter()
    accumulated = float(progress.get("elapsed_seconds", 0)) if progress_path.is_file() and start_pair else 0.0
    rows_per_pair = 2 * patches_per_image
    if workers < 1 or workers > 8:
        raise ValueError("SRM workers must be 1..8; budget RAM explicitly")
    jobs = ((str(r.cover_path), str(r.stego_path), str(r.pair_id), patch_size, patches_per_image, crop_seed) for r in frame.iloc[start_pair:].itertuples())
    context = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_srm_worker_init) if workers > 1 else nullcontext()
    with context as pool:
        stream = _bounded_ordered_map(pool, _extract_pair_features, jobs, 2*workers) if pool else map(_extract_pair_features, jobs)
        for pair_index, (vectors, current_schema) in enumerate(stream, start=start_pair):
            if schema is None:
                schema = current_schema
            if schema != current_schema:
                raise ValueError("SRM feature ordering/schema changed during extraction")
            offset = pair_index * rows_per_pair
            features[offset:offset+rows_per_pair] = vectors
            if (pair_index + 1) % 10 == 0 or pair_index + 1 == len(frame):
                print(f"SRM pairs={pair_index+1}/{len(frame)} elapsed_session={time.perf_counter()-started:.1f}s", flush=True)
                features.flush()
                _write_json(progress_path, {
                    "binding": binding,
                    "pairs_completed": pair_index + 1,
                    "schema": schema,
                    "samples_sha256": sha256_file(index_path),
                    "elapsed_seconds": accumulated + time.perf_counter() - started,
                })
    features.flush()
    _write_json(complete_path, {
        "binding": binding,
        "schema": schema,
        "features_sha256": sha256_file(feature_path),
        "samples_sha256": sha256_file(index_path),
        "elapsed_seconds": accumulated + time.perf_counter() - started,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    return feature_path


def _srm_worker_init():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)


def _extract_pair_features(job):
    cover_path, stego_path, pair_id, patch_size, patches, seed = job
    cover, stego = _load_uint8(cover_path), _load_uint8(stego_path)
    if cover.shape != stego.shape:
        raise ValueError(f"Mismatched pair {pair_id}")
    vectors, schema = [], None
    for patch in range(patches):
        top, left = deterministic_crop_coordinates(*cover.shape, patch_size, seed, pair_id, patch)
        for image in (cover, stego):
            vector, current = srm_vector(image[top:top+patch_size, left:left+patch_size])
            if schema is not None and current != schema:
                raise ValueError("SRM schema differs within pair")
            schema = current
            vectors.append(vector)
    return np.stack(vectors), schema


def _load_feature_cache(
    manifest_path: str | Path,
    feature_dir: str | Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    feature_dir = Path(feature_dir)
    complete = _read_json(feature_dir / "complete.json")
    if complete["binding"]["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("SRM feature cache does not match the pair manifest")
    if sha256_file(feature_dir / "features.npy") != complete["features_sha256"]:
        raise ValueError("SRM feature cache failed its SHA-256 check")
    if sha256_file(feature_dir / "samples.csv") != complete["samples_sha256"]:
        raise ValueError("SRM sample index failed its SHA-256 check")
    samples = pd.read_csv(feature_dir / "samples.csv", dtype={"patient_id": str})
    features = np.load(feature_dir / "features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (len(samples), SRM_DIMENSION):
        raise ValueError("SRM feature matrix and sample index have different lengths")
    return samples, features, complete


def _selected_rows(samples: pd.DataFrame, split: str, view: str, cohort: str | None = None) -> np.ndarray:
    mask = samples["split"].eq(split) & samples["view_position"].eq(view.upper())
    if cohort is not None:
        if "test_cohort" not in samples:
            raise ValueError("SRM cache lacks test cohort; cannot substitute exposed patients")
        mask &= samples.test_cohort.eq(cohort)
    rows = samples.loc[mask, "feature_row"].to_numpy(dtype=np.int64)
    if len(rows) == 0:
        raise ValueError(f"No SRM samples for split={split}, view={view}")
    return rows


def train_srm_svm(
    manifest_path: str | Path,
    feature_dir: str | Path,
    output_dir: str | Path,
    *,
    source_view: str,
    seed: int,
    c_grid: Iterable[float] = (0.01, 0.1, 1.0),
) -> Path:
    """Fit scaler on training only; tune LinearSVC C on source validation."""
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ImportError as error:
        raise ImportError("SRM+SVM requires the optional scikit-learn dependency") from error
    source = source_view.upper()
    if source not in {"AP", "PA"}:
        raise ValueError("source_view must be AP or PA")
    candidates = sorted({float(value) for value in c_grid})
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("c_grid must contain positive values")
    samples, features, feature_info = _load_feature_cache(manifest_path, feature_dir)
    output = _fresh_directory(output_dir)
    train_rows = _selected_rows(samples, "train", source)
    validation_rows = _selected_rows(samples, "validation", source)
    scaler = StandardScaler()
    # A memmap does not prevent advanced-indexing, scaling and liblinear copies.
    import psutil
    estimated_peak = (len(train_rows)*5 + len(validation_rows)*2) * SRM_DIMENSION * 8
    if estimated_peak > psutil.virtual_memory().available * .7:
        raise MemoryError(f"SRM solver conservative allocation estimate {estimated_peak/2**30:.1f} GiB exceeds 70% available RAM; use larger RAM, not fewer features/samples")
    started = time.perf_counter()
    x_train = scaler.fit_transform(features[train_rows].astype(np.float64))
    x_validation = scaler.transform(features[validation_rows].astype(np.float64))
    y_train = samples.iloc[train_rows]["label"].to_numpy(dtype=np.int64)
    y_validation = samples.iloc[validation_rows]["label"].to_numpy(dtype=np.int64)
    best: dict[str, Any] | None = None
    history: list[dict[str, float]] = []
    for c_value in candidates:
        candidate_started = time.perf_counter()
        print(f"Fitting full SRM + LinearSVC: source={source}, C={c_value}, train_rows={len(train_rows)}, features={SRM_DIMENSION}", flush=True)
        estimator = LinearSVC(
            C=c_value,
            dual=True,
            max_iter=20_000,
            tol=1e-4,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(x_train, y_train)
        margins = estimator.decision_function(x_validation)
        threshold = select_threshold(y_validation, margins)
        auc = binary_metrics(y_validation, margins, threshold)["roc_auc"]
        fit_seconds = time.perf_counter() - candidate_started
        history.append({"C": c_value, "validation_roc_auc": auc, "threshold": threshold,
                        "fit_seconds": fit_seconds, "iterations": int(estimator.n_iter_)})
        print(f"C={c_value}: validation_AUC={auc:.6f}, iterations={estimator.n_iter_}, seconds={fit_seconds:.1f}", flush=True)
        if best is None or auc > best["validation_roc_auc"]:
            best = {
                "C": c_value,
                "validation_roc_auc": auc,
                "threshold": threshold,
                "coef": estimator.coef_[0].copy(),
                "intercept": float(estimator.intercept_[0]),
            }
    assert best is not None
    # Persist threshold from the identical single-margin implementation used at
    # inference, avoiding BLAS 1D/2D roundoff at score==threshold ties.
    canonical_parameters = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": best["coef"], "intercept": best["intercept"]}
    best["threshold"] = select_threshold(y_validation, _margins(features, validation_rows, canonical_parameters))
    model_path = output / "model.npz"
    np.savez(
        model_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=best["coef"],
        intercept=best["intercept"],
        threshold=best["threshold"],
    )
    binding = {
        "model_name": SRM_MODEL_NAME,
        "model_version": SRM_MODEL_VERSION,
        "source_view": source,
        "training_seed": int(seed),
        "manifest_sha256": sha256_file(manifest_path),
        "feature_binding": feature_info["binding"],
        "feature_sha256": feature_info["features_sha256"],
    }
    _write_json(output / "model.json", {
        "binding": binding,
        "checkpoint_sha256": sha256_file(model_path),
        "classifier": "train-only StandardScaler + LinearSVC",
        "score_type": "decision_margin_not_probability",
        "C": best["C"],
        "c_grid": candidates,
        "threshold": best["threshold"],
        "best_validation_roc_auc": best["validation_roc_auc"],
        "runtime_environment": runtime_environment(),
        "training_seconds": time.perf_counter() - started,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "estimated_solver_peak_bytes": estimated_peak,
    })
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    return model_path


def _margins(features: np.ndarray, rows: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    standardized = (
        features[rows].astype(np.float64) - parameters["mean"]
    ) / parameters["scale"]
    return standardized @ parameters["coef"] + float(parameters["intercept"])


def _raw_margin_rows(samples: pd.DataFrame, rows: np.ndarray, margins: np.ndarray) -> pd.DataFrame:
    selected = samples.iloc[rows].reset_index(drop=True).copy()
    if len(selected) != len(margins):
        raise ValueError("SRM margins do not align with the selected samples")
    selected["logit"] = margins
    selected["score"] = margins
    return selected[[
        "pair_id", "image_id", "patient_id", "split", "view_position",
        "class_name", "label", "logit", "score", "patch_index", "crop_top",
        "crop_left",
    ]]


def evaluate_srm_svm(
    manifest_path: str | Path,
    feature_dir: str | Path,
    run_dir: str | Path,
    output_stem: str | Path,
    *,
    config_path: str | Path,
    target_views: Iterable[str] = ("AP", "PA"),
    test_cohort: str | None = None,
    protocol_lock: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate fixed-margin threshold on source validation and both target tests."""
    run_dir = Path(run_dir)
    from cxr_steganalysis.config import load_config
    from cxr_steganalysis.protocol_lock import validate_test_access
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str})
    effective_config = run_dir / "effective_config.yaml"
    access_config = load_config(effective_config if effective_config.exists() else config_path)
    protocol_lock_hash = validate_test_access(access_config, frame, test_cohort, protocol_lock)
    metadata = _read_json(run_dir / "model.json")
    binding = metadata["binding"]
    samples, features, feature_info = _load_feature_cache(manifest_path, feature_dir)
    if binding["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("SRM classifier was fitted on a different manifest")
    if binding["feature_binding"] != feature_info["binding"]:
        raise ValueError("SRM classifier was fitted on a different feature protocol")
    if binding["feature_sha256"] != feature_info["features_sha256"]:
        raise ValueError("SRM classifier was fitted on a different feature matrix")
    model_path = run_dir / "model.npz"
    checkpoint_hash = sha256_file(model_path)
    if checkpoint_hash != metadata["checkpoint_sha256"]:
        raise ValueError("SRM classifier checkpoint failed its SHA-256 check")
    with np.load(model_path, allow_pickle=False) as archive:
        parameters = {key: archive[key] for key in archive.files}
    source = str(binding["source_view"])
    threshold = float(parameters["threshold"])
    common = {
        "source_view": source,
        "threshold": threshold,
        "checkpoint_path": model_path,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seed": int(binding["training_seed"]),
        "payload_bpp": float(samples["payload_bpp"].iloc[0]),
        "algorithm": str(samples["algorithm"].iloc[0]),
        "model_name": SRM_MODEL_NAME,
        "model_version": SRM_MODEL_VERSION,
        "score_type": "decision_margin",
    }
    validation_rows = _selected_rows(samples, "validation", source)
    validation_scores = _margins(features, validation_rows, parameters)
    validation_predictions = _finalize_predictions(
        _raw_margin_rows(samples, validation_rows, validation_scores),
        target_view=source,
        **common,
    )
    from cxr_steganalysis.config import load_config
    from cxr_steganalysis.experiment import manifest_identity
    config = access_config
    extra = {"protocol_id": config.get("protocol_id", "legacy_pilot"), "objective": "erm",
             "test_cohort": test_cohort or "all_enrolled_test",
             "crop_seed": feature_info["binding"]["crop_seed"], "patch_size": feature_info["binding"]["patch_size"],
             "normalization": "uint8_srm", "manifest_identity": manifest_identity(pd.read_csv(manifest_path, dtype={"patient_id": str, "seed": str}))}
    if protocol_lock_hash:
        extra["protocol_lock_sha256"] = protocol_lock_hash
    for key, value in extra.items():
        validation_predictions[key] = value
    results: list[dict[str, Any]] = []
    for target_view in target_views:
        started = time.perf_counter()
        target = str(target_view).upper()
        test_rows = _selected_rows(samples, "test", target, cohort=test_cohort)
        scores = _margins(features, test_rows, parameters)
        raw = _raw_margin_rows(samples, test_rows, scores)
        predictions = _finalize_predictions(raw, target_view=target, **common)
        for key, value in extra.items():
            predictions[key] = value
        labels = raw["label"].to_numpy(dtype=np.int64)
        metrics = binary_metrics(labels, scores, threshold)
        target_samples = samples.iloc[test_rows]
        results.append({
            "experiment_name": "srm_svm_comparison",
            "model_name": SRM_MODEL_NAME,
            "model_version": SRM_MODEL_VERSION,
            **extra,
            "train_view": source,
            "test_view": target,
            "stored_source_view": source,
            "seed": int(binding["training_seed"]),
            "evaluation_seed": int(binding["training_seed"]),
            "best_checkpoint_epoch": None,
            "dataset_size": {
                "patients": int(target_samples["patient_id"].nunique()),
                "pairs": int(target_samples["pair_id"].nunique()),
                "patches_per_class": int((target_samples["label"] == 0).sum()),
                "classification_samples": len(target_samples),
            },
            "validation_size": {
                "patients": int(
                    samples.iloc[validation_rows]["patient_id"].nunique()
                ),
                "pairs": int(samples.iloc[validation_rows]["pair_id"].nunique()),
                "patches_per_class": int(
                    (samples.iloc[validation_rows]["label"] == 0).sum()
                ),
                "classification_samples": len(validation_rows),
            },
            "validation_pairs_for_threshold": int(
                samples.iloc[validation_rows]["pair_id"].nunique()
            ),
            "payload_bpp": common["payload_bpp"],
            "algorithm": common["algorithm"],
            "checkpoint": str(model_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": common["config_sha256"],
            "manifest_sha256": common["manifest_sha256"],
            "provenance_scope": "training-bound-features-and-evaluation-time-config",
            "score_type": "SVM decision margin, not probability",
            "threshold": threshold,
            "metrics": metrics,
            "runtime_seconds": time.perf_counter() - started,
            "runtime_environment": runtime_environment(),
            "_validation_predictions": validation_predictions,
            "_test_predictions": predictions,
        })
    save_evaluation_results(results, output_stem)
    return results
