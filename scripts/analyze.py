#!/usr/bin/env python3
"""Predeclared fixed-target and objective comparisons, separately for each seed."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from _bootstrap import bootstrap
bootstrap()
from cxr_steganalysis.evaluation.artifacts import completed_results, result_key
from cxr_steganalysis.evaluation.cluster_bootstrap import patient_cluster_paired_bootstrap, save_bootstrap_results, load_prediction_file
from cxr_steganalysis.evaluation.mixed_bootstrap import mixed_patient_bootstrap
from cxr_steganalysis.training.checkpoint import load_checkpoint
from cxr_steganalysis.protocol_lock import scientific_settings
from cxr_steganalysis.lab import atomic_json
from cxr_steganalysis.provenance import sha256_file


def verify_control(left, right):
    checkpoints = [load_checkpoint(p / "checkpoints/last.pt", "cpu") for p in (left, right)]
    settings = [deepcopy(scientific_settings(c["config"])) for c in checkpoints]
    for s in settings: s["training"].pop("objective", None)
    bindings = [{k:v for k,v in c["training_binding"].items() if k != "objective"} for c in checkpoints]
    checks = dict(settings_except_objective=settings[0] == settings[1], same_images=bindings[0] == bindings[1],
                  same_initial_weights=bool(checkpoints[0].get("initial_model_sha256")) and checkpoints[0].get("initial_model_sha256") == checkpoints[1].get("initial_model_sha256"),
                  same_epochs=checkpoints[0]["epoch"] == checkpoints[1]["epoch"], same_updates=checkpoints[0]["optimizer_steps"] == checkpoints[1]["optimizer_steps"])
    if not all(checks.values()): raise ValueError(f"ERM/GroupDRO control failed: {checks}")
    return checks


def analyze(root, output, replicates=10000):
    root, output = Path(root), Path(output)
    items = completed_results(root)
    keyed = {result_key(i): i for i in items}
    if len(keyed) != len(items): raise ValueError("Duplicate protocol/method/seed/scenario")
    planned = []
    contexts = sorted({k[:3] for k in keyed})
    for context in contexts:
        methods = sorted({k[3] for k in keyed if k[:3] == context and k[5] != "ALL"})
        for model in methods:
            for target, other in (("AP", "PA"), ("PA", "AP")):
                planned.append((f"mismatch_{model}_{target}", context + (model, "erm", target, target), context + (model, "erm", other, target), "cross_view"))
        # highpass is fixed a priori, not selected after seeing the weakest cell.
        for model in methods:
            if model == "highpass": continue
            for source in ("AP", "PA"):
                for target in ("AP", "PA"):
                    planned.append((f"method_{model}_minus_highpass_{source}_{target}", context + ("highpass", "erm", source, target), context + (model, "erm", source, target), "method"))
        if any(k[:3] == context and k[5] == "ALL" for k in keyed):
            for target in ("AP", "PA"):
                planned.append((f"mitigation_groupdro_minus_erm_{target}", context + ("highpass", "erm", "ALL", target), context + ("highpass", "groupdro", "ALL", target), "method"))
    pending, complete = [], []
    def store_or_validate(result, stem):
        existing = stem.with_suffix(".json")
        if existing.exists():
            old = json.loads(existing.read_text())
            keys = ["matched_prediction_sha256", "mismatched_prediction_sha256", "replicates_requested", "comparison_mode"]
            if any(old[k] != result[k] for k in keys): raise ValueError(f"Statistics input/budget changed: {existing}; use --output elsewhere")
            if not stem.with_suffix(".csv").exists():
                raise ValueError(f"Incomplete statistics artifacts: {stem}; preserve and use another --output")
        else: save_bootstrap_results(result, stem)
        complete.append(str(existing.relative_to(output)))
    for name, left_key, right_key, mode in planned:
        suffix = f"bpp{left_key[1]:g}_seed{left_key[2]}".replace(".", "p")
        if left_key not in keyed or right_key not in keyed:
            pending.append(name + "_" + suffix); continue
        left, right = keyed[left_key], keyed[right_key]
        folder = output / left_key[0]
        stem = folder / (name + "_" + suffix)
        if left_key[5] == "ALL":
            atomic_json(folder / ("control_" + suffix + ".json"), verify_control(left["run"], right["run"]))
        hashes = [sha256_file(i["predictions"]) for i in (left, right)]
        if stem.with_suffix(".json").exists():
            old = json.loads(stem.with_suffix(".json").read_text())
            store_or_validate(dict(old, matched_prediction_sha256=hashes[0], mismatched_prediction_sha256=hashes[1], replicates_requested=replicates, comparison_mode=mode), stem)
            continue
        result = patient_cluster_paired_bootstrap(left["frame"], right["frame"], target_view=left_key[6], comparison_mode=mode, replicates=replicates, matched_prediction_sha256=hashes[0], mismatched_prediction_sha256=hashes[1])
        result.update(name=name, training_seed=left_key[2], protocol_id=left_key[0], payload_bpp=left_key[1])
        store_or_validate(result, stem)
    for context in contexts:
        left_key = context + ("highpass", "erm", "ALL", "AP")
        right_key = context + ("highpass", "groupdro", "ALL", "AP")
        if left_key not in keyed or right_key not in keyed: continue
        left, right = keyed[left_key], keyed[right_key]
        verify_control(left["run"], right["run"])
        hashes = [sha256_file(i["predictions"]) for i in (left, right)]
        results = mixed_patient_bootstrap(load_prediction_file(left["predictions"]), load_prediction_file(right["predictions"]), replicates=replicates)
        for r in results:
            name = "mitigation_groupdro_minus_erm_" + r["target_view"]
            r.update(name=name, training_seed=context[2], protocol_id=context[0], payload_bpp=context[1], matched_prediction_sha256=hashes[0], mismatched_prediction_sha256=hashes[1])
            suffix = f"bpp{context[1]:g}_seed{context[2]}".replace(".", "p")
            store_or_validate(r, output / context[0] / f"{name}_{suffix}")
    receipt = dict(completed=complete, pending=pending, replicates=replicates,
                   scope="patient sampling uncertainty for fixed models; training seeds never pooled as patients",
                   asymmetry="AP-vs-PA difference of mismatch effects not tested; separate CIs do not test interaction")
    atomic_json(output / "complete.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--replicates", type=int, default=10000)
    a = p.parse_args()
    analyze(a.run_root, a.output or a.run_root / "statistics", a.replicates)
