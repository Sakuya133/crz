"""Find results relative to their run directory, including after lab transfer."""
import json
from pathlib import Path
from cxr_steganalysis.evaluation.cluster_bootstrap import load_prediction_file


def completed_results(root):
    results = []
    for path in sorted(Path(root).glob("*/evaluation/cross_view.json")):
        stem = path.with_suffix("")
        predictions = stem.with_name(stem.name + "_predictions.csv")
        validation = stem.with_name(stem.name + "_validation_predictions.csv")
        if not predictions.exists() or not validation.exists():
            raise FileNotFoundError(f"Incomplete evaluation: {path}")
        frame = load_prediction_file(predictions)
        for record in json.loads(path.read_text()):
            results.append(dict(record=record, path=path, run=path.parent.parent,
                                predictions=predictions, validation=validation,
                                frame=frame[frame.target_view == record["test_view"]].copy()))
    return results


def result_key(item):
    r = item["record"]
    return (r["protocol_id"], float(r["payload_bpp"]), int(r["seed"]), r["model_name"], r.get("objective", "erm"), r["train_view"], r["test_view"])
