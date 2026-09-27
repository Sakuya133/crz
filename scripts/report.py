#!/usr/bin/env python3
"""Rebuild aggregate tables/figures from actual predictions, never fabricated NIH numbers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from _bootstrap import bootstrap
bootstrap()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from cxr_steganalysis.evaluation.artifacts import completed_results, result_key
from cxr_steganalysis.evaluation.cross_view import validate_prediction_frame
from cxr_steganalysis.evaluation.cluster_bootstrap import load_prediction_file
from cxr_steganalysis.training.metrics import binary_metrics, select_threshold
from cxr_steganalysis.provenance import sha256_file
from cxr_steganalysis.lab import atomic_json

METRICS = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "false_positive_rate"]


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


def build_report(root, output, statistics=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    items = completed_results(root)
    if len({result_key(i) for i in items}) != len(items):
        raise ValueError("Duplicate scenario/seed; select a single experiment collection")
    rows, inputs, resources = [], {}, []
    for item in items:
        r, frame = item["record"], item["frame"]
        validate_prediction_frame(frame, expected_split="test", expected_target_view=r["test_view"])
        validation = load_prediction_file(item["validation"])
        validate_prediction_frame(validation, expected_split="validation")
        if set(validation.source_view) != {r["train_view"]}:
            raise ValueError("Validation source differs")
        if r["train_view"] != "ALL" and set(validation.view_position) != {r["train_view"]}:
            raise ValueError("Threshold must use source validation only")
        threshold = float(frame.threshold.iloc[0])
        if not np.isclose(select_threshold(validation.label.to_numpy(), validation.score.to_numpy()), threshold, atol=1e-12, rtol=1e-12):
            raise ValueError("Threshold cannot be reproduced from validation")
        if set(frame.patient_id) & set(validation.patient_id):
            raise ValueError("Test/validation patient overlap in prediction artifacts")
        m = binary_metrics(frame.label.to_numpy(), frame.score.to_numpy(), threshold)
        for key in METRICS:
            if not np.isclose(m[key], r["metrics"][key], atol=1e-12, rtol=0):
                raise ValueError(f"Metrics differ from predictions: {item['path']} {key}")
        if m["confusion_matrix"] != r["metrics"]["confusion_matrix"]:
            raise ValueError("Confusion matrix differs")
        row = dict(protocol=r["protocol_id"], payload=r["payload_bpp"], seed=r["seed"], method=r["model_name"],
                   objective=r.get("objective", "erm"), source=r["train_view"], target=r["test_view"],
                   cohort=r.get("test_cohort", "all_enrolled_test"), pairs=frame.pair_id.nunique(), patients=frame.patient_id.nunique(),
                   samples=len(frame), threshold=threshold, score_type=frame.score_type.iloc[0], **{k:m[k] for k in METRICS},
                   confusion_matrix=json.dumps(m["confusion_matrix"]), prediction_sha256=sha256_file(item["predictions"]))
        rows.append(row)
        for p in (item["path"], item["predictions"], item["validation"]): inputs[str(p.relative_to(root))] = sha256_file(p)
        resources.append(dict(run=item["run"].name, operation="inference_" + r["test_view"], seconds=r.get("runtime_seconds"),
                              scope="cached feature decision margin; excludes SRM extraction" if r["model_name"] == "srm_svm" else "PNG load + paired crop + CNN inference; see per-run JSON"))
    columns = ["protocol", "payload", "seed", "method", "objective", "source", "target", "cohort", "pairs", "patients", "samples", "threshold", "score_type", *METRICS, "confusion_matrix", "prediction_sha256"]
    table = pd.DataFrame(rows, columns=columns)
    protocols = sorted(table.protocol.unique())
    synthetic = bool(protocols) and all("synthetic" in p.lower() for p in protocols)
    if any("synthetic" in p.lower() for p in protocols) and not synthetic:
        raise ValueError("Synthetic and NIH results must have separate report roots")
    label = "SYNTHETIC SMOKE — NOT NIH" if synthetic else "NIH empirical experiment" if len(rows) else "UNAVAILABLE — no completed evaluation"
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    captions = []
    plt.rcParams.update({"font.size":10, "pdf.fonttype":42, "ps.fonttype":42})
    def save(fig, name, caption):
        fig.savefig(figures / (name + ".pdf"), bbox_inches="tight")
        fig.savefig(figures / (name + ".png"), dpi=300, bbox_inches="tight")
        plt.close(fig); captions.append(f"- `{name}.pdf` / `.png`: {caption}")
    table.to_csv(output / "metrics.csv", index=False)
    public = table[["method", "objective", "source", "target", "seed", "pairs", "patients", *METRICS]]
    (output / "comparison.tex").write_text(public.to_latex(index=False, float_format="%.4f", escape=True))
    if len(table):
        group_keys = ["protocol", "payload", "method", "objective", "source", "target", "cohort"]
        table.groupby(group_keys)[METRICS].agg(["mean", "std", "count"]).to_csv(output / "training_seed_variation.csv")
        for key, sub in table.groupby(["protocol", "payload", "seed"]):
            prefix = slug("_".join(map(str, key)))
            matches = [i for i in items if (i["record"]["protocol_id"], i["record"]["payload_bpp"], i["record"]["seed"]) == key]
            for target in ("AP", "PA"):
                fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
                for item in matches:
                    r, frame = item["record"], item["frame"]
                    if r["test_view"] != target: continue
                    fpr, tpr, _ = roc_curve(frame.label, frame.score)
                    ax.plot(fpr, tpr, label=f"{r['model_name']} {r.get('objective','erm')} {r['train_view']}→{target} ({r['metrics']['roc_auc']:.3f})")
                ax.plot([0,1], [0,1], ":", color="grey")
                sample = sub[sub.target == target]
                n = "; ".join(f"{p} pairs/{q} patients" for p,q in sample[["pairs","patients"]].drop_duplicates().itertuples(index=False, name=None))
                ax.set(xlabel="False-positive rate", ylabel="Sensitivity", xlim=(0,1), ylim=(0,1), title=f"{label}\nTarget {target}; seed {key[2]}; {n}")
                ax.legend(fontsize=8, loc="lower right")
                save(fig, f"{prefix}_roc_{target}", "AUC in legend; same target, fixed fitted seed. Curves do not recalibrate the decision threshold.")
            for (method, objective), model in sub.groupby(["method", "objective"]):
                if set(model.source) == {"ALL"}: continue
                fig, axes = plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
                for ax, metric in zip(axes, ("roc_auc", "balanced_accuracy")):
                    matrix = model.pivot(index="source", columns="target", values=metric).reindex(index=["AP","PA"],columns=["AP","PA"])
                    ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
                    for y in range(2):
                        for x in range(2):
                            v = matrix.iloc[y,x]
                            ax.text(x,y,"pending" if pd.isna(v) else f"{v:.4f}",ha="center",va="center",color="white" if pd.isna(v) or v<.6 else "black")
                    ax.set(xticks=[0,1],xticklabels=["AP","PA"],yticks=[0,1],yticklabels=["AP","PA"],xlabel="Target test view",ylabel="Training view",title=metric)
                fig.suptitle(f"{label}\n{method}, training seed {key[2]}")
                save(fig,f"{prefix}_{method}_matrix","Missing cells are pending, not zero. Common colour range [0,1].")
            fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
            for ax, target in zip(axes, ("AP","PA")):
                cell = sub[sub.target == target]
                names = cell.method + " / " + cell.objective + " / " + cell.source + "→" + target
                ax.scatter(range(len(cell)),cell.roc_auc,label="ROC-AUC",marker="o")
                ax.scatter(range(len(cell)),cell.balanced_accuracy,label="BA",marker="x")
                ax.set(xticks=range(len(cell)),xticklabels=names,ylim=(0,1),ylabel="Metric",title=f"Target {target}; seed {key[2]}")
                ax.tick_params(axis="x",labelrotation=55,labelsize=8);ax.legend()
            fig.suptitle(label)
            save(fig,f"{prefix}_methods","All completed source/target cells, not post-hoc best cases; no score pooling across models.")
    # Learning history and measured training resources (no historical timings invented).
    for run in sorted({i["run"] for i in items} | {p.parent.parent for p in root.glob("*/logs/history.csv")}):
        history = run / "logs/history.csv"
        summary = run / "logs/summary.json"
        if history.exists():
            h = pd.read_csv(history)
            resources.append(dict(run=run.name, operation="training_epoch_loops", seconds=float(h.elapsed_seconds.sum()),
                                  peak_vram_mb=float(h.peak_vram_mb.max()), peak_process_rss_mb=float(h.peak_rss_mb.max()),
                                  scope="Sum of measured epoch loop wall times; excludes startup/checkpoint serialization; process RSS high-water mark"))
            fig, axes = plt.subplots(1,2,figsize=(9,3.5),layout="constrained")
            for name in ("train_loss", "validation_loss"):
                if name in h: axes[0].plot(h.epoch+1,h[name],label=name)
            for name in ("validation_roc_auc", "validation_ap_auc", "validation_pa_auc", "selection_metric"):
                if name in h: axes[1].plot(h.epoch+1,h[name],label=name)
            axes[0].set(xlabel="Epoch",ylabel="Loss");axes[1].set(xlabel="Epoch",ylabel="Validation AUC",ylim=(0,1))
            for ax in axes:
                ax.legend(fontsize=8)
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            fig.suptitle(f"{label}\n{run.name}",fontsize=9)
            save(fig,slug(run.name)+"_learning","Checkpoint selected on source validation AUC (mixed: macro-view AUC); test never used for selection.")
        training = summary if summary.exists() else run / "model.json"
        if training.exists():
            m = json.loads(training.read_text())
            resources.append(dict(run=run.name,operation="training",seconds=m.get("training_seconds",m.get("runtime_seconds")),peak_vram_mb=m.get("peak_cuda_memory_mb"),peak_process_rss_mb=m.get("peak_process_rss_mb"),scope="See source summary; solver training excludes feature extraction"))
    state = root / "queue.json"
    status_rows = []
    if state.exists():
        status_rows = json.loads(state.read_text())["stages"]
        for s in status_rows:
            resources.append(dict(run=s["id"],operation="queue_stage",seconds=s.get("wall_seconds"),peak_tree_rss_bytes=s.get("peak_tree_rss_bytes"),scope="Wall time includes startup/I/O; summed sampled RSS can double-count shared pages"))
    pd.DataFrame(resources).to_csv(output / "resources.csv",index=False)
    pd.DataFrame([{k:s.get(k) for k in ("id","status","returncode","wall_seconds","log")} for s in status_rows]).to_csv(output / "run_status.csv",index=False)
    deltas = []
    for path in sorted(Path(statistics or root / "statistics").glob("**/*.json")):
        r = json.loads(path.read_text())
        if "estimates" not in r: continue
        for metric, estimates in r["estimates"].items():
            delta = estimates["delta"]; base = estimates["matched"]["point_estimate"]
            deltas.append(dict(name=r.get("name",path.stem),protocol=r.get("protocol_id"),seed=r.get("training_seed"),target=r["target_view"],metric=metric,
                               delta=delta["point_estimate"],ci_low=delta["lower_2_5"],ci_high=delta["upper_97_5"],
                               delta_BA_pp=100*delta["point_estimate"] if metric == "balanced_accuracy" else None,
                               relative_BA_percent=100*delta["point_estimate"]/base if metric == "balanced_accuracy" and base else None,
                               patients=r["patients"],pairs=r["pairs"],replicates=r["replicates_valid"],definition=r["delta_definition"]))
    d = pd.DataFrame(deltas)
    d.to_csv(output / "deltas.csv",index=False)
    (output / "deltas.tex").write_text(d.to_latex(index=False,float_format="%.4f",escape=True))
    if len(d):
        for (protocol,seed), group in d.groupby(["protocol","seed"]):
            for metric in ("roc_auc","balanced_accuracy"):
                sub = group[group.metric == metric].reset_index(drop=True)
                fig,ax = plt.subplots(figsize=(9,max(3,.35*len(sub)+1.8)),layout="constrained")
                factor = 100 if metric == "balanced_accuracy" else 1
                for y,r in sub.iterrows():
                    ax.plot([factor*r.ci_low,factor*r.ci_high],[y,y],color="C0")
                    ax.plot(factor*r.delta,y,"o",color="C0")
                ax.axvline(0,linestyle=":",color="grey")
                ax.set(yticks=range(len(sub)),yticklabels=sub.name+" / "+sub.target,xlabel="Δ BA (percentage points)" if factor==100 else "Δ ROC-AUC",title=f"{label}\n95% paired patient-cluster percentile CI; training seed {seed}")
                ax.tick_params(axis="y",labelsize=8)
                save(fig,slug(protocol)+f"_seed{seed}_{metric}_deltas","Mismatch delta = mismatched − matched. Method/mitigation delta = candidate − baseline. Fixed thresholds; patient uncertainty only. AP/PA CI contrast is not an asymmetry test.")
    mitigation = []
    if len(table):
        for key, sub in table[table.source == "ALL"].groupby(["protocol","payload","seed","method","objective"]):
            if set(sub.target) != {"AP","PA"}: continue
            for aggregate in ("macro","worst_view"):
                mitigation.append(dict(zip(["protocol","payload","seed","method","objective"],key), target=aggregate,
                    **{metric:float(sub[metric].mean() if aggregate == "macro" else sub[metric].max() if metric == "false_positive_rate" else sub[metric].min()) for metric in METRICS}))
    mixed = pd.DataFrame(mitigation)
    mixed.to_csv(output / "mitigation.csv",index=False)
    if len(mixed):
        for (protocol,seed), sub in mixed.groupby(["protocol","seed"]):
            fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
            for metric,marker in (("roc_auc","o"),("balanced_accuracy","x")):
                ax.scatter(range(len(sub)),sub[metric],label=metric,marker=marker)
            ax.set(xticks=range(len(sub)),xticklabels=sub.objective+" / "+sub.target,ylim=(0,1),ylabel="Metric",title=f"{label}\nMixed-view mitigation; seed {seed}")
            ax.legend();save(fig,slug(protocol)+f"_seed{seed}_mitigation","Both objectives see AP and PA. Macro AUC averages per-view AUC, not pooled raw scores; worst view computed per metric.")
    pending = [s["id"] for s in status_rows if s["status"] != "completed"]
    limitations = "Patient bootstrap CI does not include training-seed variability. Seed SD (ddof=1) is separate; one seed has unavailable SD. CI crossing zero is not proof of equivalence. Differences between AP and PA effects are descriptive: no interaction/asymmetry significance test. AUC measures ranking; BA/FPR/sensitivity use the frozen source-validation threshold. No target-test threshold retuning. Negative results remain included."
    (figures / "README.md").write_text("# Figure captions\n\n"+label+"\n\n"+"\n".join(captions)+"\n")
    markdown_table = public.to_csv(index=False) if len(public) else "unavailable\n"
    (output / "mentor_progress_id.md").write_text(f"# Progres eksperimen empiris\n\nStatus: {label}.\n\nFokus: pengaruh sumber AP/PA terhadap deteksi cover/stego, bukan diagnosis atau klasifikasi view.\n\nSelesai: {len(rows)} skenario evaluasi yang metriknya dihitung ulang dari prediksi. Tahap belum selesai: {len(pending)} (lihat run_status.csv).\n\nPembanding: HighPassResidualCNN custom; SRNet adaptasi Boroumand et al.; Full SRM 34.671 + train-only StandardScaler + LinearSVC, bukan FLD. Mixed ERM/GroupDRO hanya jika ada run Protokol B.\n\nTabel aktual (AUC/BA/sensitivity/specificity/FPR):\n\n```csv\n{markdown_table}```\n\nPerubahan absolut AUC, BA dalam poin persentase dan BA relatif ada pada deltas.csv. Angka hanya tersedia bila dua prediksi setara telah selesai. Grafik dan caption: figures/README.md.\n\nTidak ada klaim mismatch selalu merugikan, keunggulan, sebab-akibat, atau algoritme baru. Full NIH memerlukan PNG+metadata+official lists lengkap, patient assignments terkunci, serta test tambahan yang belum terpapar pilot.\n\n{limitations}\n")
    (output / "README.md").write_text(f"# {label}\n\n{len(rows)} completed evaluation cells. Open mentor_progress_id.md, metrics.csv, comparison.tex, figures/README.md, deltas.csv, training_seed_variation.csv and resources.csv. Unavailable results are not imputed as zero.\n\nRebuild: `python scripts/report.py --run-root {root} --output {output}` (no training). The report may be regenerated; original predictions/checkpoints are never overwritten.\n\n{limitations}\n")
    (output / "results.tex").write_text("% Generated from actual prediction files; no NIH claims for synthetic smoke.\n" + ("\\paragraph{Synthetic verification only.} These results verify software, not NIH detection performance.\n" if synthetic else "") + "\\input{comparison.tex}\n\\input{deltas.tex}\n% Interpret only comparable target cells; discuss budget and seed variability.\n")
    atomic_json(output / "provenance.json", dict(label=label, input_sha256=inputs, cells=len(rows), synthetic=synthetic))
    print(f"Report regenerated: {output}; {len(rows)} actual cells; {label}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--statistics",type=Path)
    a = p.parse_args(); build_report(a.run_root,a.output,a.statistics)
