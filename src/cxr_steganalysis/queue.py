"""Serial lab queue: immutable plans, explicit execution, bounded time and resume.

Synthetic fixtures are the only automatic CPU CNN use. Failed/pending stages
never become completed merely because a directory or an old checkpoint exists.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import psutil
from cxr_steganalysis.config import resolve_config_path, serializable_config
from cxr_steganalysis.experiment import effective_seeds
from cxr_steganalysis.lab import atomic_json, write_config
from cxr_steganalysis.provenance import sha256_file

LOCK_DIRECTORY = Path(__file__).resolve().parents[2] / "outputs"


def make_plan(config, output, models=None, seeds=None, sources=None, feature_cache=None):
    output = Path(output).resolve()
    models = models or config["lab"]["models"]
    seeds = seeds or config["lab"]["training_seeds"]
    sources = sources or ["AP", "PA"]
    if not set(models) <= {"highpass", "srnet", "srm_svm"} or len(models) != len(set(models)):
        raise ValueError("Required verified comparators only: highpass, srnet, srm_svm (no duplicates)")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds would create duplicate experiments")
    if not set(sources) <= {"AP", "PA"} or len(sources) != len(set(sources)):
        raise ValueError("Source views must be AP/PA without duplicates")
    mixed = config["training"].get("mixed_pairs_per_view") is not None
    if mixed and models != ["highpass"]:
        raise ValueError("Controlled mitigation preset uses the custom highpass backbone only")
    if mixed and sources != ["AP", "PA"]:
        raise ValueError("Mixed-view training cannot filter its source views")
    protocol = "B" if mixed else "A"
    stages, configs, runs = [], {}, []
    cache = Path(feature_cache).resolve() if feature_cache else output / "cache/srm"
    base_path = output / "configs/data.yaml"
    configs[base_path] = deepcopy(config)
    py = sys.executable
    def add(name, kind, markers, args, **fields):
        stages.append(dict(id=name, kind=kind, markers=[str(m) for m in markers], command=[py, "-u", *map(str, args)], **fields))
    if "srm_svm" in models:
        add("srm_features", "features", [cache / "complete.json"], ["scripts/extract_srm_features.py", "--config", base_path, "--output-dir", cache, "--workers", config.get("srm_workers", 2)])
    for model in models:
        # Extraction is deterministic; solver shuffle is fixed to one declared seed.
        # Repeating the same convex classifier three times is not CNN seed evidence.
        model_seeds = [int(config["lab"].get("svm_seed", 1337))] if model == "srm_svm" else seeds
        for seed in model_seeds:
            for source in (["ALL"] if mixed else sources):
                for objective in (["erm", "groupdro"] if mixed else ["erm"]):
                    name = f"{protocol}_{model}_{objective}_{source}_bpp{config['payload_bpp']:g}_seed{seed}"
                    run = output / name
                    cfg = deepcopy(config)
                    cfg["seed"] = seed
                    cfg["seeds"] = effective_seeds(config)
                    cfg["seeds"]["training"] = seed
                    cfg["training"].update(train_view=source, objective=objective)
                    cfg["model"]["name"] = model if model != "srm_svm" else "highpass"
                    cfg["paths"]["output_dir"] = str(run)
                    path = output / "configs" / (name + ".yaml")
                    configs[path] = cfg
                    runs.append(dict(id=name, model=model, source=source, objective=objective, seed=seed, config=str(path), directory=str(run)))
                    if model == "srm_svm":
                        args = ["scripts/train_srm_baseline.py", "--config", path, "--feature-dir", cache, "--train-view", source, "--seed", seed, "--c-grid", *config["lab"].get("c_grid", [.01, .1, 1]), "--output-dir", run]
                        add(name + "_train", "svm", [run / "model.json", run / "model.npz", run / "effective_config.yaml"], args, run_dir=str(run))
                    else:
                        add(name + "_train", "cnn", [run / "logs/summary.json", run / "checkpoints/best.pt", run / "checkpoints/last.pt"], ["scripts/train_baseline.py", "--config", path, "--output-dir", run], run_dir=str(run))
    full = config.get("dataset_profile") == "full_all_eligible"
    lock = output / "confirmatory_protocol_lock.json"
    if full:
        add("freeze_confirmatory", "freeze", [lock], ["scripts/freeze_protocol.py", "--run-configs", *[Path(r["directory"]) / "effective_config.yaml" for r in runs], "--manifest", resolve_config_path(config, "pair_manifest"), "--output", lock])
    # Finish all predeclared training/validation before opening any target tests.
    for r in runs:
        run = Path(r["directory"])
        stem = run / "evaluation/cross_view"
        common = ["--config", r["config"], "--output", stem]
        if full:
            common += ["--test-cohort", config["lab"]["cohort"], "--protocol-lock", lock]
        args = ["scripts/evaluate_srm_baseline.py", "--run-dir", run, "--feature-dir", cache, *common] if r["model"] == "srm_svm" else ["scripts/evaluate_baseline.py", "--checkpoint", run / "checkpoints/best.pt", *common]
        markers = [stem.with_suffix(".json"), stem.with_suffix(".csv"), stem.with_name(stem.name + "_predictions.csv"), stem.with_name(stem.name + "_validation_predictions.csv")]
        add(r["id"] + "_evaluate", "evaluation", markers, args)
    add("analysis", "analysis", [output / "statistics/complete.json"], ["scripts/analyze.py", "--run-root", output, "--replicates", config["lab"].get("bootstrap_replicates", 10000)])
    add("report", "report", [output / "report/README.md", output / "report/metrics.csv"], ["scripts/report.py", "--run-root", output, "--output", output / "report"])
    return dict(protocol=protocol, dataset_profile=config["dataset_profile"], runs=runs, stages=stages, output=str(output)), configs


def plan_identity(plan, configs):
    return hashlib.sha256(json.dumps(dict(plan=plan, configs={str(p): serializable_config(c) for p, c in configs.items()}), sort_keys=True).encode()).hexdigest()


def status(root):
    path = Path(root) / "queue.json"
    if not path.exists():
        print("No queue has been executed."); return
    state = json.loads(path.read_text())
    alive = False
    try:
        process = psutil.Process(state["runner_pid"])
        alive = abs(process.create_time() - state["runner_created"]) < 1
    except psutil.Error:
        pass
    print(f"State={state['status']} runner PID={state['runner_pid']} alive={alive}; update={state['updated_utc']}")
    for s in state["stages"]:
        actual = "interrupted?" if s["status"] == "running" and not alive else s["status"]
        print(f"{actual:13s} {s['id']} log={s.get('log', '-')}")


def execute(plan, configs, root, *, resume=False, budget_hours=8, validate_inputs=True):
    if budget_hours <= 0:
        raise ValueError("Budget must be positive")
    root = Path(root).resolve()
    source_root = Path(__file__).resolve().parents[2]
    scientific = next(iter(configs.values()))
    if validate_inputs:
        import torch
        from cxr_steganalysis.data.portable import check_data
        if plan["dataset_profile"] != "synthetic_smoke" and any(s["kind"] == "cnn" for s in plan["stages"]) and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no long NIH CNN fallback to CPU")
        check_data(scientific, verify_hashes=False)
    locks = LOCK_DIRECTORY
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / ".one_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "queue.json"
    signature = plan_identity(plan, configs)
    old = json.loads(state_path.read_text()) if state_path.exists() else None
    if old and not resume:
        raise FileExistsError("Queue exists; use --resume, never overwrite an experiment")
    if old and old["plan_identity"] != signature:
        raise ValueError("Models/seeds/config/paths changed; choose a new output directory")
    if not old and root.exists() and any(root.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to adopt unknown artifacts")
    root.mkdir(parents=True, exist_ok=True)
    for path, cfg in configs.items(): write_config(path, cfg)
    atomic_json(root / "plan.json", plan)
    state = old or dict(plan_identity=signature, stages=[dict(s, status="pending") for s in plan["stages"]])
    state.update(runner_pid=os.getpid(), runner_created=psutil.Process().create_time(), status="running")
    def persist():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)
    stop = [False]
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGINT, signal.SIGTERM)}
    deadline = time.monotonic() + budget_hours * 3600
    persist()
    try:
        for stage in state["stages"]:
            markers = [Path(p) for p in stage["markers"]]
            if stage["status"] == "completed":
                if not all(p.is_file() and sha256_file(p) == stage["artifacts"][str(p)] for p in markers):
                    raise ValueError(f"Completed artifact missing or modified: {stage['id']}")
                continue
            if stop[0] or time.monotonic() >= deadline:
                state["status"] = "pending"; persist(); return 124
            command = list(stage["command"])
            if stage["kind"] == "svm":
                previous_run = Path(stage["run_dir"])
                if previous_run.exists() and any(previous_run.iterdir()):
                    archive = root / "interrupted_artifacts" / (stage["id"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    previous_run.replace(archive)
                    stage["preserved_incomplete_fit"] = str(archive)
            # An interrupted evaluator may have written only some of its outputs.
            # Keep them for audit and regenerate as one unit; never trust existence.
            if stage["kind"] == "evaluation" and any(p.exists() for p in markers):
                archive = root / "interrupted_artifacts" / stage["id"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive.mkdir(parents=True)
                for p in markers:
                    if p.exists(): p.replace(archive / p.name)
            if stage["kind"] == "cnn":
                checkpoint = Path(stage["run_dir"]) / "checkpoints/last.pt"
                if checkpoint.exists(): command += ["--resume", str(checkpoint)]
            log = root / "logs" / (stage["id"] + ".log")
            log.parent.mkdir(exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS=str(scientific.get("cpu_threads", 4)), OPENBLAS_NUM_THREADS=str(scientific.get("cpu_threads", 4)), MKL_NUM_THREADS=str(scientific.get("cpu_threads", 4)), CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
            env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
            started = time.monotonic()
            interrupted = False
            peak_rss = 0
            with log.open("a") as handle:
                handle.write("\nCOMMAND " + json.dumps(command) + "\n"); handle.flush()
                process = subprocess.Popen(command, cwd=source_root, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                stage.update(status="running", pid=process.pid, command_executed=command, log=str(log), started_utc=datetime.now(timezone.utc).isoformat())
                persist(); print(f"Started {stage['id']} PID={process.pid} log={log}", flush=True)
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                        peak_rss = max(peak_rss, rss)
                    except psutil.Error: pass
                    if stop[0] or time.monotonic() >= deadline:
                        interrupted = True
                        if stage["kind"] == "cnn": process.terminate()
                        else: os.killpg(process.pid, signal.SIGTERM)
                        # Epoch checkpoints are recoverable; a power loss can lose
                        # the current epoch only. No mid-epoch exact-resume claim.
                        until = time.monotonic() + 180
                        while process.poll() is None and time.monotonic() < until: time.sleep(.5)
                        if process.poll() is None: os.killpg(process.pid, signal.SIGKILL)
                        break
                    time.sleep(.25)
                code = process.wait()
            complete = code == 0 and all(p.is_file() for p in markers)
            if complete and stage["kind"] == "cnn":
                complete = not json.loads(markers[0].read_text()).get("interrupted", False)
            stage.update(status="pending" if interrupted else "completed" if complete else "failed", returncode=code,
                         wall_seconds=time.monotonic()-started, peak_tree_rss_bytes=peak_rss,
                         rss_note="sum of sampled process RSS; shared pages may be counted twice", ended_utc=datetime.now(timezone.utc).isoformat())
            if stage["status"] == "completed": stage["artifacts"] = {str(p): sha256_file(p) for p in markers}
            else:
                state["status"] = stage["status"]; persist(); return 124 if interrupted else 1
            persist()
        state["status"] = "completed"; persist(); return 0
    except Exception as error:
        state.update(status="failed", error=str(error)); persist(); raise
    finally:
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
        lock.close()
