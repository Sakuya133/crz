"""Single-device trainer with AMP, local logs, early stopping, and resume."""

from __future__ import annotations

import csv
import json
import hashlib
import time
import resource
import signal
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cxr_steganalysis.config import serializable_config
from cxr_steganalysis.data.dataset import flatten_paired_batch
from cxr_steganalysis.models.factory import MODEL_VERSIONS
from cxr_steganalysis.reproducibility import capture_rng_state, restore_rng_state
from cxr_steganalysis.training.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from cxr_steganalysis.training.metrics import binary_metrics
from cxr_steganalysis.training.groupdro import GroupDRO


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        output_dir: str | Path,
        device: str = "auto",
        model_name: str | None = None,
        training_binding: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        digest = hashlib.sha256()
        for key, value in sorted(self.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.initial_model_sha256 = digest.hexdigest()
        inferred_name = "highpass" if model.__class__.__name__ == "HighPassResidualCNN" else None
        self.model_name = str(model_name or inferred_name or model.__class__.__name__).lower()
        self.model_version = MODEL_VERSIONS.get(self.model_name, model.__class__.__name__)
        self.training_binding = training_binding
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )
        self.binary_criterion = nn.BCEWithLogitsLoss()
        self.multiclass_criterion = nn.CrossEntropyLoss()
        self.use_amp = bool(config["mixed_precision"] and self.device.type == "cuda")
        self.scaler = _make_grad_scaler(self.use_amp)
        self.objective = str(config.get("training", {}).get("objective", "erm"))
        if self.objective not in {"erm", "groupdro"}:
            raise ValueError("Unknown objective")
        self.groupdro = GroupDRO(config.get("training", {}).get("groupdro_step_size", .01)).to(self.device)
        self.last_view_auc = {}
        self.optimizer_steps = 0
        self.amp_skipped_steps = 0
        self.stop_requested = False
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))

    def _loss_and_scores(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim == 1:
            loss = self.binary_criterion(logits, labels)
            # float32 scores: fp16 sigmoid under AMP creates ties in validation AUC,
            # which drives checkpoint selection.
            return loss, torch.sigmoid(logits.float())
        if logits.ndim == 2 and logits.shape[1] == 2:
            loss = self.multiclass_criterion(logits, labels.to(torch.long))
            return loss, torch.softmax(logits, dim=1)[:, 1]
        raise ValueError(
            "Model must return one binary logit per sample or two cover/stego logits; "
            f"got {tuple(logits.shape)}"
        )

    @staticmethod
    def _loader_generator_state(loader: DataLoader) -> torch.Tensor | None:
        generator = getattr(loader, "generator", None)
        return generator.get_state() if generator is not None else None

    @staticmethod
    def _restore_loader_generator(loader: DataLoader, state: Any) -> None:
        generator = getattr(loader, "generator", None)
        if generator is not None and state is not None:
            generator.set_state(state)

    def _assert_resume_compatible(self, checkpoint: dict[str, Any]) -> None:
        stored_name = checkpoint.get("model_name")
        if stored_name is None:
            # All format-v1 checkpoints created by this project were the custom baseline.
            stored_name = "highpass"
        if str(stored_name).lower() != self.model_name:
            raise ValueError(
                f"Checkpoint model {stored_name!r} is not compatible with "
                f"requested model {self.model_name!r}"
            )
        if checkpoint.get("model_version", self.model_version) != self.model_version:
            raise ValueError("Checkpoint architecture version differs")
        if checkpoint.get("objective", "erm") != self.objective:
            raise ValueError("Resume objective differs")
        stored_binding = checkpoint.get("training_binding")
        left_binding, right_binding = stored_binding, self.training_binding
        if stored_binding and self.training_binding and stored_binding.get("manifest_identity") == self.training_binding.get("manifest_identity") and stored_binding.get("manifest_identity"):
            left_binding = {k:v for k,v in stored_binding.items() if k != "manifest_sha256"}
            right_binding = {k:v for k,v in self.training_binding.items() if k != "manifest_sha256"}
        if stored_binding is not None and left_binding != right_binding:
            raise ValueError(
                "Resume binding mismatch: manifest, source view, data protocol, "
                "or training configuration changed"
            )
        stored_config = checkpoint.get("config", {})
        comparisons = [
            ("seed",), ("patch_size",), ("patches_per_image",), ("payload_bpp",),
            ("algorithm",), ("batch_size",), ("learning_rate",), ("weight_decay",),
            ("data", "normalize"), ("training", "train_view"),
            ("mixed_precision",), ("seeds",), ("training", "groupdro_step_size"),
            ("training", "selection_metric"), ("training", "mixed_pairs_per_view"),
        ]
        for path in comparisons:
            left: Any = stored_config
            right: Any = self.config
            try:
                for key in path:
                    left = left[key]
                    right = right[key]
            except (KeyError, TypeError):
                continue
            if left != right:
                dotted = ".".join(path)
                raise ValueError(
                    f"Resume config mismatch for {dotted}: checkpoint={left!r}, "
                    f"current={right!r}"
                )

    def _run_loader(
        self,
        loader: DataLoader,
        training: bool,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(training)
        losses: list[tuple[float, int]] = []
        all_labels: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_views = []
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for batch in loader:
                images, labels = flatten_paired_batch(batch)
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(images)
                    loss, scores = self._loss_and_scores(logits, labels)
                    if training and self.objective == "groupdro":
                        views = batch["metadata"]["view_position"]
                        groups = torch.tensor([0 if v == "AP" else 1 for v in views] * 2, device=self.device)
                        per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") if logits.ndim == 1 else F.cross_entropy(logits, labels.long(), reduction="none")
                        loss = self.groupdro(per_sample, groups)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training/validation loss; run is not valid")
                if training:
                    previous_scale = self.scaler.get_scale()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    skipped = self.use_amp and self.scaler.get_scale() < previous_scale
                    self.amp_skipped_steps += int(skipped)
                    self.optimizer_steps += int(not skipped)
                losses.append((float(loss.detach().cpu()), len(labels)))
                all_labels.append(labels.detach().cpu().numpy())
                all_scores.append(scores.detach().cpu().numpy())
                if "metadata" in batch:
                    all_views.extend(list(batch["metadata"]["view_position"]) * 2)
        if not losses:
            raise ValueError("DataLoader produced no batches")
        if all_views:
            labels_array, scores_array = np.concatenate(all_labels), np.concatenate(all_scores)
            self.last_view_auc = {v: binary_metrics(labels_array[np.array(all_views) == v], scores_array[np.array(all_views) == v])["roc_auc"] for v in sorted(set(all_views))}
        return (
            sum(loss*n for loss, n in losses) / sum(n for _, n in losses),
            np.concatenate(all_labels),
            np.concatenate(all_scores),
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        resume: str | Path | None = None,
        max_epochs: int | None = None,
    ) -> dict[str, Any]:
        start_epoch = 0
        best_metric = float("-inf")
        epochs_without_improvement = 0
        if resume is not None:
            # RNG ByteTensors must stay on CPU. Optimizer.load_state_dict moves
            # optimizer buffers to the parameter device itself.
            checkpoint = load_checkpoint(resume, "cpu")
            self._assert_resume_compatible(checkpoint)
            # Never fabricate initialization provenance for older checkpoints.
            self.initial_model_sha256 = checkpoint.get("initial_model_sha256")
            if not (self.checkpoint_dir / "best.pt").exists():
                raise FileNotFoundError("Resume requires the run's best.pt as well as last.pt; restore the complete run directory to preserve model selection")
            restore_training_state(
                checkpoint, self.model, self.optimizer, self.scheduler
            )
            if checkpoint.get("scaler_state") is not None:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
            if checkpoint.get("groupdro_state") is not None:
                self.groupdro.load_state_dict(checkpoint["groupdro_state"])
            self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
            self.amp_skipped_steps = int(checkpoint.get("amp_skipped_steps", 0))
            if checkpoint.get("rng_state") is not None:
                restore_rng_state(checkpoint["rng_state"])
            self._restore_loader_generator(
                train_loader, checkpoint.get("train_loader_generator_state")
            )
            self._restore_loader_generator(
                validation_loader, checkpoint.get("validation_loader_generator_state")
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_metric"])
            epochs_without_improvement = int(
                checkpoint.get("epochs_without_improvement", 0)
            )

        epochs = int(max_epochs if max_epochs is not None else self.config["epochs"])
        log_path = self.log_dir / "history.csv"
        if resume is None:
            existing = [
                path for path in (
                    self.checkpoint_dir / "best.pt",
                    self.checkpoint_dir / "last.pt",
                    log_path,
                    self.log_dir / "summary.json",
                ) if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite an existing training run. Choose a new "
                    f"output directory or resume its last checkpoint: {existing}"
                )
        fieldnames = [
            "epoch", "train_loss", "validation_loss", "train_roc_auc",
            "validation_roc_auc", "validation_balanced_accuracy", "learning_rate",
            "elapsed_seconds",
            "peak_vram_mb", "peak_rss_mb", "q_ap", "q_pa", "optimizer_updates",
            "validation_ap_auc", "validation_pa_auc", "selection_metric",
            "effective_optimizer_steps", "amp_skipped_steps",
        ]
        if start_epoch > 0 and log_path.exists():
            with log_path.open() as handle:
                fieldnames = next(csv.reader(handle))
        if start_epoch == 0 or not log_path.exists():
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, epochs):
            started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_loss, train_labels, train_scores = self._run_loader(train_loader, True)
            validation_loss, validation_labels, validation_scores = self._run_loader(
                validation_loader, False
            )
            train_metrics = binary_metrics(train_labels, train_scores)
            validation_metrics = binary_metrics(validation_labels, validation_scores)
            selection_metric = validation_metrics["roc_auc"]
            if self.config.get("training", {}).get("selection_metric") == "macro_view_auc":
                if set(self.last_view_auc) != {"AP", "PA"}:
                    raise ValueError("Mixed selection requires both validation views")
                selection_metric = float(np.mean(list(self.last_view_auc.values())))
            if np.isnan(selection_metric):
                selection_metric = validation_metrics["balanced_accuracy"]
            self.scheduler.step(validation_loss)
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_roc_auc": train_metrics["roc_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
                "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else 0,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "q_ap": float(self.groupdro.log_q.exp()[0]),
                "q_pa": float(self.groupdro.log_q.exp()[1]),
                "optimizer_updates": (epoch + 1) * len(train_loader),
                "effective_optimizer_steps": self.optimizer_steps,
                "amp_skipped_steps": self.amp_skipped_steps,
                "validation_ap_auc": self.last_view_auc.get("AP"),
                "validation_pa_auc": self.last_view_auc.get("PA"),
                "selection_metric": selection_metric,
            }
            with log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow({k: row[k] for k in fieldnames})
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"val_loss={validation_loss:.5f} val_auc={selection_metric:.4f}"
            )

            improved = selection_metric > best_metric
            if improved:
                best_metric = float(selection_metric)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            state = {
                "format_version": 2,
                "epoch": epoch,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_state": self.model.state_dict(),
                "initial_model_sha256": self.initial_model_sha256,
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_metric": best_metric,
                "config": serializable_config(self.config),
                "seed": int(self.config["seed"]),
                "epochs_without_improvement": epochs_without_improvement,
                "scaler_state": self.scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "train_loader_generator_state": self._loader_generator_state(train_loader),
                "validation_loader_generator_state": self._loader_generator_state(
                    validation_loader
                ),
                "training_binding": self.training_binding,
                "objective": self.objective,
                "groupdro_state": self.groupdro.state_dict(),
                "optimizer_steps": self.optimizer_steps,
                "amp_skipped_steps": self.amp_skipped_steps,
            }
            save_checkpoint(state, self.checkpoint_dir / "last.pt")
            if improved:
                save_checkpoint(state, self.checkpoint_dir / "best.pt")
            last_epoch = epoch
            if self.stop_requested:
                print("Stop requested: saved epoch-boundary checkpoint", flush=True)
                break
            if epochs_without_improvement >= int(self.config["early_stopping_patience"]):
                break

        summary = {
            "last_epoch": last_epoch,
            "best_metric": best_metric,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
            "last_checkpoint": str(self.checkpoint_dir / "last.pt"),
            "device": str(self.device),
            "mixed_precision_enabled": self.use_amp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "objective": self.objective,
            "selection_rule": self.config.get("training", {}).get("selection_metric", "source_validation_auc"),
            "budget_limited": last_epoch + 1 == epochs,
            "interrupted": self.stop_requested,
            "initial_model_sha256": self.initial_model_sha256,
            "effective_optimizer_steps": self.optimizer_steps,
            "amp_skipped_steps": self.amp_skipped_steps,
        }
        (self.log_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
