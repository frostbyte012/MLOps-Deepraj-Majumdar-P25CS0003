"""
nexusvoice/src/mlflow_tracker.py
─────────────────────────────────────────────────────────────
MLflow experiment tracking for NexusVoice pipeline.

Logs every run with:
  - Parameters: model names, thresholds, whisper model size
  - Metrics:    latencies, success rates, fallback rates
  - Artifacts:  generated code, transcription text
  - Tags:       fallback reason, model used, pipeline stage
"""

import os
import time
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "nexusvoice")


@dataclass
class RunMetadata:
    """Snapshot of one complete pipeline run for MLflow logging."""
    # Input
    voice_command: str
    whisper_model: str

    # Gemini
    gemini_model: str
    gemini_latency: float
    gemini_success: bool
    gemini_exceeded_threshold: bool

    # Fallback
    fallback_triggered: bool
    fallback_reason: str | None
    local_model: str
    local_latency: float
    local_success: bool

    # Output
    final_code: str
    model_used: str
    total_latency: float
    pipeline_success: bool

    # Threshold config
    latency_threshold: float


class MLflowTracker:
    """
    Wraps MLflow to track NexusVoice pipeline runs.

    Each call to track_run() creates one MLflow run with:
    - params:    static config (model names, threshold)
    - metrics:   dynamic values (latencies, success flags)
    - artifacts: generated code saved as .py file
    - tags:      routing decisions for easy filtering in UI
    """

    def __init__(self):
        self._mlflow = None
        self._experiment_id = None
        self._connected = False

    def _init(self):
        """Lazy-init MLflow client. Fails gracefully if server is down."""
        if self._mlflow is not None:
            return

        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    MLFLOW_EXPERIMENT,
                    tags={"project": "nexusvoice", "stage": "2"},
                )
            else:
                self._experiment_id = experiment.experiment_id

            self._connected = True
            console.print(
                f"[green]✓ MLflow connected[/green] "
                f"[dim]→ {MLFLOW_TRACKING_URI} | "
                f"experiment: {MLFLOW_EXPERIMENT}[/dim]"
            )
            logger.info(f"MLflow initialized: {MLFLOW_TRACKING_URI}")

        except ImportError:
            console.print("[yellow]⚠ MLflow not installed — tracking disabled[/yellow]")
            logger.warning("MLflow not installed")
        except Exception as e:
            console.print(f"[yellow]⚠ MLflow unavailable ({e}) — tracking disabled[/yellow]")
            logger.warning(f"MLflow connection failed: {e}")

    def track_run(self, meta: RunMetadata) -> str | None:
        """
        Log a complete pipeline run to MLflow.

        Args:
            meta: RunMetadata with all pipeline outputs.

        Returns:
            MLflow run_id string, or None if tracking failed.
        """
        self._init()

        if not self._connected or self._mlflow is None:
            logger.warning("MLflow not available — skipping tracking")
            return None

        mlflow = self._mlflow

        try:
            with mlflow.start_run(experiment_id=self._experiment_id) as run:
                run_id = run.info.run_id

                # ── Parameters (static config) ─────────────────
                mlflow.log_params({
                    "whisper_model":       meta.whisper_model,
                    "gemini_model":        meta.gemini_model,
                    "local_model":         meta.local_model,
                    "latency_threshold_s": meta.latency_threshold,
                    "pipeline_stage":      "2",
                })

                # ── Metrics (dynamic values) ───────────────────
                mlflow.log_metrics({
                    # Latencies
                    "gemini_latency_s":    round(meta.gemini_latency, 3),
                    "local_latency_s":     round(meta.local_latency, 3),
                    "total_latency_s":     round(meta.total_latency, 3),

                    # Binary outcomes (1.0 = True, 0.0 = False)
                    "gemini_success":      float(meta.gemini_success),
                    "local_success":       float(meta.local_success),
                    "pipeline_success":    float(meta.pipeline_success),
                    "fallback_triggered":  float(meta.fallback_triggered),
                    "threshold_exceeded":  float(meta.gemini_exceeded_threshold),

                    # Code quality proxy — output length
                    "generated_code_chars": float(len(meta.final_code)),
                })

                # ── Tags (for filtering in MLflow UI) ─────────
                mlflow.set_tags({
                    "model_used":      meta.model_used,
                    "fallback_reason": meta.fallback_reason or "none",
                    "fallback":        str(meta.fallback_triggered),
                    "voice_command":   meta.voice_command[:100],  # truncate long commands
                })

                # ── Artifacts (files) ──────────────────────────
                # Save generated code as a .py file
                if meta.final_code:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".py",
                        prefix="nexusvoice_",
                        delete=False,
                    ) as f:
                        f.write(f"# Voice command: {meta.voice_command}\n")
                        f.write(f"# Model: {meta.model_used}\n")
                        f.write(f"# Latency: {meta.total_latency:.2f}s\n\n")
                        f.write(meta.final_code)
                        tmp_path = f.name

                    mlflow.log_artifact(tmp_path, artifact_path="generated_code")
                    Path(tmp_path).unlink(missing_ok=True)

                # Save transcription
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".txt",
                    prefix="transcription_",
                    delete=False,
                ) as f:
                    f.write(meta.voice_command)
                    tmp_txt = f.name

                mlflow.log_artifact(tmp_txt, artifact_path="transcriptions")
                Path(tmp_txt).unlink(missing_ok=True)

                console.print(
                    f"[green]✓ MLflow run logged[/green] "
                    f"[dim]run_id={run_id[:8]}... | "
                    f"model={meta.model_used}[/dim]"
                )
                logger.info(f"MLflow run logged: {run_id}")
                return run_id

        except Exception as e:
            console.print(f"[yellow]⚠ MLflow logging error: {e}[/yellow]")
            logger.error(f"MLflow tracking failed: {e}")
            return None

    def log_session_summary(self, stats: dict) -> None:
        """
        Log aggregate session statistics at the end of a loop session.
        Creates a separate 'session_summary' run in MLflow.
        """
        self._init()
        if not self._connected or self._mlflow is None:
            return

        mlflow = self._mlflow
        try:
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name="session_summary",
            ):
                mlflow.set_tag("run_type", "session_summary")
                mlflow.log_metrics({
                    "total_runs":          float(stats.get("total_runs", 0)),
                    "gemini_successes":    float(stats.get("gemini_successes", 0)),
                    "fallbacks_triggered": float(stats.get("fallbacks_triggered", 0)),
                    "both_failed":         float(stats.get("both_failed", 0)),
                    "gemini_success_rate": float(stats.get("gemini_success_rate", 0)),
                    "fallback_rate":       float(stats.get("fallback_rate", 0)),
                })
            console.print("[green]✓ Session summary logged to MLflow[/green]")
        except Exception as e:
            logger.error(f"Session summary logging failed: {e}")
