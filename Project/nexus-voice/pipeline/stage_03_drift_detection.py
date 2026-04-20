"""
nexusvoice/pipeline/stage_03_drift_detection.py
─────────────────────────────────────────────────────────────
DVC Pipeline Stage 3: Model Drift Detection

Compares current benchmark results against a saved baseline.
Flags drift if noise-robustness metrics degrade beyond threshold.
Logs drift report to MLflow.

Run directly:  python pipeline/stage_03_drift_detection.py
Run via DVC:   dvc repro
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
logger = logging.getLogger(__name__)

PROCESSED_DIR  = Path(os.getenv("DATA_PROC_DIR",  "data/processed"))
BENCHMARK_DIR  = Path(os.getenv("BENCHMARK_DIR",  "data/benchmarks"))
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT     = os.getenv("MLFLOW_EXPERIMENT_NAME", "nexusvoice")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.10"))  # 10% degradation = drift


def load_current_results() -> dict:
    path = PROCESSED_DIR / "transcription_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Run stage_02 first: {path}")
    with open(path) as f:
        return json.load(f)


def load_baseline() -> dict | None:
    path = BENCHMARK_DIR / "baseline.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_baseline(metrics: dict) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / "baseline.json"
    baseline = {
        "created_at":  datetime.now().isoformat(),
        "whisper_model": os.getenv("WHISPER_MODEL", "tiny"),
        "metrics": metrics,
    }
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    console.print(f"[green]✓ Baseline saved: {path}[/green]")


def compute_condition_metrics(results_data: dict) -> dict:
    """Aggregate per-condition metrics from raw results."""
    from collections import defaultdict
    condition_data = defaultdict(list)

    for r in results_data.get("results", []):
        condition_data[r["condition"]].append(r)

    metrics = {}
    for condition, samples in condition_data.items():
        n = len(samples)
        if n == 0:
            continue
        metrics[condition] = {
            "word_overlap":  round(sum(s["word_overlap_ratio"] for s in samples) / n, 3),
            "success_rate":  round(sum(s["success"] for s in samples) / n, 3),
            "avg_latency_s": round(sum(s["transcription_latency_s"] for s in samples) / n, 3),
            "n_samples":     n,
        }

    # Overall
    all_samples = results_data.get("results", [])
    if all_samples:
        metrics["_overall"] = {
            "word_overlap":  round(sum(s["word_overlap_ratio"] for s in all_samples) / len(all_samples), 3),
            "success_rate":  round(sum(s["success"] for s in all_samples) / len(all_samples), 3),
            "avg_latency_s": round(sum(s["transcription_latency_s"] for s in all_samples) / len(all_samples), 3),
            "n_samples":     len(all_samples),
        }

    return metrics


def detect_drift(current: dict, baseline: dict) -> list[dict]:
    """
    Compare current metrics vs baseline.
    Returns list of drift events where degradation > DRIFT_THRESHOLD.
    """
    drift_events = []

    for condition, cur_metrics in current.items():
        if condition not in baseline:
            continue

        base_metrics = baseline[condition]

        for metric in ["word_overlap", "success_rate"]:
            cur_val  = cur_metrics.get(metric, 0)
            base_val = base_metrics.get(metric, 0)

            if base_val == 0:
                continue

            degradation = (base_val - cur_val) / base_val

            if degradation > DRIFT_THRESHOLD:
                drift_events.append({
                    "condition":   condition,
                    "metric":      metric,
                    "baseline":    base_val,
                    "current":     cur_val,
                    "degradation": round(degradation, 3),
                    "threshold":   DRIFT_THRESHOLD,
                    "severity":    "HIGH" if degradation > 0.25 else "MEDIUM",
                })

    return drift_events


def save_drift_report(drift_events: list, current_metrics: dict) -> Path:
    report = {
        "generated_at":   datetime.now().isoformat(),
        "drift_detected": len(drift_events) > 0,
        "n_drift_events": len(drift_events),
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_events":   drift_events,
        "current_metrics": current_metrics,
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / "drift_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def log_drift_to_mlflow(drift_events: list, current_metrics: dict) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name="drift_detection"):
            mlflow.set_tag("run_type", "drift_detection")
            mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
            mlflow.log_param("n_drift_events", len(drift_events))
            mlflow.log_metric("drift_detected", float(len(drift_events) > 0))
            mlflow.log_metric("n_drift_events", float(len(drift_events)))

            # Log current overall metrics
            if "_overall" in current_metrics:
                overall = current_metrics["_overall"]
                mlflow.log_metrics({
                    "current_overall_word_overlap": overall["word_overlap"],
                    "current_overall_success_rate": overall["success_rate"],
                })

            # Log drift report as artifact
            report_path = BENCHMARK_DIR / "drift_report.json"
            if report_path.exists():
                mlflow.log_artifact(str(report_path), "drift_reports")

            console.print("[green]✓ Drift report logged to MLflow[/green]")

    except Exception as e:
        console.print(f"[yellow]⚠ MLflow logging failed: {e}[/yellow]")


def print_drift_report(drift_events: list, current: dict, baseline: dict | None):
    if not baseline:
        console.print(Panel(
            "[yellow]No baseline found — current metrics saved as new baseline.\n"
            "Run again after data changes to detect drift.[/yellow]",
            title="Drift Detection",
            border_style="yellow",
        ))
        return

    if not drift_events:
        console.print(Panel(
            "[green]✓ No drift detected — all metrics within threshold[/green]",
            title="Drift Detection",
            border_style="green",
        ))
    else:
        table = Table(
            title=f"⚠ Drift Detected — {len(drift_events)} event(s)",
            box=box.ROUNDED,
        )
        table.add_column("Condition",   style="cyan")
        table.add_column("Metric",      style="white")
        table.add_column("Baseline",    justify="right")
        table.add_column("Current",     justify="right")
        table.add_column("Degradation", justify="right")
        table.add_column("Severity",    justify="center")

        for event in drift_events:
            severity_color = "red" if event["severity"] == "HIGH" else "yellow"
            table.add_row(
                event["condition"],
                event["metric"],
                str(event["baseline"]),
                str(event["current"]),
                f"{event['degradation']:.1%}",
                f"[{severity_color}]{event['severity']}[/{severity_color}]",
            )
        console.print(table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    console.print("\n[bold cyan]Stage 3: Drift Detection[/bold cyan]\n")

    # Load current results
    current_data    = load_current_results()
    current_metrics = compute_condition_metrics(current_data)

    # Load baseline
    baseline_data = load_baseline()
    baseline_metrics = baseline_data["metrics"] if baseline_data else None

    # Detect drift
    if baseline_metrics:
        drift_events = detect_drift(current_metrics, baseline_metrics)
    else:
        drift_events = []
        console.print("[dim]No baseline — establishing current as baseline[/dim]")
        save_baseline(current_metrics)

    # Print and save report
    print_drift_report(drift_events, current_metrics, baseline_metrics)
    report_path = save_drift_report(drift_events, current_metrics)
    log_drift_to_mlflow(drift_events, current_metrics)

    console.print(f"\n[dim]Report saved: {report_path}[/dim]")
    console.print(f"[bold green]✓ Stage 3 complete[/bold green]")
