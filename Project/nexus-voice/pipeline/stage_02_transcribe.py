"""
nexusvoice/pipeline/stage_02_transcribe.py
─────────────────────────────────────────────────────────────
DVC Pipeline Stage 2: Noise-Robustness Benchmarking

Runs Whisper STT on every audio sample in data/raw/
Measures Word Error Rate (WER) proxy across noise conditions.
Logs per-condition metrics to MLflow for model drift tracking.

Run directly:  python pipeline/stage_02_transcribe.py
Run via DVC:   dvc repro stage_02_transcribe
"""

import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
logger = logging.getLogger(__name__)

RAW_DIR       = Path(os.getenv("DATA_RAW_DIR",   "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROC_DIR",  "data/processed"))
BENCHMARK_DIR = Path(os.getenv("BENCHMARK_DIR",  "data/benchmarks"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT_NAME", "nexusvoice")


@dataclass
class SampleResult:
    file: str
    condition: str
    noise_level: float
    expected_command: str
    transcription: str
    transcription_latency_s: float
    word_count_expected: int
    word_count_got: int
    word_overlap_ratio: float  # proxy for accuracy (no ground truth needed)
    success: bool


def compute_word_overlap(expected: str, got: str) -> float:
    """
    Simple word overlap ratio as WER proxy.
    Real WER needs exact ground truth — this is sufficient for drift detection.
    Strips punctuation before comparison so "hello." matches "hello".
    """
    import string
    if not got.strip():
        return 0.0
    clean_expected = expected.lower().translate(str.maketrans('', '', string.punctuation))
    clean_got      = got.lower().translate(str.maketrans('', '', string.punctuation))
    expected_words = set(clean_expected.split())
    got_words      = set(clean_got.split())
    if not expected_words:
        return 0.0
    overlap = len(expected_words & got_words)
    return overlap / len(expected_words)


def load_manifest() -> list[dict]:
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]✗ No manifest found at {manifest_path}[/red]")
        console.print("[yellow]Run: python pipeline/stage_01_prepare_data.py first[/yellow]")
        raise FileNotFoundError(str(manifest_path))

    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["samples"]


def run_benchmarks(samples: list[dict]) -> list[SampleResult]:
    """Run Whisper on all samples and collect noise-robustness metrics."""
    console.print(f"\n[bold cyan]Stage 2: Noise-Robustness Benchmarking[/bold cyan]")
    console.print(f"[dim]Model: whisper-{WHISPER_MODEL} | Samples: {len(samples)}[/dim]\n")

    # Load Whisper once
    import whisper
    import soundfile as sf
    import numpy as np

    console.print(f"[cyan]Loading Whisper [{WHISPER_MODEL}]...[/cyan]")
    model = whisper.load_model(WHISPER_MODEL)
    console.print(f"[green]✓ Whisper loaded[/green]\n")

    results = []
    condition_stats: dict[str, list] = {}

    for sample in samples:
        filepath = Path(sample["file"])
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        condition    = sample.get("condition", "unknown")
        noise_level  = sample.get("noise_level", 0.0)
        expected_cmd = sample.get("command", "")

        # Load audio
        audio, _ = sf.read(str(filepath), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Transcribe
        start = time.time()
        try:
            result      = model.transcribe(audio, fp16=False, verbose=False)
            transcription = result["text"].strip()
            success     = bool(transcription)
        except Exception as e:
            transcription = ""
            success       = False
            logger.error(f"Transcription failed for {filepath}: {e}")

        latency = time.time() - start

        # Compute word overlap
        overlap = compute_word_overlap(expected_cmd, transcription)

        sample_result = SampleResult(
            file=str(filepath),
            condition=condition,
            noise_level=noise_level,
            expected_command=expected_cmd,
            transcription=transcription,
            transcription_latency_s=round(latency, 3),
            word_count_expected=len(expected_cmd.split()),
            word_count_got=len(transcription.split()),
            word_overlap_ratio=round(overlap, 3),
            success=success,
        )
        results.append(sample_result)

        # Accumulate per-condition
        if condition not in condition_stats:
            condition_stats[condition] = []
        condition_stats[condition].append(sample_result)

        console.print(
            f"  [{condition}] {filepath.name} | "
            f"overlap={overlap:.2f} | "
            f"latency={latency:.2f}s | "
            f"{'✓' if success else '✗'}"
        )

    return results, condition_stats


def save_results(results: list[SampleResult]) -> Path:
    """Save benchmark results to data/processed/ for DVC tracking."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "whisper_model": WHISPER_MODEL,
        "total_samples": len(results),
        "results": [asdict(r) for r in results],
    }

    output_path = PROCESSED_DIR / "transcription_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[dim]Results saved: {output_path}[/dim]")
    return output_path


def log_to_mlflow(condition_stats: dict, results: list[SampleResult]) -> None:
    """Log per-condition noise-robustness metrics to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)

        with mlflow.start_run(run_name=f"benchmark_whisper_{WHISPER_MODEL}"):
            mlflow.set_tag("run_type", "noise_robustness_benchmark")
            mlflow.set_tag("whisper_model", WHISPER_MODEL)
            mlflow.log_param("whisper_model", WHISPER_MODEL)
            mlflow.log_param("total_samples", len(results))

            # Overall metrics
            if results:
                overall_overlap  = sum(r.word_overlap_ratio for r in results) / len(results)
                overall_success  = sum(r.success for r in results) / len(results)
                avg_latency      = sum(r.transcription_latency_s for r in results) / len(results)

                mlflow.log_metrics({
                    "overall_word_overlap":   round(overall_overlap, 3),
                    "overall_success_rate":   round(overall_success, 3),
                    "avg_transcription_latency_s": round(avg_latency, 3),
                })

            # Per-condition metrics — key for model drift detection
            for condition, cond_results in condition_stats.items():
                if not cond_results:
                    continue
                avg_overlap  = sum(r.word_overlap_ratio for r in cond_results) / len(cond_results)
                success_rate = sum(r.success for r in cond_results) / len(cond_results)
                avg_lat      = sum(r.transcription_latency_s for r in cond_results) / len(cond_results)

                mlflow.log_metrics({
                    f"{condition}_word_overlap":   round(avg_overlap, 3),
                    f"{condition}_success_rate":   round(success_rate, 3),
                    f"{condition}_avg_latency_s":  round(avg_lat, 3),
                })

            # Log results file as artifact
            results_path = PROCESSED_DIR / "transcription_results.json"
            if results_path.exists():
                mlflow.log_artifact(str(results_path), artifact_path="benchmarks")

            console.print(f"[green]✓ Benchmark metrics logged to MLflow[/green]")

    except Exception as e:
        console.print(f"[yellow]⚠ MLflow logging failed: {e}[/yellow]")
        logger.error(f"MLflow benchmark logging failed: {e}")


def print_summary(condition_stats: dict) -> None:
    """Print a rich table showing noise robustness per condition."""
    table = Table(
        title=f"Noise-Robustness Benchmark — Whisper [{WHISPER_MODEL}]",
        box=box.ROUNDED,
        show_header=True,
    )
    table.add_column("Condition",      style="cyan")
    table.add_column("Samples",        justify="right")
    table.add_column("Success Rate",   justify="right")
    table.add_column("Word Overlap",   justify="right")
    table.add_column("Avg Latency",    justify="right")

    for condition, cond_results in sorted(condition_stats.items()):
        if not cond_results:
            continue
        n            = len(cond_results)
        success_rate = sum(r.success for r in cond_results) / n
        avg_overlap  = sum(r.word_overlap_ratio for r in cond_results) / n
        avg_latency  = sum(r.transcription_latency_s for r in cond_results) / n

        # Color-code overlap score
        overlap_str = f"{avg_overlap:.2f}"
        if avg_overlap >= 0.5:
            overlap_colored = f"[green]{overlap_str}[/green]"
        elif avg_overlap >= 0.2:
            overlap_colored = f"[yellow]{overlap_str}[/yellow]"
        else:
            overlap_colored = f"[red]{overlap_str}[/red]"

        table.add_row(
            condition,
            str(n),
            f"{success_rate:.0%}",
            overlap_colored,
            f"{avg_latency:.2f}s",
        )

    console.print()
    console.print(table)


def save_metrics(results: list[SampleResult]) -> None:
    """Save summary metrics.json for DVC contract tracking."""
    if not results:
        return

    overall_overlap = sum(r.word_overlap_ratio for r in results) / len(results)
    overall_success = sum(r.success for r in results) / len(results)
    avg_latency     = sum(r.transcription_latency_s for r in results) / len(results)

    metrics = {
        "overall_word_overlap": round(overall_overlap, 3),
        "overall_success_rate": round(overall_success, 3),
        "avg_latency_s":        round(avg_latency, 3),
    }

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BENCHMARK_DIR / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[dim]Metrics saved: {output_path}[/dim]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = load_manifest()
    results, condition_stats = run_benchmarks(samples)
    save_results(results)
    save_metrics(results)       # ← satisfies DVC contract for metrics.json
    print_summary(condition_stats)
    log_to_mlflow(condition_stats, results)

    console.print(f"\n[bold green]✓ Stage 2 complete[/bold green]")
    console.print(f"[dim]Next: dvc repro  (to run full pipeline)[/dim]")
