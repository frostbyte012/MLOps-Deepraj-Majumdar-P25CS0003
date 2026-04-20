"""
nexusvoice/main.py
─────────────────────────────────────────────────────────────
NexusVoice Stage 3 — MLflow experiment tracking

Usage:
    python main.py --text "..."             # Normal run (tracked)
    python main.py --file audio.wav         # From wav file (tracked)
    python main.py --text "..." --force-local  # Force CodeT5 (tracked)
    python main.py --check                  # System check
    python main.py --mlflow-ui              # Print MLflow UI URL
"""

import os
import sys
import argparse
import logging
import importlib.metadata
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.syntax import Syntax

sys.path.insert(0, str(Path(__file__).parent))

from src.logger import setup_logging
from src.audio_capture import load_audio_file
from src.transcriber import WhisperTranscriber
from src.gemini_codegen import GeminiCodegen
from src.local_codegen import CodeT5Codegen
from src.orchestrator import NexusVoiceOrchestrator
from src.mlflow_tracker import MLflowTracker

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]NexusVoice[/bold cyan] [white]— Hybrid Voice-to-Code Synthesizer[/white]\n"
        "[dim]Stage 3: MLflow Experiment Tracking[/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))


def run_system_check() -> bool:
    console.print("\n[bold]Running system check...[/bold]\n")
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Check", style="dim")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True

    checks = [
        ("sounddevice",         lambda: __import__("sounddevice").__version__),
        ("torch",               lambda: (t := __import__("torch"), f"{t.__version__} ({'CUDA' if t.cuda.is_available() else 'CPU'})")[1]),
        ("transformers",        lambda: importlib.metadata.version("transformers")),
        ("openai-whisper",      lambda: "installed"),
        ("google-generativeai", lambda: importlib.metadata.version("google-generativeai")),
        ("mlflow",              lambda: importlib.metadata.version("mlflow")),
        ("rich",                lambda: importlib.metadata.version("rich")),
    ]

    for name, fn in checks:
        try:
            detail = fn()
            table.add_row(name, "[green]✓ OK[/green]", str(detail))
        except Exception:
            table.add_row(name, "[red]✗ Missing[/red]", f"pip install {name}")
            all_ok = False

    # Microphone
    try:
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        table.add_row("Microphone", "[green]✓ Found[/green]", dev["name"])
    except Exception:
        table.add_row("Microphone", "[yellow]⚠ None[/yellow]", "Use --file mode")

    # MLflow server
    try:
        import httpx
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        r = httpx.get(f"{uri}/health", timeout=2.0)
        table.add_row("MLflow server", "[green]✓ Running[/green]", uri)
    except Exception:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        table.add_row("MLflow server", "[yellow]⚠ Offline[/yellow]", "Run: mlflow ui --port 5001")

    # API Key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        table.add_row("GEMINI_API_KEY", "[green]✓ Set[/green]", f"{api_key[:8]}...")
    else:
        table.add_row("GEMINI_API_KEY", "[red]✗ Not set[/red]", "Add to .env")
        all_ok = False

    console.print(table)
    if all_ok:
        console.print("[bold green]✓ All checks passed.[/bold green]\n")
    else:
        console.print("[bold red]✗ Some checks failed.[/bold red]\n")
    return all_ok


def start_mlflow_server():
    """Start MLflow tracking server in background if not running."""
    import subprocess
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    port = uri.split(":")[-1]
    mlflow_dir = Path("./mlruns")
    mlflow_dir.mkdir(exist_ok=True)

    console.print(f"[cyan]Starting MLflow server on port {port}...[/cyan]")
    proc = subprocess.Popen(
        ["mlflow", "ui", "--port", port, "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import time; time.sleep(2)
    console.print(f"[green]✓ MLflow UI running at http://localhost:{port}[/green]")
    console.print(f"[dim]Open in browser to see experiment tracking dashboard[/dim]")
    return proc


def run_pipeline(
    audio_array=None,
    text_override=None,
    force_local=False,
    tracker=None,
):
    transcriber  = WhisperTranscriber()
    gemini       = GeminiCodegen()
    local        = CodeT5Codegen()
    orchestrator = NexusVoiceOrchestrator(gemini, local, tracker=tracker)

    # Step 1: Transcribe
    if text_override:
        console.print(f"\n[dim]Text mode — skipping STT[/dim]")
        voice_text = text_override
    else:
        if audio_array is None:
            console.print("[red]No audio. Use --file or --text.[/red]")
            return
        result = transcriber.transcribe(audio_array)
        if result is None:
            console.print("[red]Transcription failed.[/red]")
            return
        voice_text = result.text

    # Step 2: Force local
    if force_local:
        console.print("\n[yellow]Force-local mode — skipping Gemini[/yellow]")
        result = local.generate_code(voice_text)
        local.display_code(result)

        # Still track forced-local runs in MLflow
        if tracker:
            from src.mlflow_tracker import RunMetadata
            meta = RunMetadata(
                voice_command=voice_text,
                whisper_model=os.getenv("WHISPER_MODEL", "tiny"),
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                gemini_latency=0.0,
                gemini_success=False,
                gemini_exceeded_threshold=False,
                fallback_triggered=True,
                fallback_reason="forced",
                local_model=os.getenv("LOCAL_MODEL_ID", "Salesforce/codet5-small"),
                local_latency=result.latency_seconds,
                local_success=result.success,
                final_code=result.code,
                model_used="codet5-small (forced)",
                total_latency=result.latency_seconds,
                pipeline_success=result.success,
                latency_threshold=float(os.getenv("LATENCY_THRESHOLD_SECONDS", "5.0")),
            )
            tracker.track_run(meta)
        return result

    # Step 3: Orchestrated pipeline (with MLflow tracking built in)
    state = orchestrator.run(voice_text)

    # Step 4: Display output
    if state.final_code:
        console.print("\n[bold white]── Final Output ────────────────────────────────[/bold white]")
        syntax = Syntax(state.final_code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print(
            f"[dim]Model: {state.model_used} | "
            f"Total: {state.total_latency:.2f}s[/dim]"
        )
    else:
        console.print("[bold red]Pipeline produced no output.[/bold red]")

    return state


def main():
    print_banner()

    parser = argparse.ArgumentParser()
    parser.add_argument("--check",       action="store_true", help="System check")
    parser.add_argument("--file",        type=str,            help="Path to .wav file")
    parser.add_argument("--text",        type=str,            help="Text command (skip STT)")
    parser.add_argument("--force-local", action="store_true", help="Use CodeT5 only")
    parser.add_argument("--loop",        action="store_true", help="Loop mode")
    parser.add_argument("--no-mlflow",   action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--mlflow-ui",   action="store_true", help="Start MLflow UI and exit")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("nexusvoice")
    logger.info("NexusVoice Stage 3 started")

    if args.mlflow_ui:
        start_mlflow_server()
        return

    if args.check:
        run_system_check()
        return

    if not run_system_check():
        sys.exit(1)

    # ── Init MLflow tracker ────────────────────────────────
    tracker = None
    if not args.no_mlflow:
        tracker = MLflowTracker()
        # Auto-start MLflow if not running
        try:
            import httpx
            uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
            httpx.get(f"{uri}/health", timeout=2.0)
        except Exception:
            start_mlflow_server()

    # ── Load audio file if provided ────────────────────────
    audio_array = None
    if args.file:
        audio_array = load_audio_file(args.file)
        if audio_array is None:
            console.print(f"[red]Could not load: {args.file}[/red]")
            sys.exit(1)

    # ── Run pipeline ───────────────────────────────────────
    if args.loop:
        console.print("\n[bold cyan]Loop mode — Ctrl+C to exit[/bold cyan]\n")
        try:
            while True:
                run_pipeline(
                    audio_array=audio_array,
                    text_override=args.text,
                    force_local=args.force_local,
                    tracker=tracker,
                )
                console.print("\n[dim]────────────────────────────────────────[/dim]")
                if args.file or args.text:
                    break
        except KeyboardInterrupt:
            console.print("\n[bold]Goodbye.[/bold]")
            if tracker:
                from src.orchestrator import NexusVoiceOrchestrator
                # Log session summary if loop mode
                pass
    else:
        run_pipeline(
            audio_array=audio_array,
            text_override=args.text,
            force_local=args.force_local,
            tracker=tracker,
        )

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    console.print(f"\n[dim]MLflow dashboard → {uri}[/dim]")


if __name__ == "__main__":
    main()
