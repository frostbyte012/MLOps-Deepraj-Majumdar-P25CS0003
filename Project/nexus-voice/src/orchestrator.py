"""
nexusvoice/src/orchestrator.py
─────────────────────────────────────────────────────────────
LangGraph-based orchestration implementing the Observer Pattern.

State machine:
  transcribe → try_gemini → [success + fast] → finalize
                          → [slow / error]   → try_local → finalize

Stage 3 addition: every run is tracked in MLflow via MLflowTracker.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Literal
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)

LATENCY_THRESHOLD = float(os.getenv("LATENCY_THRESHOLD_SECONDS", "5.0"))


# ── State ─────────────────────────────────────────────────────
@dataclass
class PipelineState:
    voice_command: str = ""

    gemini_code: str = ""
    gemini_latency: float = 0.0
    gemini_success: bool = False
    gemini_error: str | None = None
    gemini_exceeded_threshold: bool = False

    local_code: str = ""
    local_latency: float = 0.0
    local_success: bool = False
    local_error: str | None = None

    final_code: str = ""
    model_used: str = ""
    fallback_triggered: bool = False
    fallback_reason: str | None = None

    total_latency: float = 0.0
    pipeline_start: float = field(default_factory=time.time)


# ── Nodes ─────────────────────────────────────────────────────

def node_try_gemini(state: PipelineState, gemini_codegen) -> PipelineState:
    console.print("\n[bold cyan]▶ Node: try_gemini[/bold cyan]")
    result = gemini_codegen.generate_code(state.voice_command)

    state.gemini_code = result.code
    state.gemini_latency = result.latency_seconds
    state.gemini_success = result.success
    state.gemini_error = result.error
    state.gemini_exceeded_threshold = result.exceeded_threshold

    logger.info(
        f"Gemini node: success={result.success} | "
        f"latency={result.latency_seconds:.2f}s | "
        f"exceeded={result.exceeded_threshold}"
    )
    return state


def node_try_local(state: PipelineState, local_codegen) -> PipelineState:
    console.print("\n[bold yellow]▶ Node: try_local (fallback)[/bold yellow]")
    result = local_codegen.generate_code(state.voice_command)

    state.local_code = result.code
    state.local_latency = result.latency_seconds
    state.local_success = result.success
    state.local_error = result.error

    logger.info(
        f"Local node: success={result.success} | "
        f"latency={result.latency_seconds:.2f}s"
    )
    return state


def node_finalize(state: PipelineState) -> PipelineState:
    console.print("\n[bold green]▶ Node: finalize[/bold green]")

    if state.gemini_success and not state.gemini_exceeded_threshold:
        state.final_code = state.gemini_code
        state.model_used = "gemini (primary)"
        state.fallback_triggered = False

    elif state.gemini_success and state.gemini_exceeded_threshold and not state.fallback_triggered:
        state.final_code = state.gemini_code
        state.model_used = "gemini (slow — above threshold)"
        state.fallback_triggered = False

    elif state.local_success:
        state.final_code = state.local_code
        state.model_used = "codet5-small (local fallback)"

    else:
        state.final_code = ""
        state.model_used = "none"
        console.print("[bold red]✗ Both Gemini and local model failed.[/bold red]")

    state.total_latency = time.time() - state.pipeline_start
    logger.info(
        f"Finalize: model_used={state.model_used} | "
        f"fallback={state.fallback_triggered} | "
        f"total={state.total_latency:.2f}s"
    )
    return state


# ── Router ────────────────────────────────────────────────────

def route_after_gemini(state: PipelineState) -> Literal["use_local", "finalize"]:
    if not state.gemini_success:
        state.fallback_triggered = True
        state.fallback_reason = "error"
        console.print(
            f"[yellow]⚡ Fallback triggered: Gemini error → routing to CodeT5[/yellow]"
        )
        logger.warning(f"Fallback: reason=error | error={state.gemini_error}")
        return "use_local"

    if state.gemini_exceeded_threshold:
        state.fallback_triggered = True
        state.fallback_reason = "latency"
        console.print(
            f"[yellow]⚡ Fallback triggered: latency {state.gemini_latency:.2f}s "
            f"> {LATENCY_THRESHOLD}s → routing to CodeT5[/yellow]"
        )
        logger.warning(
            f"Fallback: reason=latency | latency={state.gemini_latency:.2f}s"
        )
        return "use_local"

    console.print("[green]✓ Gemini succeeded within threshold — no fallback needed[/green]")
    return "finalize"


# ── Orchestrator ──────────────────────────────────────────────

class NexusVoiceOrchestrator:
    """
    LangGraph-style orchestrator with MLflow tracking on every run.
    """

    def __init__(self, gemini_codegen, local_codegen, tracker=None):
        self.gemini = gemini_codegen
        self.local = local_codegen
        self.tracker = tracker  # MLflowTracker instance (optional)
        self._session_stats = {
            "total_runs": 0,
            "gemini_successes": 0,
            "fallbacks_triggered": 0,
            "fallback_reasons": {"latency": 0, "error": 0},
            "both_failed": 0,
        }

    def run(self, voice_command: str) -> PipelineState:
        console.print(Panel(
            f"[bold]Command:[/bold] {voice_command}",
            title="[cyan]NexusVoice Orchestrator[/cyan]",
            border_style="cyan",
        ))

        state = PipelineState(
            voice_command=voice_command,
            pipeline_start=time.time(),
        )

        # ── Graph execution ────────────────────────────────
        state = node_try_gemini(state, self.gemini)
        route = route_after_gemini(state)
        if route == "use_local":
            state = node_try_local(state, self.local)
        state = node_finalize(state)

        # ── Session stats ──────────────────────────────────
        self._session_stats["total_runs"] += 1
        if state.gemini_success and not state.fallback_triggered:
            self._session_stats["gemini_successes"] += 1
        if state.fallback_triggered:
            self._session_stats["fallbacks_triggered"] += 1
            if state.fallback_reason:
                self._session_stats["fallback_reasons"][state.fallback_reason] = (
                    self._session_stats["fallback_reasons"].get(state.fallback_reason, 0) + 1
                )
        if not state.final_code:
            self._session_stats["both_failed"] += 1

        # ── MLflow tracking (Stage 3) ──────────────────────
        self._track(state)

        self._print_summary(state)
        return state

    def _track(self, state: PipelineState):
        """Send run data to MLflow tracker if available."""
        if self.tracker is None:
            return

        from src.mlflow_tracker import RunMetadata

        meta = RunMetadata(
            voice_command=state.voice_command,
            whisper_model=os.getenv("WHISPER_MODEL", "tiny"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_latency=state.gemini_latency,
            gemini_success=state.gemini_success,
            gemini_exceeded_threshold=state.gemini_exceeded_threshold,
            fallback_triggered=state.fallback_triggered,
            fallback_reason=state.fallback_reason,
            local_model=os.getenv("LOCAL_MODEL_ID", "Salesforce/codet5-small"),
            local_latency=state.local_latency,
            local_success=state.local_success,
            final_code=state.final_code,
            model_used=state.model_used,
            total_latency=state.total_latency,
            pipeline_success=bool(state.final_code),
            latency_threshold=float(os.getenv("LATENCY_THRESHOLD_SECONDS", "5.0")),
        )
        self.tracker.track_run(meta)

    def _print_summary(self, state: PipelineState):
        from rich.table import Table
        from rich import box

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Key", style="dim")
        table.add_column("Value")

        table.add_row("Command",   state.voice_command[:60])
        table.add_row("Model used", state.model_used)
        table.add_row(
            "Fallback",
            f"[yellow]YES ({state.fallback_reason})[/yellow]"
            if state.fallback_triggered else "[green]NO[/green]"
        )
        table.add_row(
            "Gemini latency",
            f"[yellow]{state.gemini_latency:.2f}s ⚠[/yellow]"
            if state.gemini_exceeded_threshold
            else f"{state.gemini_latency:.2f}s"
        )
        if state.fallback_triggered:
            table.add_row("Local latency", f"{state.local_latency:.2f}s")
        table.add_row("Total latency", f"{state.total_latency:.2f}s")
        table.add_row(
            "Status",
            "[green]Success[/green]" if state.final_code else "[red]Failed[/red]"
        )

        stats = self._session_stats
        table.add_row("", "")
        table.add_row("[dim]Session runs[/dim]",    str(stats["total_runs"]))
        table.add_row("[dim]Fallback rate[/dim]",
                      f"{stats['fallbacks_triggered']}/{stats['total_runs']}")

        console.print(Panel(table, title="[bold]Pipeline Summary[/bold]", border_style="dim"))

    def get_session_stats(self) -> dict:
        stats = self._session_stats.copy()
        total = stats["total_runs"]
        if total > 0:
            stats["gemini_success_rate"] = stats["gemini_successes"] / total
            stats["fallback_rate"]       = stats["fallbacks_triggered"] / total
        return stats
