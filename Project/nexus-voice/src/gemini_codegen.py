"""
nexusvoice/src/gemini_codegen.py
─────────────────────────────────────────────────────────────
Code generation via Gemini 1.5 Flash.
Includes latency tracking (used in Stage 3 for fallback logic).
"""

import os
import time
import logging
from dataclasses import dataclass, field
from rich.console import Console
from rich.syntax import Syntax

console = Console()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LATENCY_THRESHOLD = float(os.getenv("LATENCY_THRESHOLD_SECONDS", "5.0"))

# System prompt — tells Gemini to ONLY produce clean, runnable code
SYSTEM_PROMPT = """You are NexusVoice, an AI that converts spoken developer commands into executable code.

Rules:
1. Output ONLY the requested code — no preamble, no "Here is...", no markdown fences.
2. Add brief inline comments explaining non-obvious lines.
3. If the command is ambiguous, produce the most common/useful interpretation.
4. Default language is Python unless the user specifies otherwise.
5. Keep code concise and production-quality.

Example input: "write a function that reverses a string"
Example output:
def reverse_string(s: str) -> str:
    return s[::-1]  # Python slice with step -1 reverses the sequence
"""


@dataclass
class CodegenResult:
    code: str
    prompt_used: str
    model: str
    latency_seconds: float
    success: bool = True
    error: str | None = None
    exceeded_threshold: bool = False


class GeminiCodegen:
    """
    Wrapper around google-generativeai for voice-to-code generation.
    Tracks per-call latency for Stage 3 fallback logic.
    """

    def __init__(self):
        self._client = None
        self._model = None
        self._latency_history: list[float] = []

    def _init_client(self):
        """Initialize Gemini client. Called lazily on first use."""
        if self._model is not None:
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY not set.\n"
                "1. Get a free key at https://aistudio.google.com\n"
                "2. Add it to your .env file: GEMINI_API_KEY=your_key"
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
                generation_config={
                    "temperature": 0.2,       # Low temp = more deterministic code
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            )
            console.print(f"[green]✓ Gemini [{GEMINI_MODEL}] initialized[/green]")
            logger.info(f"Gemini client initialized with model: {GEMINI_MODEL}")

        except ImportError:
            console.print("[bold red]✗ google-generativeai not installed.[/bold red]")
            console.print("[yellow]Run: pip install google-generativeai[/yellow]")
            raise
        except Exception as e:
            console.print(f"[bold red]✗ Gemini init failed: {e}[/bold red]")
            raise

    def generate_code(self, voice_command: str) -> CodegenResult:
        """
        Generate code from a transcribed voice command.

        Args:
            voice_command: Plain text from Whisper transcription.

        Returns:
            CodegenResult with generated code and latency metadata.
        """
        self._init_client()

        prompt = f"Voice command: {voice_command}"
        console.print(f"\n[cyan]Sending to Gemini:[/cyan] [italic]{voice_command}[/italic]")

        start = time.time()
        try:
            response = self._model.generate_content(prompt)
            latency = time.time() - start
            self._latency_history.append(latency)

            code = response.text.strip()

            # Strip accidental markdown fences if Gemini adds them
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

            exceeded = latency > LATENCY_THRESHOLD

            if exceeded:
                console.print(
                    f"[yellow]⚠ Gemini latency {latency:.2f}s exceeded threshold "
                    f"{LATENCY_THRESHOLD}s[/yellow]"
                )
                logger.warning(f"Latency threshold exceeded: {latency:.2f}s > {LATENCY_THRESHOLD}s")
            else:
                console.print(f"[green]✓ Gemini responded in {latency:.2f}s[/green]")

            logger.info(f"Gemini codegen: {latency:.2f}s | exceeded={exceeded} | chars={len(code)}")

            return CodegenResult(
                code=code,
                prompt_used=voice_command,
                model=GEMINI_MODEL,
                latency_seconds=latency,
                exceeded_threshold=exceeded,
            )

        except Exception as e:
            latency = time.time() - start
            error_msg = str(e)
            console.print(f"[bold red]✗ Gemini error ({latency:.2f}s): {error_msg}[/bold red]")
            logger.error(f"Gemini failed after {latency:.2f}s: {error_msg}")

            return CodegenResult(
                code="",
                prompt_used=voice_command,
                model=GEMINI_MODEL,
                latency_seconds=latency,
                success=False,
                error=error_msg,
            )

    def get_p95_latency(self) -> float | None:
        """Return P95 latency across all calls this session (for adaptive threshold in Stage 3)."""
        if len(self._latency_history) < 5:
            return None
        sorted_latencies = sorted(self._latency_history)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def display_code(self, result: CodegenResult) -> None:
        """Pretty-print generated code to terminal with syntax highlighting."""
        if not result.success or not result.code:
            console.print("[red]No code to display.[/red]")
            return

        console.print("\n[bold white]── Generated Code ──────────────────────────────[/bold white]")
        syntax = Syntax(result.code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print(f"[dim]Model: {result.model} | Latency: {result.latency_seconds:.2f}s[/dim]")
