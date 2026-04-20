"""
nexusvoice/src/local_codegen.py
─────────────────────────────────────────────────────────────
Offline code generation using Salesforce/codet5-small.
Runs fully locally — no API key, no internet required.
Used as fallback when Gemini exceeds latency threshold or fails.
"""

import os
import time
import logging
from dataclasses import dataclass
from rich.console import Console
from rich.syntax import Syntax

console = Console()
logger = logging.getLogger(__name__)

HF_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "Salesforce/codet5-small")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./models")
MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "256"))


@dataclass
class LocalCodegenResult:
    code: str
    prompt_used: str
    model: str
    latency_seconds: float
    success: bool = True
    error: str | None = None


class CodeT5Codegen:
    """
    Wrapper around Salesforce/codet5-small for offline code generation.
    Model is lazy-loaded on first use and cached in ./models/.
    Falls back gracefully if transformers isn't installed.
    """

    def __init__(self, model_id: str = HF_MODEL_ID):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Load CodeT5 model and tokenizer. Downloads once, cached after."""
        if self._model is not None:
            return

        console.print(f"[cyan]Loading local model: [bold]{self.model_id}[/bold]...[/cyan]")
        console.print(f"[dim]Cache dir: {HF_CACHE_DIR} (downloads ~300MB on first run)[/dim]")

        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            # Pick best available device
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            start = time.time()

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=HF_CACHE_DIR,
		token=os.getenv("HF_TOKEN"),
            )
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=HF_CACHE_DIR,
		token=os.getenv("HF_TOKEN"),
            ).to(self._device)

            self._model.eval()  # Inference mode — disables dropout

            elapsed = time.time() - start
            console.print(
                f"[green]✓ CodeT5-small loaded in {elapsed:.1f}s "
                f"[dim]({self._device})[/dim][/green]"
            )
            logger.info(f"CodeT5 loaded in {elapsed:.2f}s on {self._device}")

        except ImportError:
            console.print("[bold red]✗ transformers not installed.[/bold red]")
            console.print("[yellow]Run: pip install transformers[/yellow]")
            raise
        except Exception as e:
            console.print(f"[bold red]✗ Failed to load CodeT5: {e}[/bold red]")
            raise

    def _build_prompt(self, voice_command: str) -> str:
        """
        CodeT5 works best with structured prompts.
        Format the voice command into a code generation instruction.
        """
        # Normalize common voice patterns
        command = voice_command.lower().strip()
        command = command.replace("write a ", "").replace("write an ", "")
        command = command.replace("create a ", "").replace("make a ", "")

        # CodeT5 prompt format — concise and directive
        return f"def {command.replace(' ', '_')}():"

    def generate_code(self, voice_command: str) -> LocalCodegenResult:
        """
        Generate code from a voice command using local CodeT5.

        Args:
            voice_command: Plain text from Whisper transcription.

        Returns:
            LocalCodegenResult with generated code and timing.
        """
        self._load_model()

        import torch

        prompt = self._build_prompt(voice_command)
        console.print(f"\n[yellow]⚡ Local CodeT5 generating...[/yellow] [dim](offline)[/dim]")
        logger.info(f"CodeT5 generating for: '{voice_command}'")

        start = time.time()
        try:
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
            ).to(self._device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs["input_ids"],
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=4,              # Beam search for better quality
                    early_stopping=True,
                    no_repeat_ngram_size=2,   # Avoid repetition
                    temperature=0.7,
                )

            latency = time.time() - start

            # Decode output
            code = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Wrap in function if CodeT5 didn't include def
            if not code.strip().startswith("def") and not code.strip().startswith("class"):
                code = f"{prompt}\n    {code}"

            console.print(f"[green]✓ CodeT5 generated in {latency:.2f}s [dim](local)[/dim][/green]")
            logger.info(f"CodeT5 success: {latency:.2f}s | chars={len(code)}")

            return LocalCodegenResult(
                code=code,
                prompt_used=voice_command,
                model=self.model_id,
                latency_seconds=latency,
            )

        except Exception as e:
            latency = time.time() - start
            console.print(f"[bold red]✗ CodeT5 error ({latency:.2f}s): {e}[/bold red]")
            logger.error(f"CodeT5 failed: {e}")

            return LocalCodegenResult(
                code="",
                prompt_used=voice_command,
                model=self.model_id,
                latency_seconds=latency,
                success=False,
                error=str(e),
            )

    def display_code(self, result: LocalCodegenResult) -> None:
        """Pretty-print generated code with syntax highlighting."""
        if not result.success or not result.code:
            console.print("[red]No code to display.[/red]")
            return

        console.print("\n[bold white]── Generated Code (Local) ──────────────────────[/bold white]")
        syntax = Syntax(result.code, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print(f"[dim]Model: {result.model} | Latency: {result.latency_seconds:.2f}s | Device: {self._device}[/dim]")

    def warmup(self):
        """
        Pre-load the model so first real request isn't slow.
        Call this at startup in Docker entrypoint.
        """
        console.print("[dim]Warming up CodeT5...[/dim]")
        self.generate_code("print hello world")
        console.print("[dim]CodeT5 warm.[/dim]")
