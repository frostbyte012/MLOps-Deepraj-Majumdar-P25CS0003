"""
nexusvoice/src/transcriber.py
─────────────────────────────────────────────────────────────
Local speech-to-text using openai/whisper-tiny.
Runs fully offline — no API key required.
"""

import os
import time
import logging
import numpy as np
from dataclasses import dataclass
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float
    model_used: str
    confidence: float | None = None  # Whisper doesn't expose this directly


class WhisperTranscriber:
    """
    Wrapper around openai-whisper for local STT.
    Model is loaded once and reused across calls (expensive to reload).
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None  # Lazy-loaded on first use

    def _load_model(self):
        """Load Whisper model (downloads on first run, cached after)."""
        if self._model is not None:
            return

        console.print(f"[cyan]Loading Whisper model: [bold]{self.model_size}[/bold]...[/cyan]")
        try:
            import whisper
            start = time.time()
            self._model = whisper.load_model(self.model_size)
            elapsed = time.time() - start
            console.print(f"[green]✓ Whisper [{self.model_size}] loaded in {elapsed:.1f}s[/green]")
            logger.info(f"Whisper model '{self.model_size}' loaded in {elapsed:.2f}s")
        except ImportError:
            console.print("[bold red]✗ openai-whisper not installed.[/bold red]")
            console.print("[yellow]Run: pip install openai-whisper[/yellow]")
            raise
        except Exception as e:
            console.print(f"[bold red]✗ Failed to load Whisper: {e}[/bold red]")
            raise

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult | None:
        """
        Transcribe a numpy float32 audio array.

        Args:
            audio: 1D float32 array at 16kHz (from audio_capture.py)

        Returns:
            TranscriptionResult or None on failure.
        """
        self._load_model()

        if audio is None or len(audio) == 0:
            logger.warning("Empty audio passed to transcriber")
            return None

        # Check audio is not silent (RMS < 0.01 = likely no speech)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            console.print("[yellow]⚠ Audio appears silent (RMS: {:.4f}). Check microphone.[/yellow]".format(rms))
            logger.warning(f"Low audio energy detected: RMS={rms:.4f}")

        console.print("[cyan]Transcribing...[/cyan]")
        try:
            start = time.time()

            # Whisper expects float32 numpy array at 16kHz
            result = self._model.transcribe(
                audio,
                language=None,          # Auto-detect language
                task="transcribe",
                fp16=False,             # CPU-safe (set True if you have CUDA)
                verbose=False,
            )

            elapsed = time.time() - start
            text = result["text"].strip()
            language = result.get("language", "unknown")

            if not text:
                console.print("[yellow]⚠ No speech detected in recording.[/yellow]")
                return None

            console.print(f"[bold green]✓ Transcribed ({elapsed:.1f}s):[/bold green] [white]{text}[/white]")
            logger.info(f"Transcription ({elapsed:.2f}s): '{text}' | lang={language}")

            return TranscriptionResult(
                text=text,
                language=language,
                duration_seconds=elapsed,
                model_used=f"whisper-{self.model_size}",
            )

        except Exception as e:
            console.print(f"[bold red]✗ Transcription error: {e}[/bold red]")
            logger.error(f"Transcription failed: {e}")
            return None

    def transcribe_file(self, path: str) -> TranscriptionResult | None:
        """Transcribe directly from a .wav file path (for testing)."""
        from src.audio_capture import load_audio_file
        audio = load_audio_file(path)
        if audio is None:
            return None
        return self.transcribe(audio)
