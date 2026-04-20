"""
nexusvoice/src/audio_capture.py
─────────────────────────────────────────────────────────────
Handles microphone recording using sounddevice.
Returns a numpy array ready for Whisper transcription.
"""

import os
import time
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()
logger = logging.getLogger(__name__)

SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
CHANNELS = int(os.getenv("AUDIO_CHANNELS", 1))
RECORD_SECONDS = int(os.getenv("AUDIO_RECORD_SECONDS", 5))


def list_audio_devices() -> None:
    """Print all available audio input devices."""
    console.print("\n[bold cyan]Available audio devices:[/bold cyan]")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            console.print(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']})")


def check_audio_device() -> bool:
    """
    Verify a working input device exists.
    Returns True if audio is available, False otherwise.
    """
    try:
        default = sd.query_devices(kind="input")
        logger.info(f"Default input device: {default['name']}")
        return True
    except Exception as e:
        logger.error(f"No input device found: {e}")
        return False


def record_audio(
    duration: int = RECORD_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    save_path: str | None = None,
) -> np.ndarray | None:
    """
    Record audio from the default microphone.

    Args:
        duration:    Recording length in seconds.
        sample_rate: Sample rate (16kHz matches Whisper's expectation).
        save_path:   Optional path to save the .wav file for debugging.

    Returns:
        numpy array of shape (samples,) with dtype float32, or None on error.
    """
    if not check_audio_device():
        console.print("[bold red]✗ No microphone detected. Check your audio setup.[/bold red]")
        return None

    try:
        console.print(f"\n[bold green]● Recording for {duration}s...[/bold green] Speak your command now.")

        # Record — sounddevice returns (samples, channels)
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="float32",
        )

        # Live countdown
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Listening...", total=duration)
            for _ in range(duration * 10):
                time.sleep(0.1)
                progress.advance(task, 0.1)

        sd.wait()  # Block until recording is complete
        console.print("[bold green]✓ Recording complete.[/bold green]")

        # Flatten to 1D (Whisper wants mono float32)
        audio_flat = audio.flatten()

        # Optionally save for inspection / debugging
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(save_path, audio_flat, sample_rate)
            logger.info(f"Audio saved to {save_path}")

        return audio_flat

    except sd.PortAudioError as e:
        console.print(f"[bold red]✗ PortAudio error: {e}[/bold red]")
        console.print("[yellow]Tip: Run [bold]sudo apt install portaudio19-dev[/bold] if missing.[/yellow]")
        logger.error(f"PortAudio error: {e}")
        return None
    except Exception as e:
        console.print(f"[bold red]✗ Unexpected audio error: {e}[/bold red]")
        logger.error(f"Audio capture failed: {e}")
        return None


def load_audio_file(path: str) -> np.ndarray | None:
    """
    Load an existing .wav file as a numpy array.
    Useful for testing without a live mic.

    Args:
        path: Path to .wav file.

    Returns:
        numpy float32 array, or None on error.
    """
    try:
        audio, sr = sf.read(path, dtype="float32")
        if sr != SAMPLE_RATE:
            logger.warning(f"File sample rate {sr}Hz differs from target {SAMPLE_RATE}Hz")
        if audio.ndim > 1:
            audio = audio[:, 0]  # Take first channel if stereo
        logger.info(f"Loaded audio: {path} ({len(audio)/SAMPLE_RATE:.1f}s)")
        return audio
    except Exception as e:
        logger.error(f"Failed to load audio file {path}: {e}")
        return None
