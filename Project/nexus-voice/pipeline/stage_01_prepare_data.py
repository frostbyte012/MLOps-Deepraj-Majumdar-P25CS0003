"""
nexusvoice/pipeline/stage_01_prepare_data.py
─────────────────────────────────────────────────────────────
DVC Pipeline Stage 1: Dataset Preparation

Downloads a small developer-accent slice of Mozilla Common Voice
(or generates synthetic test audio if offline) and saves to data/raw/.

DVC tracks this output — any change to the dataset triggers
downstream pipeline stages to re-run automatically.

Run directly:  python pipeline/stage_01_prepare_data.py
Run via DVC:   dvc repro stage_01_prepare_data
"""

import os
import json
import logging
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

RAW_DIR        = Path(os.getenv("DATA_RAW_DIR",   "data/raw"))
PROCESSED_DIR  = Path(os.getenv("DATA_PROC_DIR",  "data/processed"))
SAMPLE_COUNT   = int(os.getenv("DATASET_SAMPLES", "50"))
SAMPLE_RATE    = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
DURATION_S     = 5   # seconds per clip


def generate_synthetic_audio(
    duration: int = DURATION_S,
    sample_rate: int = SAMPLE_RATE,
    noise_level: float = 0.01,
) -> np.ndarray:
    """
    Generate synthetic voice-like audio for offline testing.
    Simulates a developer speaking a coding command using:
    - Sine waves at speech frequencies (100-300 Hz fundamental)
    - Added Gaussian noise (controllable SNR)
    - Amplitude envelope (attack/decay) to simulate speech rhythm
    """
    t = np.linspace(0, duration, duration * sample_rate)

    # Fundamental frequency — simulate male/female voice range
    f0 = np.random.uniform(100, 300)

    # Build harmonics (voice = sum of harmonics)
    audio = np.zeros_like(t)
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic  # harmonics decay
        audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.7

    # Add speech rhythm — amplitude envelope
    envelope = np.ones_like(t)
    # Simulate word boundaries every ~0.5s
    for i in range(0, len(t), sample_rate // 2):
        gap = slice(i, min(i + sample_rate // 20, len(t)))
        envelope[gap] *= 0.1

    audio *= envelope

    # Add controllable noise
    noise = np.random.normal(0, noise_level, len(audio))
    audio = (audio + noise).astype(np.float32)

    return audio


def create_dataset_manifest(samples: list[dict], output_path: Path) -> None:
    """Save dataset manifest JSON — DVC uses this to track data lineage."""
    manifest = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source": "Mozilla Common Voice 17.0 (synthetic slice for offline testing)",
        "sample_count": len(samples),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_S,
        "samples": samples,
    }
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"[dim]Manifest saved: {output_path}[/dim]")


def prepare_common_voice_sample():
    """
    In production: download from Mozilla Common Voice via datasets library.
    For offline/dev: generate synthetic audio that mimics the data format.

    To use real Common Voice data:
        pip install datasets
        from datasets import load_dataset
        ds = load_dataset("mozilla-foundation/common_voice_17_0",
                          "en", split="train", streaming=True,
                          token=os.getenv("HF_TOKEN"))
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Stage 1: Preparing dataset ({SAMPLE_COUNT} samples)[/bold cyan]")
    console.print(f"[dim]Output: {RAW_DIR}[/dim]\n")

    # Check if real Common Voice data exists
    real_data = list(RAW_DIR.glob("*.wav"))
    if real_data:
        console.print(f"[green]✓ Found {len(real_data)} existing .wav files — skipping generation[/green]")
        return [{"file": str(f), "source": "real"} for f in real_data]

    # Generate synthetic dataset
    console.print(f"[yellow]Generating {SAMPLE_COUNT} synthetic audio samples...[/yellow]")
    console.print(f"[dim](Replace with real Common Voice data for production)[/dim]\n")

    # Noise levels to simulate different recording conditions
    noise_levels = {
        "clean":     0.005,
        "low_noise": 0.02,
        "mid_noise": 0.05,
        "high_noise":0.10,
    }

    # Coding commands that developers might speak
    commands = [
        "write a function that sorts a list",
        "create a binary search algorithm",
        "implement a stack data structure",
        "write a class for a linked list",
        "create a recursive fibonacci function",
        "implement bubble sort",
        "write a function to reverse a string",
        "create a dictionary from two lists",
        "implement a queue using two stacks",
        "write a decorator for caching",
    ]

    samples = []
    per_condition = SAMPLE_COUNT // len(noise_levels)

    for condition, noise_level in noise_levels.items():
        condition_dir = RAW_DIR / condition
        condition_dir.mkdir(exist_ok=True)

        for i in track(
            range(per_condition),
            description=f"[cyan]{condition}[/cyan]",
        ):
            audio = generate_synthetic_audio(noise_level=noise_level)
            command = commands[i % len(commands)]
            filename = f"{condition}_{i:03d}.wav"
            filepath = condition_dir / filename

            sf.write(str(filepath), audio, SAMPLE_RATE)

            samples.append({
                "file": str(filepath),
                "condition": condition,
                "noise_level": noise_level,
                "command": command,
                "duration_s": DURATION_S,
                "sample_rate": SAMPLE_RATE,
            })

    # Save manifest
    manifest_path = RAW_DIR / "manifest.json"
    create_dataset_manifest(samples, manifest_path)

    console.print(f"\n[bold green]✓ Dataset prepared: {len(samples)} samples[/bold green]")
    console.print(f"[dim]Conditions: {list(noise_levels.keys())}[/dim]")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()

    os.environ["DATASET_SAMPLES"] = str(args.samples)

    logging.basicConfig(level=logging.INFO)
    samples = prepare_common_voice_sample()
    console.print(f"\n[bold]Next:[/bold] Run [cyan]python pipeline/stage_02_transcribe.py[/cyan]")
