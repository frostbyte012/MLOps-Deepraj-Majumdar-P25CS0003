"""
nexusvoice/webapp/server.py
─────────────────────────────────────────────────────────────
FastAPI backend for NexusVoice Web UI.

Endpoints:
  GET  /           → serve the web UI
  POST /transcribe → receive audio blob → Whisper STT → return text
  POST /generate   → text command → orchestrator → return code
  GET  /health     → Cloud Run health check
  GET  /metrics    → latest MLflow run metrics
  GET  /status     → pipeline status

Run locally:
  uvicorn webapp.server:app --host 0.0.0.0 --port 8080 --reload

Access from Mac via existing SSH tunnel:
  ssh -L 8080:localhost:8080 deepraj@10.6.0.52
  → open http://localhost:8080
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import soundfile as sf

from src.logger import setup_logging
from src.transcriber import WhisperTranscriber
from src.gemini_codegen import GeminiCodegen
from src.local_codegen import CodeT5Codegen
from src.orchestrator import NexusVoiceOrchestrator
from src.mlflow_tracker import MLflowTracker

# ── Setup ─────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger("nexusvoice.server")

app = FastAPI(
    title="NexusVoice",
    description="Hybrid Voice-to-Code Synthesizer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded singletons ────────────────────────────────────
_transcriber  = None
_gemini       = None
_local        = None
_orchestrator = None
_tracker      = None


def get_orchestrator():
    global _transcriber, _gemini, _local, _orchestrator, _tracker
    if _orchestrator is None:
        _transcriber  = WhisperTranscriber()
        _gemini       = GeminiCodegen()
        _local        = CodeT5Codegen()
        _tracker      = MLflowTracker()
        _orchestrator = NexusVoiceOrchestrator(_gemini, _local, tracker=_tracker)
    return _orchestrator, _transcriber


# ── Request/Response models ───────────────────────────────────
class TextRequest(BaseModel):
    command: str


class PipelineResponse(BaseModel):
    code: str
    model_used: str
    fallback_triggered: bool
    fallback_reason: str | None
    gemini_latency: float
    total_latency: float
    success: bool


# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Cloud Run health check endpoint."""
    return {"status": "ok", "service": "nexusvoice"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>NexusVoice UI not found</h1>")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Receive audio blob from browser mic → Whisper STT → return text.
    Accepts: audio/webm, audio/wav, audio/ogg
    """
    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".webm", delete=False
        ) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Convert to numpy float32 via soundfile
        try:
            audio_data, sr = sf.read(tmp_path, dtype="float32")
        except Exception:
            # Try ffmpeg conversion if soundfile fails (webm needs ffmpeg)
            import subprocess
            wav_path = tmp_path.replace(".webm", ".wav")
            subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-ar", "16000",
                 "-ac", "1", wav_path, "-y", "-loglevel", "quiet"],
                check=True
            )
            audio_data, sr = sf.read(wav_path, dtype="float32")
            Path(wav_path).unlink(missing_ok=True)

        Path(tmp_path).unlink(missing_ok=True)

        # Flatten to mono
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        # Transcribe
        _, transcriber = get_orchestrator()
        result = transcriber.transcribe(audio_data)

        if result is None:
            return JSONResponse({"text": "", "error": "No speech detected"})

        return JSONResponse({
            "text":     result.text,
            "language": result.language,
            "duration": result.duration_seconds,
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=PipelineResponse)
async def generate_code(req: TextRequest):
    """
    Run full orchestrated pipeline on a text command.
    Returns generated code + pipeline metadata.
    """
    if not req.command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")

    try:
        orchestrator, _ = get_orchestrator()
        state = orchestrator.run(req.command)

        return PipelineResponse(
            code=state.final_code,
            model_used=state.model_used,
            fallback_triggered=state.fallback_triggered,
            fallback_reason=state.fallback_reason,
            gemini_latency=round(state.gemini_latency, 2),
            total_latency=round(state.total_latency, 2),
            success=bool(state.final_code),
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Return latest session stats from the orchestrator."""
    try:
        orchestrator, _ = get_orchestrator()
        stats = orchestrator.get_session_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/status")
async def get_status():
    """Return system status — model availability, config."""
    return JSONResponse({
        "whisper_model":   os.getenv("WHISPER_MODEL", "tiny"),
        "gemini_model":    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "local_model":     os.getenv("LOCAL_MODEL_ID", "Salesforce/codet5-small"),
        "threshold_s":     float(os.getenv("LATENCY_THRESHOLD_SECONDS", "5.0")),
        "mlflow_uri":      os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        "environment":     os.getenv("ENVIRONMENT", "development"),
    })