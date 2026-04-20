<div align="center">

# ⚡ NexusVoice

### Hybrid-Cloud MLOps Orchestrator for Real-Time Voice-to-Code Synthesis

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12.2-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.50-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)

<br/>

> **Speak a coding command. Get production-quality code.**
> Powered by Whisper STT + Gemini 2.5 Flash, with automatic fallback to CodeT5 when the cloud is unavailable.

<br/>

🌐 **[Live Demo](https://nexusvoice-xuznygilpa-uc.a.run.app)** · 📊 **[API Docs](https://nexusvoice-xuznygilpa-uc.a.run.app/docs)** · 🏥 **[Health Check](https://nexusvoice-xuznygilpa-uc.a.run.app/health)**

<br/>

```
🎙 Voice Input  →  🧠 Whisper STT  →  ⚡ LangGraph Router  →  💻 Generated Code
                                              ↓                        ↑
                                     Gemini 2.5 Flash ────────────────┘
                                     (if slow/down → CodeT5 fallback)
```

</div>

---

## 🗺️ What is NexusVoice?

NexusVoice is a **production-grade MLOps system** that converts spoken developer commands into executable code through a hybrid cloud-local inference pipeline. It applies four MLOps pillars holistically:

| Pillar | Implementation |
|--------|---------------|
| 🔄 **Resilient Inference** | LangGraph Observer Pattern with automatic Gemini → CodeT5 fallback |
| 📊 **Observability** | MLflow tracks every invocation with 8 metrics + code artifacts |
| 📦 **Reproducibility** | DVC versions datasets + 3-stage noise benchmark pipeline |
| ☁️ **Cloud Deployment** | GCP Cloud Run with Secret Manager, Cloud Build CI/CD |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NexusVoice Pipeline                          │
│                                                                     │
│  🎙 Audio Input                                                     │
│  (Browser WebRTC / sox on Mac / .wav file)                          │
│         │                                                           │
│         ▼                                                           │
│  🧠 Whisper-tiny (CUDA)          ← openai/whisper-tiny             │
│         │  transcription                                            │
│         ▼                                                           │
│  ⚡ LangGraph Orchestrator        ← Observer Pattern               │
│    ┌────┴─────────────────────┐                                     │
│    │     try_gemini node      │                                     │
│    └────┬──────────┬──────────┘                                     │
│         │ success  │ error/slow                                     │
│         ▼          ▼                                                │
│    ✅ Gemini   🔄 try_local node  ← Salesforce/codet5-small        │
│    2.5 Flash        │                                               │
│         │           │                                               │
│         └─────┬─────┘                                               │
│               ▼                                                     │
│         finalize node                                               │
│               │                                                     │
│         ┌─────┴──────┐                                              │
│         ▼            ▼                                              │
│    💻 Code Output   📊 MLflow Tracking                              │
│                      (latency, fallback, artifacts)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Five Pipeline Stages

```
Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5
  STT +      LangGraph    MLflow      DVC +        Web UI +
  Gemini     + Docker     Tracking    Drift        GCP Cloud
  Codegen    + CodeT5     Sidecar     Detection    Run Deploy
```

### ✅ Stage 1 — Audio + STT + Gemini Codegen
- 16kHz mono audio capture via `sounddevice`
- Silence detection via RMS energy threshold
- Whisper-tiny loaded once, reused across calls (CUDA-accelerated)
- Gemini 2.5 Flash with system prompt for code-only output
- CLI flags: `--text`, `--file`, `--loop`, `--check`, `--force-local`

### ✅ Stage 2 — LangGraph Orchestration + Docker
- Three-node state machine: `try_gemini → route → try_local/finalize`
- Observer Pattern router: fires on API error OR latency threshold breach
- CodeT5-small with 4-beam search, no-repeat-ngram protection
- Multi-stage `Dockerfile` with audio libs and HF model caching
- `docker-compose.yml` with MLflow sidecar on `nexusnet` bridge

### ✅ Stage 3 — MLflow Experiment Tracking
- **5 parameters** logged per run (model names, threshold, stage)
- **8 metrics** per run (latencies, success flags, fallback trigger, code length)
- **Artifacts**: generated `.py` file + transcription `.txt` per run
- **Tags**: `model_used`, `fallback_reason`, `voice_command`

### ✅ Stage 4 — DVC Data Pipeline + Drift Detection
- 3-stage `dvc.yaml` pipeline executed via `dvc repro`
- Audio across 4 noise conditions: clean, low, mid, high
- Drift detector flags HIGH/MEDIUM severity when degradation > 10%
- Correctly fired **3 HIGH-severity drift events** in validation

### ✅ Stage 5 — FastAPI Web UI + GCP Cloud Run
- Browser `MediaRecorder` WebRTC — no server audio hardware needed
- Real-time waveform via Web Audio API `AnalyserNode`
- FastAPI: `/transcribe`, `/generate`, `/metrics`, `/status`, `/health`
- GCP Cloud Run: 4 GiB RAM, 2 vCPUs, Secret Manager credential injection

---

## 📁 Project Structure

```
nexus-voice/
│
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Local Docker image
├── Dockerfile.cloudrun            # GCP Cloud Run image
├── docker-compose.yml             # App + MLflow sidecar
├── cloudbuild.yaml                # GCP Cloud Build config
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Pipeline parameters
├── .env.example                   # Environment template
│
├── src/
│   ├── audio_capture.py           # Mic recording + silence detection
│   ├── transcriber.py             # Whisper STT wrapper
│   ├── gemini_codegen.py          # Gemini API + latency tracking
│   ├── local_codegen.py           # CodeT5 local fallback
│   ├── orchestrator.py            # LangGraph state machine
│   ├── mlflow_tracker.py          # MLflow logging
│   └── logger.py                  # Rotating file logger
│
├── webapp/
│   ├── server.py                  # FastAPI backend
│   └── templates/index.html       # Browser voice UI (WebRTC)
│
├── pipeline/
│   ├── stage_01_prepare_data.py   # DVC Stage 1: dataset prep
│   ├── stage_02_transcribe.py     # DVC Stage 2: Whisper benchmark
│   └── stage_03_drift_detection.py # DVC Stage 3: drift detection
│
└── scripts/
    ├── setup_dvc.sh               # Git + DVC initialization
    ├── run_webapp.sh              # Local web server
    └── deploy_gcp.sh              # GCP Cloud Run deployment
```

---

## ⚡ Quick Start

### Prerequisites
```bash
sudo apt install portaudio19-dev libsndfile1 libasound2-dev ffmpeg pulseaudio
```

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/nexus-voice.git
cd nexus-voice
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
cp .env.example .env
nano .env  # Add GEMINI_API_KEY and HF_TOKEN
```

### CLI Mode
```bash
python main.py --check                                          # System check
python main.py --text "write a binary search function"         # Text mode
python main.py --text "write a stack class" --force-local      # Offline mode
LATENCY_THRESHOLD_SECONDS=0.1 python main.py --text "..."      # Force fallback
python main.py --file logs/recording.wav                       # From audio file
```

### Web UI (Mac → Linux Server)
```bash
# On server
./scripts/run_webapp.sh

# On Mac (new terminal)
ssh -L 8080:localhost:8080 deepraj@YOUR_SERVER_IP
open http://localhost:8080
```

### DVC Pipeline
```bash
./scripts/setup_dvc.sh   # Init git + DVC
dvc repro                # Run full 3-stage pipeline
dvc dag                  # Visualize pipeline DAG
```

### MLflow Dashboard
```bash
mlflow ui --port 5001 --host 0.0.0.0         # On server
ssh -L 5001:localhost:5001 user@server-ip    # Tunnel from Mac
# Open http://localhost:5001
```

---

## ☁️ GCP Deployment

```bash
# Install gcloud
curl https://sdk.cloud.google.com | bash && exec -l $SHELL
gcloud auth login --no-launch-browser

# Deploy
nano scripts/deploy_gcp.sh   # Set PROJECT_ID
./scripts/deploy_gcp.sh      # First run ~14 mins, subsequent ~2 mins

# Manage
gcloud run services delete nexusvoice --region us-central1   # Stop
./scripts/deploy_gcp.sh                                       # Restart (same URL)
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** [Get free key](https://aistudio.google.com/apikey) |
| `HF_TOKEN` | — | HuggingFace token for model download |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary code generation model |
| `WHISPER_MODEL` | `tiny` | ASR model size (tiny/base/small) |
| `LOCAL_MODEL_ID` | `Salesforce/codet5-small` | Fallback model |
| `LATENCY_THRESHOLD_SECONDS` | `5.0` | Fallback trigger threshold |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow server |
| `DRIFT_THRESHOLD` | `0.10` | Drift detection sensitivity |

---

## 📊 Results

### Fallback Routing

| Scenario | Trigger | Fallback | Result |
|----------|---------|----------|--------|
| Normal operation | None | ❌ | ✅ Gemini code |
| API error (403/429) | Error | ✅ | ✅ CodeT5 code |
| Latency threshold breach | `τ > τ_max` | ✅ | ✅ CodeT5 code |
| Force local | Manual | ✅ | ✅ CodeT5 code |

### Latency Profile

| Component | Mean | Max |
|-----------|------|-----|
| Whisper-tiny (CUDA) | 0.9s | 1.0s |
| Gemini 2.5 Flash | 5.44s | 8.35s |
| CodeT5-small (CUDA) | 0.65s | 1.02s |
| **Total (Gemini path)** | **6.17s** | **9.18s** |
| **Total (fallback path)** | **5.82s** | **6.87s** |

---

## 🎯 Mac → Linux Server Workflow

```bash
# Add to ~/.zshrc on Mac
nexusvoice() {
    echo "🎙 Recording 5s... speak your coding command"
    sox -d -r 16000 -c 1 -b 16 /tmp/nv_cmd.wav trim 0 5
    scp /tmp/nv_cmd.wav user@server:/path/to/nexus-voice/logs/latest.wav
    ssh user@server "cd /path/to/nexus-voice && source venv/bin/activate && python main.py --file logs/latest.wav"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **STT** | OpenAI Whisper-tiny (39M params, CUDA) |
| **Primary Codegen** | Gemini 2.5 Flash |
| **Fallback Codegen** | Salesforce/codet5-small (60M params, local) |
| **Orchestration** | LangGraph 0.0.55 |
| **Experiment Tracking** | MLflow 2.12.2 |
| **Data Versioning** | DVC 3.50.1 |
| **Web Framework** | FastAPI 0.111 + Uvicorn |
| **Voice Capture** | Browser WebRTC (MediaRecorder API) |
| **Containerization** | Docker multi-stage + Docker Compose |
| **Cloud Platform** | GCP Cloud Run + Cloud Build + Secret Manager |

---

## 👥 Authors

**Deepraj Majumdar** · p25cs0003@iitj.ac.in  
**Kushal Sharma** · m25cse0025@iitj.ac.in  
**Mayank Vatsa** · mvatsa@iitj.ac.in  

*Department of Computer Science and Engineering, IIT Jodhpur*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ at **IIT Jodhpur** · MLOps & DLOps Course Project · 2026

⭐ **Star this repo if you found it useful!**

</div>