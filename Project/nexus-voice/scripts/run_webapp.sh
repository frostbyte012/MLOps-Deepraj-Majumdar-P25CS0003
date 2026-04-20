#!/usr/bin/env bash
# nexusvoice/scripts/run_webapp.sh
# Run the FastAPI web server locally on the Linux server
# Access from Mac via: ssh -L 8080:localhost:8080 deepraj@10.6.0.52

set -e

cd "$(dirname "$0")/.."

source venv/bin/activate

# Install web server deps if not already installed
pip install fastapi==0.111.0 uvicorn==0.29.0 python-multipart==0.0.9 \
    pydantic==2.7.1 librosa --quiet

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  NexusVoice Web UI — Starting Server     ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  Server:   http://0.0.0.0:8080"
echo "  API docs: http://localhost:8080/docs"
echo ""
echo "  From your Mac, open a new terminal and run:"
echo "  ssh -L 8080:localhost:8080 deepraj@10.6.0.52"
echo "  Then open: http://localhost:8080"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

uvicorn webapp.server:app \
    --host 0.0.0.0 \
    --port 8080 \
    --reload \
    --log-level info