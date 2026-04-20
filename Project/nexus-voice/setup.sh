#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# nexusvoice/setup.sh
# One-shot setup for NexusVoice Stage 1 on Linux (native)
# Run: chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "\n${CYAN}${BOLD}NexusVoice — Stage 1 Setup${NC}\n"

# ── 1. System dependencies ──────────────────────────────────
echo -e "${BOLD}[1/5] Installing system audio libraries...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y \
        portaudio19-dev \
        libsndfile1 \
        libasound2-dev \
        pulseaudio \
        ffmpeg \
        python3-dev \
        python3-venv
    echo -e "${GREEN}✓ System packages installed${NC}"
else
    echo -e "${YELLOW}⚠ apt-get not found. Install manually:${NC}"
    echo "  portaudio19-dev, libsndfile1, libasound2-dev, pulseaudio, ffmpeg"
fi

# ── 2. Python version check ─────────────────────────────────
echo -e "\n${BOLD}[2/5] Checking Python version...${NC}"
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3)
PY_VERSION=$($PYTHON --version 2>&1)
echo -e "${GREEN}✓ Using: $PY_VERSION ($PYTHON)${NC}"

if ! $PYTHON -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo -e "${RED}✗ Python 3.10+ required. Current: $PY_VERSION${NC}"
    exit 1
fi

# ── 3. Virtual environment ──────────────────────────────────
echo -e "\n${BOLD}[3/5] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo -e "${GREEN}✓ venv created${NC}"
else
    echo -e "${YELLOW}⚠ venv already exists — skipping creation${NC}"
fi

source venv/bin/activate

# ── 4. Python packages ──────────────────────────────────────
echo -e "\n${BOLD}[4/5] Installing Python packages...${NC}"
pip install --upgrade pip --quiet

# Install PyAudio separately (needs portaudio headers)
echo "  → Installing PyAudio..."
pip install pyaudio --quiet || echo -e "${YELLOW}  ⚠ PyAudio failed — sounddevice will be used instead (that's fine)${NC}"

# Install main requirements
echo "  → Installing requirements.txt..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Python packages installed${NC}"

# ── 5. Environment file ─────────────────────────────────────
echo -e "\n${BOLD}[5/5] Setting up .env file...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠ .env created from template.${NC}"
    echo -e "${YELLOW}  → Open .env and set your GEMINI_API_KEY${NC}"
    echo -e "${YELLOW}  → Get a free key: https://aistudio.google.com${NC}"
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

# ── Done ────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}✓ Setup complete!${NC}\n"
echo -e "Next steps:"
echo -e "  ${CYAN}1.${NC} source venv/bin/activate"
echo -e "  ${CYAN}2.${NC} Edit .env and add your GEMINI_API_KEY"
echo -e "  ${CYAN}3.${NC} python main.py --check        ${YELLOW}# verify everything works${NC}"
echo -e "  ${CYAN}4.${NC} python main.py --text \"write a hello world in Python\"   ${YELLOW}# test without mic${NC}"
echo -e "  ${CYAN}5.${NC} python main.py                ${YELLOW}# live mic mode${NC}"
echo ""
