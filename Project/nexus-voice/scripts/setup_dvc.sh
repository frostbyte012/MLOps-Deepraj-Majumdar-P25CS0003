#!/usr/bin/env bash
# nexusvoice/scripts/setup_dvc.sh
# Sets up git + DVC for the NexusVoice data pipeline
# Run once: chmod +x scripts/setup_dvc.sh && ./scripts/setup_dvc.sh

set -e
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "\n${CYAN}${BOLD}NexusVoice — Stage 4: DVC Setup${NC}\n"

# ── 1. Install DVC ───────────────────────────────────────────
echo -e "${BOLD}[1/5] Installing DVC...${NC}"
pip install dvc==3.50.1 --quiet
echo -e "${GREEN}✓ DVC installed: $(dvc --version)${NC}"

# ── 2. Init git if needed ────────────────────────────────────
echo -e "\n${BOLD}[2/5] Initialising git...${NC}"
if [ ! -d ".git" ]; then
    git init
    git config user.email "nexusvoice@dev.local"
    git config user.name "NexusVoice"
    echo -e "${GREEN}✓ Git initialised${NC}"
else
    echo -e "${YELLOW}⚠ Git already initialised${NC}"
fi

# ── 3. Init DVC ──────────────────────────────────────────────
echo -e "\n${BOLD}[3/5] Initialising DVC...${NC}"
if [ ! -d ".dvc" ]; then
    dvc init
    echo -e "${GREEN}✓ DVC initialised${NC}"
else
    echo -e "${YELLOW}⚠ DVC already initialised${NC}"
fi

# ── 4. Configure DVC to use local cache ─────────────────────
echo -e "\n${BOLD}[4/5] Configuring DVC local cache...${NC}"
dvc config cache.type hardlink,symlink,copy
mkdir -p .dvc/cache
echo -e "${GREEN}✓ DVC cache configured${NC}"

# ── 5. Initial git commit ────────────────────────────────────
echo -e "\n${BOLD}[5/5] Creating initial git commit...${NC}"

# Stage all project files (excluding data/ and models/)
git add \
    main.py \
    requirements.txt \
    params.yaml \
    dvc.yaml \
    src/ \
    pipeline/ \
    scripts/ \
    .dvc/.gitignore \
    .dvc/config \
    2>/dev/null || true

# Create .gitignore if not present
cat > .gitignore << 'EOF'
# Python
venv/
__pycache__/
*.pyc
.env

# DVC cached data (tracked by DVC, not git)
/data/raw/
/data/processed/
/models/

# DVC artifacts (git tracks .dvc files, not the data)
*.dvc

# MLflow
mlruns/

# Logs
logs/
*.log
EOF

git add .gitignore 2>/dev/null || true

git diff --staged --quiet || git commit -m "feat: NexusVoice Stage 4 — DVC pipeline setup

- Add dvc.yaml pipeline with 3 stages
- Add params.yaml for reproducible benchmarks
- Add pipeline/ scripts for data prep, transcription, drift detection
- Configure DVC local cache
"

echo -e "\n${GREEN}${BOLD}✓ DVC setup complete!${NC}\n"
echo -e "Next steps:"
echo -e "  ${CYAN}1.${NC} python pipeline/stage_01_prepare_data.py  ${YELLOW}# generate dataset${NC}"
echo -e "  ${CYAN}2.${NC} python pipeline/stage_02_transcribe.py     ${YELLOW}# benchmark Whisper${NC}"
echo -e "  ${CYAN}3.${NC} python pipeline/stage_03_drift_detection.py ${YELLOW}# check drift${NC}"
echo -e "  ${CYAN}4.${NC} dvc repro                                   ${YELLOW}# run full pipeline${NC}"
echo -e "  ${CYAN}5.${NC} dvc dag                                     ${YELLOW}# visualise pipeline DAG${NC}"
echo ""
