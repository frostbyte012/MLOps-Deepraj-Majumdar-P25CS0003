#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# nexusvoice/scripts/deploy_gcp.sh
# Deploy NexusVoice to GCP Cloud Run
#
# Prerequisites:
#   1. GCP account with billing enabled
#   2. gcloud CLI installed (brew install google-cloud-sdk on Mac)
#   3. Run this from your Mac (not the Linux server)
#
# Usage:
#   chmod +x scripts/deploy_gcp.sh
#   ./scripts/deploy_gcp.sh
# ─────────────────────────────────────────────────────────────

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Config — edit these ───────────────────────────────────────
PROJECT_ID="nexusvoice-2026"         # Your GCP project ID (e.g. nexusvoice-2026)
REGION="us-central1"   # Cloud Run region
SERVICE_NAME="nexusvoice"
IMAGE_NAME="nexusvoice-app"

echo -e "\n${CYAN}${BOLD}NexusVoice — GCP Cloud Run Deployment${NC}\n"

# ── Step 0: Check gcloud is installed ────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found${NC}"
    echo -e "Install it: ${CYAN}brew install google-cloud-sdk${NC}"
    echo -e "Then run:   ${CYAN}gcloud init${NC}"
    exit 1
fi

# ── Step 1: Get/set project ───────────────────────────────────
echo -e "${BOLD}[1/7] Setting GCP project...${NC}"
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${YELLOW}No project set. Run: gcloud projects list${NC}"
        echo -e "Then set it: ${CYAN}gcloud config set project YOUR_PROJECT_ID${NC}"
        exit 1
    fi
fi

gcloud config set project "$PROJECT_ID"
echo -e "${GREEN}✓ Project: $PROJECT_ID${NC}"

# ── Step 2: Enable required APIs ─────────────────────────────
echo -e "\n${BOLD}[2/7] Enabling GCP APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com \
    --quiet
echo -e "${GREEN}✓ APIs enabled${NC}"

# ── Step 3: Store secrets in Secret Manager ──────────────────
echo -e "\n${BOLD}[3/7] Storing secrets in GCP Secret Manager...${NC}"

# Read from local .env
GEMINI_KEY=$(grep GEMINI_API_KEY .env | cut -d= -f2)
HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2 2>/dev/null || echo "")

if [ -z "$GEMINI_KEY" ]; then
    echo -e "${YELLOW}⚠ GEMINI_API_KEY not found in .env${NC}"
else
    echo "$GEMINI_KEY" | gcloud secrets create nexusvoice-gemini-key \
        --data-file=- --quiet 2>/dev/null || \
    echo "$GEMINI_KEY" | gcloud secrets versions add nexusvoice-gemini-key \
        --data-file=- --quiet
    echo -e "${GREEN}✓ Gemini API key stored in Secret Manager${NC}"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "$HF_TOKEN" | gcloud secrets create nexusvoice-hf-token \
        --data-file=- --quiet 2>/dev/null || \
    echo "$HF_TOKEN" | gcloud secrets versions add nexusvoice-hf-token \
        --data-file=- --quiet
    echo -e "${GREEN}✓ HF token stored in Secret Manager${NC}"
fi

# ── Step 4: Build and push Docker image ──────────────────────
echo -e "\n${BOLD}[4/7] Building Docker image with Cloud Build...${NC}"
echo -e "${YELLOW}(This takes 5-10 mins on first build — models are downloaded)${NC}\n"

IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

gcloud builds submit \
    --config cloudbuild.yaml \
    --timeout=30m \
    .

echo -e "${GREEN}✓ Image built and pushed: $IMAGE_URI${NC}"

# ── Step 5: Deploy to Cloud Run ───────────────────────────────
echo -e "\n${BOLD}[5/7] Deploying to Cloud Run...${NC}"

gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_URI" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "ENVIRONMENT=production,WHISPER_MODEL=tiny,GEMINI_MODEL=gemini-2.5-flash,LATENCY_THRESHOLD_SECONDS=5.0,HF_CACHE_DIR=/app/models,TRANSFORMERS_CACHE=/app/models/transformers" \
    --set-secrets "GEMINI_API_KEY=nexusvoice-gemini-key:latest" \
    --quiet

echo -e "${GREEN}✓ Deployed to Cloud Run${NC}"

# ── Step 6: Get service URL ───────────────────────────────────
echo -e "\n${BOLD}[6/7] Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format "value(status.url)")

echo -e "${GREEN}✓ Service URL: ${CYAN}${SERVICE_URL}${NC}"

# ── Step 7: Health check ──────────────────────────────────────
echo -e "\n${BOLD}[7/7] Running health check...${NC}"
sleep 5
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")

if [ "$HTTP_STATUS" = "200" ]; then
    echo -e "${GREEN}✓ Health check passed (HTTP 200)${NC}"
else
    echo -e "${YELLOW}⚠ Health check returned HTTP $HTTP_STATUS (may need a moment to warm up)${NC}"
fi

# ── Done ──────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}✓ NexusVoice deployed to GCP Cloud Run!${NC}\n"
echo -e "  ${CYAN}Web UI:${NC}      ${SERVICE_URL}"
echo -e "  ${CYAN}API Docs:${NC}    ${SERVICE_URL}/docs"
echo -e "  ${CYAN}Health:${NC}      ${SERVICE_URL}/health"
echo -e "  ${CYAN}Status:${NC}      ${SERVICE_URL}/status"
echo ""
echo -e "  ${YELLOW}Note:${NC} First request after idle may take 10-15s (cold start)"
echo -e "  ${YELLOW}Cost:${NC} ~\$0 for demo usage (within free tier limits)"
echo ""
echo -e "To view logs:"
echo -e "  ${CYAN}gcloud run services logs read $SERVICE_NAME --region $REGION${NC}"
echo ""
echo -e "To delete when done:"
echo -e "  ${CYAN}gcloud run services delete $SERVICE_NAME --region $REGION${NC}"