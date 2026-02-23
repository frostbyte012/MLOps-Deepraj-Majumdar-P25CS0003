# Assignment 3: HuggingFace Fine-Tuning & Docker Deployment

Fine-tuning **DistilBERT** to classify Goodreads book reviews into 8 genres.

## Model
**`distilbert-base-cased`** — 40% smaller and 60% faster than BERT while retaining 97% performance. Cased variant preserves capitalization important for book review text.

🤗 **HuggingFace:** `https://huggingface.co/frostbyte012/distilbert-goodreads-genre`

## Project Structure
```
├── src/
│   ├── data.py       # Data loading & preprocessing
│   ├── utils.py      # Metrics & helpers
│   ├── train.py      # Fine-tuning script
│   └── eval.py       # Evaluation script
├── Dockerfile        # Training container
├── Dockerfile.eval   # Production eval container (Task 9)
├── .dockerignore
├── requirements.txt
└── run_pipeline.sh   # End-to-end pipeline
```

## Quick Start
```bash
# Train + push to HuggingFace Hub
python src/train.py --epochs 3 --batch_size 10 --push_to_hub \
    --hf_token YOUR_TOKEN --hub_model_id YOUR_USERNAME/distilbert-goodreads-genre

# Evaluate from Hub
python src/eval.py --model_path YOUR_USERNAME/distilbert-goodreads-genre --from_hub
```

## Docker
```bash
# Full pipeline (trains locally with GPU, eval via Docker)
./run_pipeline.sh

# Eval-only container (Task 9)
docker build --build-arg HF_MODEL_ID=YOUR_USERNAME/distilbert-goodreads-genre \
    -t goodreads-eval -f Dockerfile.eval .
docker run -v $(pwd)/results:/app/results goodreads-eval
```

## Results

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| TF-IDF + Logistic Regression (baseline) | ~0.71 | ~0.70 |
| DistilBERT (local) | ~0.82 | ~0.82 |
| DistilBERT (from HuggingFace Hub) | ~0.82 | ~0.82 |

## Training Config
| Parameter | Value |
|-----------|-------|
| Base model | distilbert-base-cased |
| Epochs | 3 |
| Batch size | 10 |
| Learning rate | 5e-5 |
| Train samples | ~6,400 |
| Test samples | ~1,600 |