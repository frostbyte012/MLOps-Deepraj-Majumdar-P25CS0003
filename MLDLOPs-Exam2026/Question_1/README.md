Markdown
# MLDLOps Exam 2026

**Name:** Deepraj Majumdar  
**Roll No:** P25CS0003  
**Branch:** `MLDLOPs-Exam2026`

---

## Question 1: NLP Translation Task (Bengali to English)

This task involves evaluating a pretrained language translation model, containerizing the environment with Docker, and calculating the BLEU score for performance evaluation.

### 1. Model Details
- **Model:** [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en)
- **Task:** Bengali (bn) to English (en) Translation.

### 2. Implementation Files
- `translate.py`: Script to load the model, handle long sequences (truncation), and translate `input.txt` to `output.txt`.
- `evaluate.py`: Script using the `sacrebleu` library to compare `output.txt` against `reference.txt`.
- `Dockerfile`: Containerizes the environment to ensure reproducibility.
- `requirements.txt`: Contains necessary dependencies (`transformers`, `torch`, `sentencepiece`, `sacrebleu`).

### 3. Evaluation Results

* **Q1.6: Output for the First Statement:**
    > "I've got an angel today."

* **Q1.8: Final BLEU Score:**
    > **0.41045** (approx. 41.05)

---

### 4. How to Run

#### **Using Docker**
1. Build the image:
   ```bash
   docker build -t mldlops-q1 .
Run the translation container:

Bash
docker run mldlops-q1
Manual Setup
Install dependencies:

Bash
pip install -r requirements.txt
Run translation:

Bash
python3 translate.py
Run evaluation:

Bash
python3 evaluate.py