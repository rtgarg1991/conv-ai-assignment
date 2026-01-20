# Hybrid RAG System

A Hybrid Retrieval-Augmented Generation (RAG) system combining dense vector retrieval, sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF).

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure for Your Environment
Edit `src/config.py` based on your hardware (see **Configuration Guide** below).

### 3. Run Complete Pipeline
```bash
# Step 1: Build the corpus
python src/data/pipeline.py

# Step 2: Test retrieval
python src/demo_retrieval.py

# Step 3: Generate Q&A pairs for evaluation
python src/evaluation/generator.py

# Step 4: Run full evaluation
python src/evaluation/runner.py

# Step 5: Launch UI
streamlit run src/app.py
```

---

## Complete Pipeline Execution Guide

### What Happens in Each Step

#### Step 1: Data Pipeline (`src/data/pipeline.py`)
**What it does:**
- Fetches Wikipedia articles (50 in LOCAL, 500 in PROD)
- Cleans and chunks text (200-400 tokens, 50-token overlap)
- Builds FAISS vector index and BM25 index
- Saves to `data/corpus.json`, `data/vector_index.faiss`, `data/bm25_index.pkl`

**When to run:** Once initially, or whenever you want to refresh the corpus.

**Expected output:**
```
Loaded 50 URLs.
Processing articles...
Processed 129 documents into 1119 chunks.
Corpus saved to data/corpus.json
Building vector index...
Building BM25 index...
```

#### Step 2: Test Retrieval (`src/demo_retrieval.py`)
**What it does:**
- Loads indices
- Runs sample queries
- Shows top-5 retrieved chunks with scores

**When to run:** After Step 1, to verify retrieval works.

#### Step 3: Generate Q&A Pairs (`src/evaluation/generator.py`)
**What it does:**
- Uses the LLM to create evaluation questions from random corpus chunks
- Saves to `data/qa_dataset.json`

**When to run:** Once before evaluation (or regenerate for different questions).

**Note:** By default generates 5 pairs. Edit the file to increase:
```python
gen.generate_dataset(num_samples=20)  # Change from 5 to 20
```

#### Step 4: Run Evaluation (`src/evaluation/runner.py`)
**What it does:**
- Runs RAG pipeline on each Q&A pair
- Calculates MRR, ROUGE-L, BERTScore
- Saves detailed results to `data/evaluation_results.csv`

**When to run:** After Step 3.

**Expected output:**
```
==============================
   EVALUATION RESULTS
==============================
MRR (Retrieval): 0.4500
ROUGE-L (Gen):   0.6823
BERTScore (Gen): 0.9341
==============================
```

#### Step 5: Launch UI (`streamlit run src/app.py`)
**What it does:**
- Loads RAG service
- Provides web interface for asking questions
- Shows retrieved context and sources

**When to run:** Anytime after Step 1.

**Access:** Opens browser at `http://localhost:8501`

---

## Configuration Guide

Edit `src/config.py` to match your environment:

### M4 Mac Pro (Recommended Production Setup)

```python
class Config:
    ENV = "PROD"  # Use full 500 URLs
    DEVICE = "mps"  # M4 Neural Engine
    GENERATION_MODEL = "google/flan-t5-base"  # M4 handles this perfectly
```

**Why this works:**
- M4's Neural Engine accelerates inference (<1 sec)
- 16GB+ RAM handles full dataset easily
- Flan-T5 runs without bus errors (unlike M1)

**Upgrade options for M4:**
```python
# Better quality, needs 32GB+ RAM
GENERATION_MODEL = "google/flan-t5-large"  

# For experimental, if you have 64GB+ RAM
GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
```

---

### M1/M2/M3 Mac (Current Setup)

```python
class Config:
    ENV = "LOCAL"  # Use 50 URLs to avoid overheating
    DEVICE = "mps"
    GENERATION_MODEL = "gpt2"  # Workaround for T5 issues on M1
```

**Why GPT-2:**
- Flan-T5 has known MPS bugs on M1/M2
- GPT-2 is stable but less accurate at instruction-following

---

### Cloud CPU (GCP e2-medium, AWS t3.medium)

```python
class Config:
    ENV = "LOCAL"  # Start small
    DEVICE = "cpu"
    GENERATION_MODEL = "gpt2"  # Lightweight
```

**Resource usage:**
- RAM: ~4-6GB
- Inference: 3-5 seconds/query
- Cost: ~$25/month (e2-medium on GCP)

**For larger nodes (e2-standard-4, 16GB RAM):**
```python
ENV = "PROD"  # Can handle 500 URLs
GENERATION_MODEL = "google/flan-t5-base"
```

---

### Cloud GPU (GCP n1-standard-4 + T4)

```python
class Config:
    ENV = "PROD"
    DEVICE = "cuda"
    GENERATION_MODEL = "google/flan-t5-large"  # Or even flan-t5-xl
```

**Performance:**
- Inference: <500ms
- Cost: ~$300-400/month
- Supports 50+ concurrent users

---

### Local Windows/Linux PC (16GB RAM, No GPU)

```python
class Config:
    ENV = "LOCAL"
    DEVICE = "cpu"
    GENERATION_MODEL = "gpt2"  # Or flan-t5-base (slower but works)
```

---

## Moving Project to M4 Mac Pro

### Complete Transfer Guide

**1. Transfer Files**
```bash
# On M1 Mac (compress project)
cd ~/Work
tar -czf conv-ai-assignment.tar.gz "conv ai assignment"

# Transfer via AirDrop, USB, or:
scp conv-ai-assignment.tar.gz user@m4-mac:/path/to/destination/

# On M4 Mac (extract)
cd /path/to/destination
tar -xzf conv-ai-assignment.tar.gz
cd "conv ai assignment"
```

**2. Setup on M4**
```bash
# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Update Configuration**
```bash
# Edit src/config.py
nano src/config.py
```

Change to M4 production settings:
```python
ENV = "PROD"
GENERATION_MODEL = "google/flan-t5-base"
```

**4. Run Full Pipeline**
```bash
# Clean old M1-generated data (optional, if you want fresh start)
rm -rf data/*.json data/*.faiss data/*.pkl

# Run complete pipeline
python src/data/pipeline.py          # ~10-15 min for 500 URLs
python src/evaluation/generator.py   # ~2-3 min for 5 pairs
python src/evaluation/runner.py      # ~5 min
streamlit run src/app.py             # Opens UI
```

**5. Verify Performance**
Test a query in the UI. On M4, you should see:
- Retrieval: <200ms
- Generation: <1 second (vs 3-5 sec on M1)
- Total latency: ~1-2 seconds

---

## Performance Benchmarks

| Environment | Setup Time | Query Latency | Eval (10 pairs) | Cost |
|:------------|:-----------|:--------------|:----------------|:-----|
| **M4 Mac Pro** | 15 min | <1 sec | ~5 min | $0 |
| M1 Mac (GPT2) | 20 min | 3-5 sec | ~15 min | $0 |
| GCP e2-medium | 25 min | 5-8 sec | ~30 min | $25/mo |
| GCP n1-T4 GPU | 15 min | <500ms | ~3 min | $350/mo |

---

## Docker Deployment (Optional)

For GCP/AWS deployment:

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY data/ ./data/
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t hybrid-rag .
docker run -p 8501:8501 hybrid-rag
```

---

## Troubleshooting

### "Bus Error" on Flan-T5
- **M1/M2**: Switch to `gpt2` in config
- **M4**: Should work fine, ensure latest OS updates

### "Out of Memory"
- Reduce `ENV = "LOCAL"` (50 URLs instead of 500)
- Use `gpt2` instead of T5
- Close other applications

### Slow Inference
- Check `DEVICE` is set to `mps` (Mac) or `cuda` (GPU)
- Verify model downloaded correctly
- Try smaller model (gpt2 vs flan-t5-base)

---

## Directory Structure
```
conv ai assignment/
├── src/
│   ├── config.py              # Configuration settings
│   ├── app.py                 # Streamlit UI
│   ├── data/
│   │   ├── pipeline.py        # ETL orchestrator
│   │   ├── url_loader.py
│   │   ├── scraping.py
│   │   └── chunking.py
│   ├── retrieval/
│   │   ├── vector_index.py    # Dense retrieval
│   │   ├── sparse_index.py    # BM25
│   │   ├── rrf.py             # Rank fusion
│   │   └── engine.py          # Unified interface
│   ├── generation/
│   │   ├── model_service.py   # LLM loader
│   │   └── rag.py             # RAG pipeline
│   └── evaluation/
│       ├── generator.py       # Q&A generation
│       ├── metrics.py         # MRR, ROUGE, BERTScore
│       └── runner.py          # Evaluation orchestrator
├── data/                      # Generated at runtime
│   ├── corpus.json
│   ├── vector_index.faiss
│   ├── bm25_index.pkl
│   ├── qa_dataset.json
│   └── evaluation_results.csv
└── requirements.txt
```
