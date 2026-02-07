# Hybrid RAG System

A Hybrid Retrieval-Augmented Generation (RAG) system combining dense vector retrieval, sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF). The project includes an automated evaluation pipeline with 100 Q&A pairs, URL-level MRR, additional metrics, ablation studies, error analysis, and an HTML report.

**Quick Start**
1. Setup environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure your environment
Edit `/Users/rohitgarg/Work/conv ai assignment/src/config.py` for device and model settings.

3. Run the full pipeline
```bash
# Step 1: Build the corpus (fixed + random URLs)
python /Users/rohitgarg/Work/conv ai assignment/src/data/pipeline.py

# Step 2: Test retrieval (optional)
python /Users/rohitgarg/Work/conv ai assignment/src/demo_retrieval.py

# Step 3: Generate Q&A pairs (100)
python /Users/rohitgarg/Work/conv ai assignment/src/evaluation/generator.py

# Step 4: Run full evaluation (MRR, ROUGE-L, BERTScore + per-question)
python /Users/rohitgarg/Work/conv ai assignment/src/evaluation/runner.py

# Step 5: Run ablation studies
python /Users/rohitgarg/Work/conv ai assignment/src/evaluation/ablation.py

# Step 6: Run error analysis
python /Users/rohitgarg/Work/conv ai assignment/src/evaluation/error_analysis.py

# Step 7: Generate HTML report
python /Users/rohitgarg/Work/conv ai assignment/src/evaluation/report_generator.py

# Step 8: Launch UI
streamlit run /Users/rohitgarg/Work/conv ai assignment/src/app.py
```

---

**Pipeline Details**

**Step 1: Data Pipeline** (`/Users/rohitgarg/Work/conv ai assignment/src/data/pipeline.py`)
- Loads fixed URLs (200) from `data/fixed_urls.json`.
- Samples random URLs (300) using Wikipedia categories + Random API.
- Scrapes, cleans, and chunks text (200-400 tokens, 50 overlap).
- Saves corpus to `data/corpus.json`.

**Step 2: Retrieval Demo** (`/Users/rohitgarg/Work/conv ai assignment/src/demo_retrieval.py`)
- Loads or builds FAISS and BM25 indices.
- Executes sample queries and prints top results.

**Step 3: Q&A Generation** (`/Users/rohitgarg/Work/conv ai assignment/src/evaluation/generator.py`)
- Generates 100 Q&A pairs from the corpus.
- Includes question types: factual, comparative, inferential, multi-hop.
- Outputs `data/qa_dataset.json`.

**Step 4: Evaluation** (`/Users/rohitgarg/Work/conv ai assignment/src/evaluation/runner.py`)
- Runs RAG end-to-end per question.
- Metrics:
  - URL-level MRR (mandatory)
  - ROUGE-L
  - BERTScore
- Outputs:
  - `data/evaluation_results.csv` (per-question metrics)
  - `data/evaluation_summary.json`

**Step 5: Ablation Studies** (`/Users/rohitgarg/Work/conv ai assignment/src/evaluation/ablation.py`)
- Compares dense-only, sparse-only, and hybrid variants.
- Outputs `data/ablation_results.json`.

**Step 6: Error Analysis** (`/Users/rohitgarg/Work/conv ai assignment/src/evaluation/error_analysis.py`)
- Categorizes failures and provides recommendations.
- Outputs `data/error_analysis.json`.

**Step 7: HTML Report** (`/Users/rohitgarg/Work/conv ai assignment/src/evaluation/report_generator.py`)
- Generates a complete report with:
  - Summary metrics
  - Metric justifications
  - Ablation results
  - Error analysis
  - Retrieval heatmap
- Output: `data/evaluation_report.html`

**Step 8: UI** (`/Users/rohitgarg/Work/conv ai assignment/src/app.py`)
- Streamlit UI with retrieval scores, sources, and latency breakdown.

---

**Configuration Guide**
Edit `/Users/rohitgarg/Work/conv ai assignment/src/config.py`:
- `ENV`: `PROD` uses 200 fixed + 300 random URLs.
- `DEVICE`: `mps`, `cuda`, or `cpu`.
- `EMBEDDING_MODEL_NAME`: sentence-transformers model.
- `GENERATION_MODEL`: `google/flan-t5-base` by default.
- `TOP_N_RETRIEVAL`: number of final RRF chunks (currently 10).
- `RRF_K`: RRF constant (default 60).
- `RRF_WEIGHT_DENSE` / `RRF_WEIGHT_SPARSE`: weighting for fusion.

---

**Key Outputs**
- `data/fixed_urls.json`: 200 fixed URLs.
- `data/corpus.json`: processed chunks with metadata.
- `data/vector_index.faiss`: dense index (built on first retrieval).
- `data/bm25_index.pkl`: sparse index (built on first retrieval).
- `data/qa_dataset.json`: 100 Q&A pairs with question types.
- `data/evaluation_results.csv`: per-question metrics.
- `data/evaluation_summary.json`: aggregate metrics.
- `data/ablation_results.json`: ablation metrics.
- `data/error_analysis.json`: error analysis.
- `data/evaluation_report.html`: full HTML report.

---

**Troubleshooting**
- **Model download errors**: Ensure internet access for Hugging Face models or pre-cache them.
- **MPS issues on older Macs**: Use `DEVICE = "cpu"` and a smaller model.
- **Slow inference**: Reduce model size or use `cuda` if available.

---

**Directory Structure**
```
conv ai assignment/
├── src/
│   ├── config.py
│   ├── app.py
│   ├── demo_retrieval.py
│   ├── data/
│   │   ├── pipeline.py
│   │   ├── url_loader.py
│   │   ├── scraping.py
│   │   ├── chunking.py
│   │   └── validate_urls.py
│   ├── retrieval/
│   │   ├── vector_index.py
│   │   ├── sparse_index.py
│   │   ├── rrf.py
│   │   └── engine.py
│   ├── generation/
│   │   ├── model_service.py
│   │   └── rag.py
│   └── evaluation/
│       ├── generator.py
│       ├── metrics.py
│       ├── runner.py
│       ├── ablation.py
│       ├── error_analysis.py
│       └── report_generator.py
├── data/
│   ├── fixed_urls.json
│   ├── corpus.json
│   ├── vector_index.faiss
│   ├── bm25_index.pkl
│   ├── qa_dataset.json
│   ├── evaluation_results.csv
│   ├── evaluation_summary.json
│   ├── ablation_results.json
│   ├── error_analysis.json
│   └── evaluation_report.html
└── requirements.txt
```
