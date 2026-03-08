# AreteusML - Project Walkthrough

A complete guide to understanding every part of this project, what the graphs mean, and what to say in interviews.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [The Dataset](#the-dataset)
3. [Phase 1: Baselines](#phase-1-baselines)
4. [Phase 2: ModernBERT Training](#phase-2-modernbert-training)
5. [Phase 3: ONNX Optimization](#phase-3-onnx-optimization)
6. [Phase 3b: FastAPI Serving](#phase-3b-fastapi-serving)
7. [Phase 4: Monitoring & Drift Detection](#phase-4-monitoring--drift-detection)
8. [Phase 5: Dagster Pipelines](#phase-5-dagster-pipelines)
9. [Phase 6: Explainability](#phase-6-explainability)
10. [Reading the Dashboard Graphs](#reading-the-dashboard-graphs)
11. [Results Summary](#results-summary)
12. [Key Decisions Cheat Sheet](#key-decisions-cheat-sheet)
13. [How to Run Everything](#how-to-run-everything)

---

## What This Project Is

A **text classification system** that takes a customer banking query like "How do I change my PIN?" and classifies it into one of **77 intent categories** (like `change_pin`, `card_arrival`, `lost_or_stolen_card`).

It's not just a model - it's the **full ML pipeline**: data ingestion, validation, training, optimization, serving, monitoring, orchestration, explainability, and a dashboard. This is what production ML looks like.

**Tech stack:** Python, PyTorch, HuggingFace Transformers, ONNX Runtime, FastAPI, Streamlit, Dagster, MLflow, Evidently, SHAP, Plotly, Pandera.

---

## The Dataset

**Banking77** - a real benchmark dataset used in industry and research papers.

| Stat | Value |
|------|-------|
| Training samples | 9,158 |
| Test samples | 1,963 |
| Number of classes | 77 |
| Average query length | ~12 words |

**Why 77 classes is hard:** Most text classifiers deal with 2-10 classes. With 77, many classes are semantically very similar:
- "card_arrival" vs "card_delivery_estimate"
- "lost_or_stolen_card" vs "card_not_working"
- "transfer_not_received" vs "balance_not_updated_after_bank_transfer"

A model needs to truly *understand* language to tell these apart, not just match keywords.

---

## Phase 1: Baselines

Before using deep learning, we trained 3 classical ML models to establish a **performance floor**.

| Model | Accuracy | F1 | How It Works |
|-------|----------|----|-------------|
| TF-IDF + SVM | **87.9%** | 0.879 | Convert text to word-frequency vectors, find best separating boundary |
| TF-IDF + Logistic Regression | 84.5% | 0.845 | Same vectors, predict probability per class |
| TF-IDF + Random Forest | 83.9% | 0.839 | Same vectors, ensemble of decision trees |

**What is TF-IDF?** Term Frequency-Inverse Document Frequency. It converts text into numbers by counting how often words appear, weighted by how unique they are. "the" appears everywhere (low weight), "PIN" is specific (high weight). It's a **bag-of-words** approach - it ignores word order entirely.

**Why SVM wins among baselines:** SVMs find the optimal boundary between classes in high-dimensional space. With TF-IDF features (thousands of dimensions), SVMs excel because they handle high-dimensional sparse data well.

**Why 87.9% is the number to beat:** Any deep learning model that can't beat this isn't worth the added complexity, cost, and maintenance. This is our sanity check.

---

## Phase 2: ModernBERT Training

### Why ModernBERT?

| Model | Released | Why Not |
|-------|----------|---------|
| BERT | 2018 | Outdated, no FlashAttention |
| DistilBERT | 2019 | Smaller but weaker on 77 classes |
| DeBERTa-v3 | 2021 | Great accuracy but 2-4x slower |
| **ModernBERT** | **Dec 2024** | **Latest, FlashAttention built-in, 8192 context** |

Picking a December 2024 model shows you follow current research, not just defaulting to 2018 BERT.

### Training Optimizations

Your laptop has a **6GB RTX 3060** - not enough for ModernBERT training. Here's every optimization we used and why:

**1. Adafactor instead of AdamW**
- AdamW (the standard optimizer) stores 2 extra copies of every parameter for momentum tracking (~3GB for ModernBERT)
- Adafactor factorizes the second moment into row and column vectors, cutting optimizer memory to ~1.5GB
- *Interview line: "I chose Adafactor to reduce optimizer memory footprint from ~3GB to ~1.5GB by factorizing the second moment matrix"*

**2. Gradient Checkpointing**
- Normally, all intermediate activations are stored for the backward pass (uses lots of memory)
- Gradient checkpointing throws them away and recomputes them during backprop
- Trades ~30% more compute time for ~60% less memory
- *Interview line: "Gradient checkpointing trades compute for memory - recomputes activations during backprop instead of storing them"*

**3. Batch Size 2 + Gradient Accumulation 32**
- Only 2 samples in GPU memory at once (tiny footprint)
- But gradients are accumulated over 32 mini-batches before updating weights
- Effective batch size = 2 x 32 = 64 (same as if we had a big GPU)
- *Interview line: "Effective batch size of 64 through gradient accumulation, but only 2 samples in memory at any time"*

**4. FP16 Mixed Precision**
- Forward pass uses 16-bit floats (half the memory)
- Backward pass keeps 32-bit for numerical stability
- *Interview line: "Mixed precision training halves activation memory with negligible accuracy impact"*

**5. Class-Weighted Cross-Entropy**
- Banking77 has uneven class sizes (some intents have 30 samples, others have 15)
- Without weighting, the model would ignore rare classes
- We built a custom `WeightedTrainer` that inversely weights by class frequency
- *Interview line: "Custom WeightedTrainer with inverse frequency weighting to prevent the model from ignoring rare intent classes"*

**6. MAX_LENGTH=64**
- Default BERT uses 512 tokens. Banking77 queries average ~12 words
- Setting max_length=64 saves huge memory (attention is O(n^2) with sequence length)
- *Interview line: "Reduced sequence length from 512 to 64 since banking queries are short, cutting attention memory by 64x"*

**Even with all this, 6GB was too tight.** Trained on **Kaggle T4 (16GB VRAM, free tier)** - $0 cost.

### Results

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Accuracy** | **91.3%** | 91.3% of predictions are correct |
| **F1 Macro** | **91.4%** | Average F1 across all 77 classes equally (doesn't favor big classes) |
| F1 Weighted | 91.3% | Average F1 weighted by class size |
| Precision | 91.7% | When it predicts a class, it's right 91.7% of the time |
| Recall | 91.4% | It finds 91.4% of the actual samples for each class |

**+3.4% over the best baseline (SVM at 87.9%).** On 77 classes with high semantic overlap, this is significant. The original Banking77 paper reports ~93% with RoBERTa-large (a much bigger, slower model). We're close with a smaller, faster model.

### Training Progression

How the model improved over 10 epochs:

| Epoch | Eval Loss | Accuracy | F1 Macro | What's Happening |
|-------|-----------|----------|----------|-----------------|
| 1 | 1.440 | 65.0% | 0.652 | Model just started learning, still confused |
| 2 | 0.412 | 88.7% | 0.887 | **Huge jump** - learned most patterns |
| 3 | 0.341 | 90.8% | 0.910 | Refining, almost at peak |
| 4 | 0.303 | 92.3% | 0.924 | Getting the hard cases right |
| 5 | 0.291 | 92.9% | 0.931 | Diminishing returns starting |
| 6 | 0.274 | **93.1%** | **0.933** | **Best eval checkpoint** |
| 7 | 0.280 | 93.0% | 0.932 | Slightly worse - early stopping would trigger here |
| 8-10 | ~0.273 | 93.1% | 0.932-0.933 | Plateaued, no more gains |

**Note:** Eval accuracy (93.1%) is slightly higher than final test accuracy (91.3%) because eval was on the validation split during training. The test set is held out and never seen during training.

---

## Phase 3: ONNX Optimization

The PyTorch model works but is big and slow. For production, we convert to ONNX (Open Neural Network Exchange).

| Format | Model Size | Latency (per query) | Speedup |
|--------|-----------|---------------------|---------|
| PyTorch FP32 | **571 MB** | **45.66 ms** | 1x (baseline) |
| ONNX FP32 | 577 MB | 19.86 ms | 2.3x faster |
| **ONNX INT8** | **146 MB** | **10.11 ms** | **4.52x faster** |

### What Each Format Means

- **PyTorch FP32**: The original model. 32-bit floating point numbers. Full precision, full size.
- **ONNX FP32**: Same precision but optimized compute graph. ONNX Runtime fuses operations, removes redundancy.
- **ONNX INT8**: Quantized to 8-bit integers. Each number uses 1 byte instead of 4. 4x smaller, faster because CPUs process integers faster than floats.

### Why Not GPU Serving?

GPUs cost money. At 10ms per query on CPU, there's no need. If you're serving 100 queries/second, a single CPU instance handles it. GPU serving (TorchServe, Triton) makes sense for multi-model serving at massive scale - overkill for a single classifier.

### Technical Gotchas We Solved

- `optimum` library (the standard ONNX exporter) failed on ModernBERT's LayerNorm, so we used direct `torch.onnx.export` with opset 18
- Had to clear `value_info` before INT8 quantization to fix shape inference errors
- Windows needed `PYTHONIOENCODING=utf-8` for ONNX export (emoji in logs crashed it)

---

## Phase 3b: FastAPI Serving

The API that serves predictions:

| Endpoint | What It Does |
|----------|-------------|
| `POST /predict` | Single text -> label, confidence, latency (~10ms) |
| `POST /predict/batch` | Up to 100 texts at once |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `GET /model/info` | Model metadata (name, classes, backend) |
| `GET /model/health` | Model-specific health check |
| `POST /feedback` | Submit corrections for predictions |
| `GET /feedback/stats` | Feedback statistics |

**Key features:**
- **Redis caching** - same query returns cached result instantly
- **Rate limiting** - 100/min single, 20/min batch (prevents abuse)
- **Singleton model loading** - model loads once at startup, shared across requests
- **Low-confidence flagging** - predictions below 50% confidence get a warning in the response
- **Security headers middleware** - standard HTTP security headers

---

## Phase 4: Monitoring & Drift Detection

**Why monitoring matters:** Models degrade over time. If customers start asking different questions than what the model was trained on, accuracy drops silently. This is called **data drift**.

### What We Built

1. **Reference predictions** - ran the model on the entire test set, saved predictions + confidence scores as the "known good" baseline
2. **Current predictions** - simulated production data with realistic drift:
   - 80% random subset (different volume)
   - 5% of labels randomly changed (simulates misclassifications increasing)
   - Confidence reduced by 0-15% (simulates model becoming less sure)
3. **Evidently drift report** - statistical comparison of reference vs current distributions
4. **Alert system** - three checks:
   - **Data drift**: are the distributions significantly different?
   - **Confidence drop**: is mean confidence falling?
   - **Label shift**: is any single class distribution changing dramatically?
5. **Retrain trigger** - if any alert fires, flag for retraining

### Production Monitoring Integration

The monitoring module is wired directly into the API:

- **PerformanceTracker** records every prediction's latency and confidence to both an in-memory buffer (for real-time windowed metrics) and SQLite (for persistence)
- **AlertManager** checks thresholds after each prediction in a background task
- Every prediction response includes a `prediction_id` (UUID) that enables the feedback loop

Metrics available via `get_metrics_summary(window)`:
- `count`, `latency_mean`, `latency_p50`, `latency_p95`
- `confidence_mean`, `confidence_min`, `throughput_per_sec`

Alert thresholds:
- Confidence below 0.7 -> WARNING
- P95 latency above 100ms -> WARNING
- Error rate above 5% -> CRITICAL
- Data drift detected -> CRITICAL

---

## Phase 5: Dagster Pipelines

Dagster orchestrates the ML pipeline as a **DAG** (Directed Acyclic Graph) - a visual map of what depends on what.

### Training Pipeline
```
raw_data -> validated_data -> augmented_data -> trained_model -> evaluated_model -> registered_model
```

What each step does:
- **raw_data**: Downloads Banking77 from HuggingFace, saves as parquet
- **validated_data**: Pandera schema validation (text is non-empty, labels are 0-76)
- **augmented_data**: Synonym replacement and word swapping for underrepresented classes
- **trained_model**: Fine-tunes ModernBERT (or uses existing_model to skip training)
- **evaluated_model**: Loads metrics from artifacts
- **registered_model**: Registers in MLflow if F1 > 0.90

### Monitoring Pipeline
```
reference_data + current_predictions -> drift_report -> alert_check -> retrain_trigger
```

### Feedback Loop

The dashboard includes a **Feedback** page where users can:
1. Enter a `prediction_id` from any API response
2. Select the correct label from a dropdown of all 77 intent classes
3. Optionally add a comment
4. Submit the correction to `POST /feedback`

Feedback statistics show:
- Total corrections submitted
- Most-corrected classes (indicating where the model struggles)

This data informs retraining decisions - if a specific class accumulates many corrections, it may need more training data or augmentation.

### Why Dagster Over Airflow?

- **Python-native** - define pipelines in Python, not YAML/config
- **Asset-based** - thinks in terms of data artifacts, not tasks. "I want this dataset to exist" vs "run this script"
- **Better for ML** - built-in support for data lineage, incremental computation

---

## Phase 6: Explainability

Three complementary views into how the model makes decisions:

### SHAP (Global Feature Importance)

SHAP (SHapley Additive exPlanations) shows which TF-IDF features matter most across all classes. This is computed on the **SVM baseline** (not ModernBERT) because SHAP works best with traditional ML models.

The SHAP interaction plot in the dashboard shows a matrix of feature interactions - which pairs of words influence predictions together.

### Per-Class Feature Importance

For each of the 77 intent classes, the top words that drive predictions:

| Class | Top Words | Makes Sense? |
|-------|-----------|-------------|
| activate_my_card | "activated", "card", "activate" | Yes - direct keyword match |
| change_pin | "pin", "change", "my" | Yes - the core intent words |
| receiving_money | "card", "my card", "can" | Somewhat - broader context words |

This helps validate the model is learning sensible patterns, not memorizing noise.

### Attention Heatmaps

These show what **ModernBERT specifically focuses on** for each prediction. The heatmap shows attention weight per token.

For "Can you help me activate my card":
- **Darkest (highest attention):** [CLS] token, [SEP] token, "activate", "card"
- **Lightest (lowest attention):** "Can", "you", "help", "me"
- **What this means:** The model correctly focuses on "activate" and "card" (the intent-defining words) and ignores the filler words. This is evidence the model learned meaningful patterns.

The [CLS] and [SEP] tokens always have high attention - that's normal. [CLS] is the token used for classification, so the model naturally routes information there.

---

## Reading the Dashboard Graphs

### Overview Page

**Metric Cards (top row):**
Five cards showing test set performance. The key numbers:
- Accuracy: 91.3% (overall correctness)
- F1 Macro: 91.4% (fairness across all 77 classes)

**Training Loss Chart (left):**
- **X-axis:** Epoch (one pass through the training data)
- **Y-axis:** Loss (how wrong the model is - lower is better)
- **Blue line (Train Loss):** Drops steeply from ~8.9 to near 0. This is the model memorizing training data.
- **Orange line (Eval Loss):** Drops from 1.44 to 0.27. This is real performance on unseen data.
- **What to look for:** Train loss near 0 while eval loss is 0.27 means slight overfitting (model memorized some training noise), but the gap is small enough that it's fine.
- **Good sign:** Eval loss flattens after epoch 6, meaning early stopping at epoch 6 would have been optimal.

**Eval Accuracy Chart (right):**
- **X-axis:** Epoch
- **Y-axis:** Accuracy on validation set
- Big jump from 65% (epoch 1) to 88.7% (epoch 2) - the model "gets it" quickly
- Gradual climb to 93.1% by epoch 6, then plateaus
- **What to look for:** Smooth curve that flattens = healthy training. Jagged or dropping = problem.

**Confusion Matrix:**
- 77x77 grid. X-axis = predicted label, Y-axis = true label
- **Dark diagonal line** = correct predictions (predicted matches true)
- **Off-diagonal dots** = mistakes (predicted wrong class)
- **What ours looks like:** Strong dark diagonal with very few off-diagonal dots. This is good - the model gets most classes right
- **What to look for in interview:** Point to any off-diagonal cluster and say "these two classes get confused because they're semantically similar"

**Per-Class Table:**
- Every row is one of 77 classes with precision, recall, F1-score, and support (number of test samples)
- Look for classes with low F1 - those are the hard ones where similar intents get confused

### Model Comparison Page

**Metric Cards:** One card per model showing accuracy and F1.

**Bar Charts:**
- Left chart: Accuracy comparison. All 4 bars. ModernBERT's bar should be clearly taller.
- Right chart: F1 Macro comparison. Same pattern.
- **What to say:** "SVM is a strong baseline at 87.9%, but ModernBERT pushes 3.4% higher because it understands semantic similarity between intents that bag-of-words can't capture."

**Confusion Matrix Tabs:** Click through each model's confusion matrix. Notice how the baseline models have more off-diagonal noise (more mistakes) compared to ModernBERT.

### Inference Page

Type any banking query and hit Predict:
- **Intent card:** The predicted class name
- **Confidence card:** How sure the model is (0-100%)
- **Latency card:** How long it took (should be ~10ms)
- **Confidence bar:** Visual green/yellow/red indicator
- **Low confidence warning:** Appears if confidence < 50%

**Good demo queries to try:**
- "How do I change my PIN?" -> `change_pin` (high confidence)
- "My card hasn't arrived yet" -> `card_arrival` (high confidence)
- "I want to send money abroad" -> should pick a transfer-related intent
- "asdfghjkl" -> low confidence (model correctly says "I don't know")

### Explainability Page

**SHAP Summary:** Matrix showing feature interactions from the SVM model. Each cell shows how a pair of TF-IDF features interact to influence predictions.

**Per-Class Features:** Dropdown to select any of 77 classes. Shows the top words that drive predictions for that class. Useful for validating the model makes sense.

**Attention Heatmaps:** Grid of 8 sample predictions. Each heatmap shows token-by-token attention weights. Darker = more attention. Look for the model focusing on intent-defining words and ignoring filler.

**Evidently Drift Report:** Interactive HTML report comparing reference predictions vs simulated current predictions. Shows statistical tests for whether distributions have shifted. Green = no drift, red = drift detected.

---

## Results Summary

### The Numbers

| What | Value |
|------|-------|
| Best baseline (SVM) | 87.9% accuracy |
| **ModernBERT** | **91.3% accuracy, 91.4% F1** |
| Improvement over baseline | +3.4% |
| ONNX INT8 latency | 10.11ms per query |
| Speedup over PyTorch | 4.52x |
| Model size reduction | 571MB -> 146MB (4x smaller) |
| Training cost | $0 (Kaggle free tier) |
| Number of classes | 77 |

### Is This Good?

**Yes.** Context:
- Random guess on 77 classes = 1.3% accuracy. We're at 91.3%.
- Original Banking77 paper: ~93% with RoBERTa-large (a bigger, slower model). We're 1.7% behind with a smaller, faster model.
- 3.4% improvement over SVM justifies the deep learning complexity.
- 10ms CPU inference means no GPU costs in production.
- The full pipeline (data -> serving -> monitoring) is what separates "I trained a model" from "I built a production ML system."

---

## Key Decisions Cheat Sheet

Quick reference for interviews - "why did you choose X over Y?"

| Decision | Chose | Over | Why |
|----------|-------|------|-----|
| Model | ModernBERT (Dec 2024) | BERT, DeBERTa | Latest, FlashAttention, shows current research awareness |
| Optimizer | Adafactor | AdamW | Half the optimizer memory (~1.5GB vs ~3GB) |
| Serving format | ONNX INT8 | PyTorch, TorchServe | 4.5x faster, 4x smaller, CPU-only, single file |
| API framework | FastAPI | Flask, Django | Async, auto-docs, Pydantic validation, fastest Python framework |
| Orchestrator | Dagster | Airflow, Prefect | Python-native, asset-based (better for ML), data lineage |
| Dashboard | Streamlit | Dash, Grafana | Fastest to build, Python-native, good enough for ML dashboards |
| Experiment tracking | MLflow | W&B, Neptune | Open source, local-first, no account needed |
| Drift detection | Evidently | Custom | Statistical tests built-in, HTML reports, industry standard |
| Explainability | SHAP + Attention | LIME | SHAP has theoretical guarantees (Shapley values), attention is model-native |
| Data validation | Pandera | Great Expectations | Lighter weight, DataFrame-native, sufficient for this scale |
| Quantization | INT8 | FP16, INT4 | Best latency/accuracy tradeoff. FP16 is minimal gain, INT4 risks accuracy loss |

---

## How to Run Everything

### With Docker

```bash
docker compose up --build
```

This starts 5 services: API (8000), Dashboard (8501), Redis (6379), Prometheus (9090), Grafana (3000).

### One Command (without Docker)

```bash
cd "C:\Projects\Professional\Portfolio Projects\AreteusML"
uv run python scripts/run_all.py
```

This starts all 4 services:

| Service | URL | What It Shows |
|---------|-----|---------------|
| Streamlit | http://localhost:8501 | Dashboard - the main demo (5 pages) |
| FastAPI | http://localhost:8000 | API backend |
| Dagster | http://localhost:3000 | Pipeline DAG visualization |
| MLflow | http://localhost:5000 | Experiment tracking |
| Grafana | http://localhost:3000 | API monitoring (Docker mode only) |

**Note:** Dagster (run_all.py) and Grafana (Docker) both use port 3000. Use one or the other, not both simultaneously.

Press **Ctrl+C** to stop everything. Or from another terminal:
```bash
uv run python scripts/run_all.py stop
```

### Generate Artifacts (if needed)

If the data files are missing, regenerate them:
```bash
uv run python scripts/generate_reference_predictions.py
uv run python scripts/generate_current_predictions.py
uv run python scripts/generate_drift_report.py
uv run python -m ml.explainability.attention_viz
```

### Interview Demo Flow (2 minutes)

1. Run `python scripts/run_all.py`
2. Open **Streamlit** (localhost:8501)
   - Overview: "91.3% accuracy on 77 classes, here are the training curves"
   - Model Comparison: "3.4% over SVM baseline, here's why deep learning was worth it"
   - Inference: Type a query, show real-time prediction at 10ms
   - Explainability: "The model focuses on intent-defining words, not filler"
3. Open **Dagster** (localhost:3000): "Full pipeline from data ingestion to monitoring"
4. Open **MLflow** (localhost:5000): "Every experiment tracked with metrics"
5. Close: "Built for $0 on Kaggle, serves on CPU at 10ms, full production pipeline"
