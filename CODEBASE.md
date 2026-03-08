# AreteusML Interview Cheat Sheet

## Q&A

### Why ModernBERT?

Released December 2024, it's the most current encoder-only transformer. Built-in FlashAttention gives 2-4x speedup over DeBERTa-v3 with comparable accuracy on classification tasks. 8192 context window future-proofs for longer inputs. Picking it over DistilBERT or DeBERTa shows awareness of current research, not just defaulting to 2020-era models. Training required 16GB VRAM (Kaggle T4) due to the 77-class head and gradient checkpointing overhead, but inference runs on CPU with ONNX.

### Why ONNX over TorchServe?

The goal was CPU-optimized inference with minimal operational complexity. ONNX Runtime with INT8 quantization hits <10ms latency on CPU, eliminating GPU cost in production. TorchServe requires a model archiver, config files, and a Java-based server process. Triton is the right call for multi-model GPU serving at scale, but overkill for a single classifier. ONNX gives a single portable file that runs anywhere with a pip install. The INT8 model is 146MB vs 575MB for PyTorch, with negligible accuracy loss.

### Why Adafactor over AdamW?

AdamW stores two momentum states per parameter (~3GB for ModernBERT-base). On a 6GB GPU that's already tight with model weights, gradients, and activations. Adafactor factorizes the second moment into row and column vectors, cutting optimizer memory to ~1.5GB. We combined this with gradient checkpointing, batch size 2 with gradient accumulation to 32, and FP16 to fit training on constrained hardware. Ultimately trained on Kaggle T4 (16GB VRAM) for stability, but the optimizations would enable training on 8GB GPUs.

### Why class-weighted cross-entropy?

Banking77 has uneven class distribution across 77 intents. Without weighting, the model would overfit to frequent classes and underperform on rare ones. sklearn's compute_class_weight("balanced") inversely weights by class frequency. We pass these weights to a custom WeightedTrainer that overrides Trainer.compute_loss with a weighted CrossEntropyLoss. This improved F1 macro (which equally weights all 77 classes) without hurting overall accuracy.

### When NOT to use deep learning?

When you have <1K labeled examples, tabular data with clear feature engineering, or strict latency/cost constraints. For Banking77 with 13K examples across 77 classes, deep learning is justified because the semantic similarity between classes (e.g., "card_arrival" vs "card_delivery_estimate") requires understanding language, not just keyword matching. Our TF-IDF + SVM baseline hit 87.9%, which is respectable but plateaus because bag-of-words can't capture semantic nuance. ModernBERT pushed to 91.3% by understanding context.

### How do you handle training-serving skew?

The ONNX export freezes the tokenizer and model together, so there's no version mismatch between training and serving. Text preprocessing is minimal by design (the tokenizer handles everything), reducing surface area for skew. Same max_length=64, same tokenizer config, same label mapping. Pandera schemas validate data shape and types at ingestion time. MLflow tracks the exact training parameters and data splits for reproducibility.

### How does the API serving work?

FastAPI with ONNX Runtime as the inference backend. Single prediction endpoint (~10ms latency), batch prediction, model health check, and model info endpoints. Redis caching for repeated queries. Rate limiting via slowapi. The model loads once at startup via a singleton pattern. Low-confidence predictions (below threshold) are flagged for human review in the response. Docker Compose packages the API + Redis for one-command deployment. A Streamlit dashboard provides the frontend: model overview with training curves, baseline vs ModernBERT comparison, live inference, and explainability (SHAP, attention heatmaps, Evidently drift reports).

### How would you scale this?

Horizontally: ONNX Runtime supports batched inference, so you scale FastAPI pods behind a load balancer. For more classes or domains, you'd move to a shared embedding layer with per-domain classification heads. If latency budget allows, swap to ModernBERT-large for accuracy gains. For real-time retraining, add a feature store and event-driven pipeline triggers. The current architecture (stateless API + Redis cache) is already horizontally scalable.

### What would you do differently with more time/budget?

Implement A/B testing infrastructure to compare model versions in production. Build a feedback loop where low-confidence predictions get routed to human review and become training data. Explore contrastive learning or SetFit for few-shot classes. Add Triton Inference Server if multi-model serving becomes necessary. Deploy the full stack to cloud (currently runs locally with a single `python scripts/run_all.py` command). Add PostgreSQL-backed prediction logging for production monitoring instead of local parquet files.

### What were the biggest challenges?

Training crashed my laptop 4 times before I moved to Kaggle. First attempt: batch_size=16, no gradient checkpointing, instant OOM. Second: batch_size=8 with huge checkpoint saves filling RAM. Third: a game competing for VRAM. Fourth: even with all optimizations (Adafactor, batch=2, grad_accum=16, gradient checkpointing), 6GB was too tight. Lesson: know your hardware budget before committing to a training strategy. ONNX export also required debugging: optimum library failed on ModernBERT's LayerNorm, so I switched to direct torch.onnx.export with opset 18 and had to clear value_info before INT8 quantization to fix shape inference errors.

### How does monitoring work?

Every prediction records latency and confidence to an in-memory buffer (for real-time windowed metrics) and SQLite (data/monitoring.db) for persistence. The PerformanceTracker uses SQLAlchemy with the same fallback pattern as the audit service. An AlertManager checks thresholds after each prediction: confidence drops below 0.7, p95 latency exceeds 100ms, or error rate spikes above 5%. Alerts are logged to monitoring/reports/alerts.jsonl. The API returns a prediction_id (uuid4) with every response, enabling the feedback loop where users can submit corrections tied to specific predictions.

### How does the feedback loop work?

Each /predict response includes a prediction_id. Users submit corrections via POST /feedback with the prediction_id and correct label. Feedback is stored in SQLite via the audit service (FeedbackLog table). The dashboard has a dedicated Feedback page where users can submit corrections (with a dropdown of all 77 label names) and view statistics on most-corrected classes. This data feeds into retraining decisions - high correction volume on specific classes signals the model needs improvement there.

### How does Docker deployment work?

Docker Compose orchestrates 5 services: Redis (caching), API (FastAPI + ONNX), Dashboard (Streamlit), Prometheus (metrics scraping), and Grafana (dashboards). The API falls back to SQLite when PostgreSQL isn't available. Model files are mounted read-only. Artifacts are mounted into the dashboard container for reading training metrics. The Grafana dashboard shows request rate, latency percentiles, error rate, and confidence distribution. One command: docker compose up --build.

### What about DVC?

DVC (Data Version Control) tracks large files that don't belong in git: ml/models/, data/, and artifacts/. Configured with a local remote at /tmp/dvc-storage. This means model checkpoints, training data, and generated artifacts are versioned separately from code. In production you'd point DVC to S3 or GCS.

## Architecture Walkthrough (2-Minute Demo Flow)

Run `python scripts/run_all.py` to start all services, then walk through:

1. **Start with the problem**: "Banking77 has 77 intent classes with high semantic overlap. Customers say similar things meaning different things."
2. **Show the dashboard overview** (localhost:8501): Training curves, 91.3% accuracy, confusion matrix, per-class metrics - all loaded from real artifacts.
3. **Show model comparison**: "TF-IDF + SVM gets 87.9%. ModernBERT pushes to 91.3%. Here's why." Side-by-side bar charts and confusion matrices.
4. **Show live inference**: Type a query in the dashboard, get prediction with confidence and latency. "Under 10ms on CPU with ONNX INT8."
5. **Show explainability**: SHAP feature importance, attention heatmaps showing which tokens drive predictions, Evidently drift report.
6. **Show the pipeline** (localhost:3000): Dagster asset graph - data loading, validation, augmentation, training, evaluation, monitoring with drift detection.
7. **Show experiment tracking** (localhost:5000): MLflow with logged metrics and model versions.
8. **Close**: "Pandera validates data, MLflow tracks experiments, ONNX serves on any CPU, Dagster orchestrates the pipeline, Streamlit makes it demo-ready. Built on a $0 budget with a 6GB laptop GPU and free Kaggle T4."

## Topics I've Studied

### Recommendation Systems
Collaborative filtering (matrix factorization, ALS), content-based filtering, hybrid approaches. Two-tower models for retrieval. Cold start strategies. Evaluation: NDCG, MAP, recall@k. Production patterns: candidate generation -> ranking -> re-ranking.

### Ranking and Search
Learning to rank (pointwise, pairwise, listwise). BM25 as baseline. Semantic search with dense retrieval (bi-encoders) and cross-encoder reranking. Approximate nearest neighbors (HNSW, IVF). Reciprocal rank fusion for hybrid search.

### Computer Vision
CNN architectures (ResNet, EfficientNet) and why they work (skip connections, compound scaling). Vision Transformers and patch embeddings. Transfer learning: freeze backbone, fine-tune head. Data augmentation as regularization. Object detection pipeline: backbone -> neck -> head.

### Time Series
Stationarity and differencing. ARIMA family as baselines. Seq2seq with attention for multi-step forecasting. Temporal Fusion Transformers for interpretable forecasting. Feature engineering: lags, rolling stats, calendar features. Walk-forward validation, never random splits.

### Online Learning
Concept drift detection (ADWIN, DDM). Model update strategies: full retrain vs incremental. Hoeffding trees for streaming classification. Reservoir sampling for maintaining representative buffers. Production pattern: shadow mode -> gradual rollout -> full deployment.

### Distributed Training
Data parallelism (DDP) vs model parallelism. Gradient synchronization: all-reduce, ring-allreduce. Mixed precision training (FP16 forward, FP32 gradients). DeepSpeed ZeRO stages for memory optimization. Communication overhead as the real bottleneck, not compute.

### Feature Stores
Offline store (batch features for training) vs online store (low-latency serving). Point-in-time correctness to prevent future leakage. Feast architecture: registry, offline store (BigQuery/Parquet), online store (Redis/DynamoDB). Feature engineering as a shared service across teams.

### ML Math Fundamentals
Cross-entropy loss and why it works for classification (KL divergence connection). Gradient descent variants: SGD momentum, Adam (adaptive learning rates per parameter). Attention mechanism: scaled dot-product, why we divide by sqrt(d_k). Regularization: L1 (sparsity), L2 (weight decay), dropout (ensemble approximation). Bias-variance tradeoff: underfitting vs overfitting, and how model complexity / data size shifts the curve.
