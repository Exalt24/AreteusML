ARETEUSML INTERVIEW CHEAT SHEET


Q&A


WHY MODERNBERT?

Released December 2024, so it's the newest encoder-only transformer available. I evaluated three main options for text classification on Banking77:

DeBERTa-v3 was the safe pick since it consistently tops GLUE and SuperGLUE benchmarks, and a 2025 paper from ACL confirmed it still edges out ModernBERT on raw classification accuracy in some tasks. But DeBERTa's disentangled attention mechanism is computationally heavier, and it actually struggled on text retrieval tasks in that same study, which tells me its architecture is less general-purpose. Training and inference are noticeably slower.

DistilBERT would have been fine for a simpler problem, but it's a 2019 model with a 512 token limit and no architectural innovations since. Using it in 2025 signals "I grabbed the first thing from a tutorial," not "I evaluated options."

SetFit was genuinely tempting since it achieves 92.7% on IMDB with just 8 samples per class using contrastive learning. For few-shot scenarios it's incredible. But Banking77 has 13K examples, so we're not in few-shot territory. SetFit's contrastive approach also doesn't scale as cleanly to 77 classes.

ModernBERT's built-in FlashAttention gives 2-4x speedup over DeBERTa with comparable accuracy on classification. The 8192 context window future-proofs for longer inputs. Training required 16GB VRAM (Kaggle T4) due to the 77-class head and gradient checkpointing overhead, but inference runs on CPU with ONNX.

Limitation I'd acknowledge: if I needed absolute peak accuracy and had budget for a bigger GPU, DeBERTa-v3-large would probably win by 1-2 points. ModernBERT was the right tradeoff between performance, speed, and showing awareness of current research. If the project were few-shot (say, 50 labeled examples per class), I'd switch to SetFit with ModernBERT embeddings since that combination is showing really strong results in 2025 benchmarks.


WHY ONNX OVER TORCHSERVE?

The goal was CPU-optimized inference with minimal operational complexity. I looked at four options:

TorchServe requires a model archiver step, config files, and runs a Java-based server process. It's the official PyTorch serving solution, but the operational overhead is significant for a single model. You're basically running a JVM to serve a Python model.

TensorRT would give the absolute best latency on NVIDIA GPUs, sometimes 2-3x faster than ONNX on GPU workloads. It achieves this through aggressive kernel fusion and hardware-specific optimization. But it locks you into NVIDIA hardware, the optimization step can take 30+ minutes, and the serialized engine isn't portable across GPU architectures. For a project targeting CPU inference, TensorRT makes zero sense.

OpenVINO is Intel's answer, optimized for Intel CPUs and their discrete GPUs. If I knew the deployment target was always Intel hardware, OpenVINO could squeeze out an extra 10-20% over ONNX Runtime on CPU. But that's a big assumption, and the ecosystem is smaller.

ONNX Runtime with INT8 quantization hits under 10ms latency on CPU, eliminating GPU cost in production. It gives a single portable file that runs anywhere with a pip install. The INT8 model is 146MB vs 575MB for PyTorch, with negligible accuracy loss. The industry is actually converging on ONNX as the interoperability standard, with both TensorRT and OpenVINO available as ONNX Runtime execution providers. So ONNX is the universal middle ground.

Triton Inference Server is the right call for multi-model GPU serving at scale (it supports dynamic batching, model ensembles, concurrent model execution), but it's overkill for a single classifier.

Trade-off I accept: ONNX Runtime on CPU will never match TensorRT on a V100 for raw throughput. If this were a high-throughput production system doing millions of predictions per hour, I'd use ONNX Runtime with the TensorRT execution provider to get both portability and GPU acceleration. For this project's scale, pure CPU inference at under 10ms is more than sufficient and keeps infrastructure costs at zero.


WHY ADAFACTOR OVER ADAMW?

AdamW stores two momentum states per parameter, roughly 3GB for ModernBERT-base. On a 6GB GPU that's already tight with model weights, gradients, and activations. Adafactor factorizes the second moment into row and column vectors, cutting optimizer memory to roughly 1.5GB. We combined this with gradient checkpointing, batch size 2 with gradient accumulation to 32, and FP16 to fit training on constrained hardware.

The alternatives I considered: 8-bit Adam (bitsandbytes) quantizes optimizer states and would save similar memory, but adds a dependency and can introduce numerical instability in some cases. LORA/QLoRA would freeze most parameters and only train adapters, dramatically reducing memory, but for a classification fine-tune on a base model I wanted full parameter updates to get the best accuracy. LORA makes more sense for adapting already-instruction-tuned models.

Ultimately trained on Kaggle T4 (16GB VRAM) for stability, but the optimizations would enable training on 8GB GPUs. The lesson: Adafactor is the right default when you're memory-constrained and doing full fine-tuning. It's not free though. Adafactor can be less stable than AdamW, especially with small batch sizes, because the factorized approximation introduces noise. The gradient accumulation to effective batch size 32 helps compensate for that.

Scalability note: at larger scale (ModernBERT-large, multi-GPU), you'd switch back to AdamW with DeepSpeed ZeRO Stage 2, which shards optimizer states across GPUs. Adafactor is a single-GPU memory optimization, not a distributed training strategy.


WHY CLASS-WEIGHTED CROSS-ENTROPY?

Banking77 has uneven class distribution across 77 intents. Without weighting, the model would overfit to frequent classes and underperform on rare ones. sklearn's compute_class_weight("balanced") inversely weights by class frequency. We pass these weights to a custom WeightedTrainer that overrides Trainer.compute_loss with a weighted CrossEntropyLoss. This improved F1 macro (which equally weights all 77 classes) without hurting overall accuracy.

Alternatives I evaluated: focal loss down-weights easy examples and focuses on hard ones, which can help with extreme imbalance. But Banking77's imbalance is moderate, not extreme (it's not like 99/1 fraud detection), so focal loss would add complexity without proportional benefit. Oversampling (SMOTE for text via back-translation or paraphrasing) was another option, but augmenting text data is noisier than augmenting tabular data. Class weighting is the simplest intervention that actually moves the metric I care about (F1 macro).

Limitation: class weighting assumes all misclassifications within a class are equally costly. In reality, confusing "card_arrival" with "card_delivery_estimate" is much less severe than confusing "card_arrival" with "terminate_account." A cost-sensitive loss matrix would handle this, but requires domain expertise to define the cost structure, which wasn't available for a portfolio project.


WHEN NOT TO USE DEEP LEARNING?

When you have under 1K labeled examples, tabular data with clear feature engineering, or strict latency/cost constraints. For Banking77 with 13K examples across 77 classes, deep learning is justified because the semantic similarity between classes (e.g., "card_arrival" vs "card_delivery_estimate") requires understanding language, not just keyword matching. Our TF-IDF + SVM baseline hit 87.9%, which is respectable but plateaus because bag-of-words can't capture semantic nuance. ModernBERT pushed to 91.3% by understanding context.

Honest reflection: that 3.4 percentage point improvement cost significantly more compute, complexity, and development time. In a production setting, the business value of those 3.4 points determines whether the complexity is justified. For a customer-facing banking chatbot routing to the wrong department, each misclassification is a frustrated customer, so those points matter. For an internal analytics tool, maybe the SVM is good enough.

The baseline also serves a diagnostic purpose. If the SVM had hit 95%, that would signal the problem is mostly lexical, and deep learning would add complexity without proportional gain. The 87.9% ceiling with a reasonable gap tells me there's semantic signal the SVM can't capture.


HOW DO YOU HANDLE TRAINING-SERVING SKEW?

The ONNX export freezes the tokenizer and model together, so there's no version mismatch between training and serving. Text preprocessing is minimal by design (the tokenizer handles everything), reducing surface area for skew. Same max_length=64, same tokenizer config, same label mapping. Pandera schemas validate data shape and types at ingestion time. MLflow tracks the exact training parameters and data splits for reproducibility.

Why Pandera over Great Expectations: Great Expectations is the enterprise standard for data validation with its Data Docs, checkpoint system, and multi-engine support (Spark, SQL, pandas). But it's heavy. The setup involves expectation suites, data contexts, checkpoint configs, and a rendering pipeline. For validating pandas DataFrames in an ML pipeline, Pandera is faster to write (it uses type-hint-style decorators), faster to execute (lighter abstraction layer), and the schemas live right in the code where they're used. Great Expectations shines when you have data contracts across teams. Pandera shines when you want unit-test-style validation inline.

Limitation: Pandera validates structure (types, ranges, nulls) but doesn't catch semantic drift. A column could pass all schema checks but contain a completely different distribution of values. That's where Evidently's drift detection comes in as a complementary layer.


HOW DOES THE API SERVING WORK?

FastAPI with ONNX Runtime as the inference backend. Single prediction endpoint (under 10ms latency), batch prediction, model health check, and model info endpoints. Redis caching for repeated queries. Rate limiting via slowapi. The model loads once at startup via a singleton pattern. Low-confidence predictions (below threshold) are flagged for human review in the response. Docker Compose packages the API + Redis for one-command deployment. A Streamlit dashboard provides the frontend: model overview with training curves, baseline vs ModernBERT comparison, live inference, and explainability (SHAP, attention heatmaps, Evidently drift reports).

Why FastAPI over dedicated ML serving frameworks: BentoML would give me automatic request batching (it collects individual requests and feeds them to the model in batches), model packaging into "Bentos," and a more structured deployment story. Ray Serve would add distributed serving with autoscaling and the ability to compose multiple models into inference graphs. Both are purpose-built for ML.

I chose FastAPI because the project has a single model, not a multi-model pipeline. BentoML's batching matters when you're GPU-bound and want to maximize throughput, but with ONNX on CPU at under 10ms, individual requests are already fast enough. Ray Serve's distributed execution is overhead when you're running one model on one machine. FastAPI gave me full control over the API design, easy integration with Redis caching, and the simplest path to the Streamlit dashboard integration.

Trade-off: if this grew to serve 5 models with complex preprocessing chains, I'd move to BentoML for the packaging and batching, or Ray Serve if I needed autoscaling across a cluster. FastAPI becomes a maintenance burden at that scale because you end up reimplementing features that ML serving frameworks provide out of the box.


HOW DOES MONITORING WORK?

Every prediction records latency and confidence to an in-memory buffer (for real-time windowed metrics) and SQLite (data/monitoring.db) for persistence. The PerformanceTracker uses SQLAlchemy with the same fallback pattern as the audit service. An AlertManager checks thresholds after each prediction: confidence drops below 0.7, p95 latency exceeds 100ms, or error rate spikes above 5%. Alerts are logged to monitoring/reports/alerts.jsonl. The API returns a prediction_id (uuid4) with every response, enabling the feedback loop.

For infrastructure monitoring, Docker Compose runs Prometheus (scraping metrics) and Grafana (dashboards). Prometheus scrapes the FastAPI /metrics endpoint for request rate, latency percentiles, error rate, and confidence distribution.

Why Prometheus + Grafana over Datadog or New Relic: Datadog would give me anomaly detection with ML-based alerting, 1000+ integrations, and zero infrastructure management. It's genuinely better for teams that want to focus on product, not operations. New Relic is similar. But both are SaaS with per-host or per-GB pricing that adds up fast, and for a portfolio project running locally, paying for cloud monitoring makes no sense.

Prometheus + Grafana is free, self-hosted, and the industry standard for Kubernetes-native monitoring. The operational cost is real though: you have to manage Prometheus storage, configure retention, set up alerting rules manually (no ML-based anomaly detection), and scale the stack yourself. For a production system at a company, I'd seriously evaluate Datadog's free tier or Grafana Cloud (managed Prometheus) to avoid the operational burden while keeping costs reasonable.

For ML-specific monitoring, Evidently AI handles drift detection and model performance tracking. I chose it over NannyML and WhyLabs because Evidently is open-source, has the largest community mindshare (19% vs NannyML's 7.5%), and generates HTML reports I can embed in the Streamlit dashboard. NannyML has a unique capability of estimating model performance without ground truth labels (using CBPE), which is valuable in production where labels arrive late. WhyLabs recently open-sourced under Apache 2.0 and focuses on real-time monitoring at enterprise scale. For this project's scope, Evidently's report generation was the right fit. In production with delayed labels, I'd add NannyML's performance estimation alongside Evidently's drift reports.


HOW DOES THE FEEDBACK LOOP WORK?

Each /predict response includes a prediction_id. Users submit corrections via POST /feedback with the prediction_id and correct label. Feedback is stored in SQLite via the audit service (FeedbackLog table). The dashboard has a dedicated Feedback page where users can submit corrections (with a dropdown of all 77 label names) and view statistics on most-corrected classes. This data feeds into retraining decisions since high correction volume on specific classes signals the model needs improvement there.

Limitation: the current feedback loop is manual. In production, you'd want automated retraining triggers: when correction volume on a class exceeds a threshold, or when Evidently detects significant drift, automatically kick off a Dagster pipeline run to retrain with the new data. The infrastructure for that is built (Dagster pipeline exists, feedback storage exists), but the trigger connection between them isn't wired up yet. That's the gap between "monitoring tells you something is wrong" and "the system fixes itself."


HOW DOES DOCKER DEPLOYMENT WORK?

Docker Compose orchestrates 5 services: Redis (caching), API (FastAPI + ONNX), Dashboard (Streamlit), Prometheus (metrics scraping), and Grafana (dashboards). The API falls back to SQLite when PostgreSQL isn't available. Model files are mounted read-only. Artifacts are mounted into the dashboard container for reading training metrics. The Grafana dashboard shows request rate, latency percentiles, error rate, and confidence distribution. One command: docker compose up --build.

Scalability path: the current setup is single-machine. To scale horizontally, you'd put the FastAPI service behind a load balancer (nginx or Traefik), move from SQLite to PostgreSQL for concurrent writes, and add a shared Redis cluster. Kubernetes would replace Docker Compose, with Prometheus Operator for monitoring. The stateless API design means horizontal scaling is straightforward since each pod loads its own ONNX model copy, and Redis handles shared cache state.


WHAT ABOUT DVC?

DVC (Data Version Control) tracks large files that don't belong in git: ml/models/, data/, and artifacts/. Configured with a local remote at /tmp/dvc-storage. This means model checkpoints, training data, and generated artifacts are versioned separately from code. In production you'd point DVC to S3 or GCS.

Why DVC over alternatives: LakeFS provides Git-like branching semantics directly on your data lake and scales to petabytes with zero-copy branching. It's architecturally superior for large-scale data engineering. But it requires running a lakeFS server, configuring it with your object store, and the overhead only pays off when you're managing hundreds of millions of objects across teams. Interestingly, lakeFS acquired DVC in 2025 but kept it open-source.

Delta Lake adds ACID transactions and time-travel to data lakes, which is great for data engineering but overkill for ML artifact versioning. It's solving a different problem (reliable data pipelines) vs DVC's problem (versioning training artifacts alongside code).

DVC fits because it's serverless, Git-integrated, and CLI-first. I run "dvc push" and "dvc pull" alongside git commands. The learning curve is minimal for anyone who knows git. The limitation is that DVC's performance degrades at very large scale (hundreds of millions of files), and it doesn't provide the branching/merging semantics that lakeFS does. For a single-developer ML project with moderate data sizes, DVC is the right level of tooling.


WHY MLFLOW FOR EXPERIMENT TRACKING?

MLflow tracks metrics, parameters, artifacts, and model versions across training runs. One pip install, one "mlflow ui" command, and you have a local tracking server with no account creation or authentication required.

The main alternative is Weights & Biases (W&B), and honestly, W&B has a better UI. The dashboards are more interactive, the sweep optimization is built-in, and real-time collaboration features are superior. There's a reason teams at major research labs prefer it. But W&B is a hosted service. Even the free tier sends your experiment data to their servers. For a portfolio project I wanted full local control with zero external dependencies.

Neptune is the third option, positioned between MLflow and W&B. Good UI, flexible metadata tracking, but also SaaS-first.

I chose MLflow because: (1) fully open-source and self-hosted, (2) zero cost at any scale, (3) integrates natively with the HuggingFace Trainer via MLflowCallback, (4) the model registry provides stage transitions (staging, production, archived) that map cleanly to deployment workflows, and (5) it's the most widely adopted in industry, so the skills transfer directly.

Limitation: MLflow's UI feels dated compared to W&B. The visualization capabilities are basic, you can't do interactive comparisons as easily, and there's no built-in hyperparameter sweep functionality. I compensate with the Streamlit dashboard for richer visualizations. In a team setting with budget, I'd use W&B for the collaboration features and keep MLflow as the model registry since they're not mutually exclusive.


WHY DAGSTER FOR ORCHESTRATION?

Dagster orchestrates the full ML pipeline: data loading, Pandera validation, augmentation, training, evaluation, ONNX export, and monitoring with drift detection. Each step is a Dagster asset, not a task.

The key alternatives are Airflow and Prefect. Airflow is the industry standard with 30+ million monthly downloads and is battle-tested at massive scale. But Airflow thinks in terms of DAGs and tasks (do this, then that), not data assets (this dataset depends on that dataset). For ML pipelines where the focus is on the data artifacts flowing through the system, Dagster's asset-centric model is more natural. Airflow 3.0 (April 2025) added Data Assets to address this gap, but it's a bolt-on to a task-centric architecture.

Prefect is developer-friendly with dynamic flows and runtime control (circuit-breakers, SLA alerting). It's great for cloud-native teams that want flexibility. But Prefect is more about workflow execution than data lineage, and it doesn't give you the same asset dependency visualization that Dagster does.

I chose Dagster because: (1) the asset graph gives a visual map of how data flows through the ML pipeline, which is exactly what I want for demo-ability, (2) strong local dev experience with "dagster dev" giving you a web UI immediately, (3) built-in support for IO managers that handle reading/writing assets, and (4) the software-defined assets paradigm maps cleanly to ML artifacts (training data, processed data, model, evaluation report are all assets with dependencies).

Limitation: Dagster has a steeper learning curve than Prefect. The concepts (assets, resources, IO managers, sensors) take time to internalize. Airflow's simpler mental model (tasks in a DAG) is easier to onboard new team members to. Also, Dagster's community is smaller than Airflow's, so you'll find fewer Stack Overflow answers. For a team already running Airflow with hundreds of DAGs, migrating to Dagster rarely makes sense. For a greenfield ML project, Dagster's asset-centric approach is the better starting point.


HOW WOULD YOU SCALE THIS?

Horizontally: ONNX Runtime supports batched inference, so you scale FastAPI pods behind a load balancer. For more classes or domains, you'd move to a shared embedding layer with per-domain classification heads. If latency budget allows, swap to ModernBERT-large for accuracy gains. For real-time retraining, add a feature store and event-driven pipeline triggers. The current architecture (stateless API + Redis cache) is already horizontally scalable.

The scaling bottlenecks in order: (1) SQLite for monitoring/feedback data since it doesn't handle concurrent writes, so you'd swap to PostgreSQL first, (2) single-machine Dagster since you'd move to Dagster Cloud or a Kubernetes-deployed Dagster instance, (3) model size if you move to larger models, at which point BentoML's adaptive batching or Ray Serve's autoscaling becomes worth the operational cost.


WHAT WOULD YOU DO DIFFERENTLY WITH MORE TIME/BUDGET?

Implement A/B testing infrastructure to compare model versions in production. Build an automated feedback loop where low-confidence predictions get routed to human review and corrections automatically trigger retraining via Dagster sensors. Explore SetFit with ModernBERT embeddings for few-shot classes that have limited training data. Add Triton Inference Server if multi-model serving becomes necessary. Deploy the full stack to cloud (currently runs locally with a single "python scripts/run_all.py" command). Add PostgreSQL-backed prediction logging for production monitoring instead of local SQLite. Wire NannyML's performance estimation into the monitoring stack for ground-truth-free performance tracking.

Honestly, the biggest gap is the lack of automated retraining. The monitoring detects problems, the feedback loop collects corrections, but a human still has to decide to retrain. Closing that loop is the difference between an ML system and a truly autonomous ML platform.


WHAT WERE THE BIGGEST CHALLENGES?

Training crashed my laptop 4 times before I moved to Kaggle. First attempt: batch_size=16, no gradient checkpointing, instant OOM. Second: batch_size=8 with huge checkpoint saves filling RAM. Third: a game competing for VRAM. Fourth: even with all optimizations (Adafactor, batch=2, grad_accum=16, gradient checkpointing), 6GB was too tight. Lesson: know your hardware budget before committing to a training strategy.

ONNX export also required debugging: the optimum library failed on ModernBERT's LayerNorm, so I switched to direct torch.onnx.export with opset 18 and had to clear value_info before INT8 quantization to fix shape inference errors. This is a real limitation of being on the cutting edge. ModernBERT was only 2 months old when I started, so tooling hadn't caught up. DeBERTa would have exported cleanly through optimum because it's been around for years. That's the trade-off of using the newest model: you gain performance and signal awareness of current research, but you pay in debugging time when tooling assumptions don't hold.


ARCHITECTURE WALKTHROUGH (2-MINUTE DEMO FLOW)

Run "python scripts/run_all.py" to start all services, then walk through:

1. Start with the problem: "Banking77 has 77 intent classes with high semantic overlap. Customers say similar things meaning different things."
2. Show the dashboard overview (localhost:8501): Training curves, 91.3% accuracy, confusion matrix, per-class metrics, all loaded from real artifacts.
3. Show model comparison: "TF-IDF + SVM gets 87.9%. ModernBERT pushes to 91.3%. Here's why." Side-by-side bar charts and confusion matrices.
4. Show live inference: Type a query in the dashboard, get prediction with confidence and latency. "Under 10ms on CPU with ONNX INT8."
5. Show explainability: SHAP feature importance, attention heatmaps showing which tokens drive predictions, Evidently drift report.
6. Show the pipeline (localhost:3000): Dagster asset graph showing data loading, validation, augmentation, training, evaluation, monitoring with drift detection.
7. Show experiment tracking (localhost:5000): MLflow with logged metrics and model versions.
8. Close: "Pandera validates data, MLflow tracks experiments, ONNX serves on any CPU, Dagster orchestrates the pipeline, Streamlit makes it demo-ready. Built on a $0 budget with a 6GB laptop GPU and free Kaggle T4."


TOPICS I'VE STUDIED

RECOMMENDATION SYSTEMS
Collaborative filtering (matrix factorization, ALS), content-based filtering, hybrid approaches. Two-tower models for retrieval. Cold start strategies. Evaluation: NDCG, MAP, recall@k. Production patterns: candidate generation, ranking, re-ranking.

RANKING AND SEARCH
Learning to rank (pointwise, pairwise, listwise). BM25 as baseline. Semantic search with dense retrieval (bi-encoders) and cross-encoder reranking. Approximate nearest neighbors (HNSW, IVF). Reciprocal rank fusion for hybrid search.

COMPUTER VISION
CNN architectures (ResNet, EfficientNet) and why they work (skip connections, compound scaling). Vision Transformers and patch embeddings. Transfer learning: freeze backbone, fine-tune head. Data augmentation as regularization. Object detection pipeline: backbone, neck, head.

TIME SERIES
Stationarity and differencing. ARIMA family as baselines. Seq2seq with attention for multi-step forecasting. Temporal Fusion Transformers for interpretable forecasting. Feature engineering: lags, rolling stats, calendar features. Walk-forward validation, never random splits.

ONLINE LEARNING
Concept drift detection (ADWIN, DDM). Model update strategies: full retrain vs incremental. Hoeffding trees for streaming classification. Reservoir sampling for maintaining representative buffers. Production pattern: shadow mode, gradual rollout, full deployment.

DISTRIBUTED TRAINING
Data parallelism (DDP) vs model parallelism. Gradient synchronization: all-reduce, ring-allreduce. Mixed precision training (FP16 forward, FP32 gradients). DeepSpeed ZeRO stages for memory optimization. Communication overhead as the real bottleneck, not compute.

FEATURE STORES
Offline store (batch features for training) vs online store (low-latency serving). Point-in-time correctness to prevent future leakage. Feast architecture: registry, offline store (BigQuery/Parquet), online store (Redis/DynamoDB). Feature engineering as a shared service across teams.

ML MATH FUNDAMENTALS
Cross-entropy loss and why it works for classification (KL divergence connection). Gradient descent variants: SGD momentum, Adam (adaptive learning rates per parameter). Attention mechanism: scaled dot-product, why we divide by sqrt(d_k). Regularization: L1 (sparsity), L2 (weight decay), dropout (ensemble approximation). Bias-variance tradeoff: underfitting vs overfitting, and how model complexity / data size shifts the curve.
