# Heterogeneous GNN-Based Fraudulent Review Detection

A **Heterogeneous Graph Neural Network** system that detects fraudulent Yelp reviews by modeling **User-Review-Product** interactions as a typed graph with relational convolutions, Sentence-BERT embeddings, and Focal Loss.

---

## Performance Metrics

![Performance Metrics](docs/performance_metrics.png)

### Test Set Results (10,000 reviews)

| Metric | Score |
|--------|-------|
| **Accuracy** | 81.8% |
| **Macro F1-Score** | 59.5% |
| **Fake-Class F1** | 29.4% |
| **AUC-ROC** | 70.4% |
| **PR-AUC** | 25.4% |
| **Best Val Macro-F1** | 61.2% |

### Per-Class Performance

![Classification Details](docs/classification_details.png)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Real Reviews** | 90% | 89% | 90% | 8,718 |
| **Fake Reviews** | 29% | 30% | 29% | 1,282 |

### Baseline Comparison

![Baseline Comparison](docs/baseline_comparison.png)

| Metric | Old (GraphSAGE + TF-IDF) | New (HeteroGNN + SBERT) | Change |
|--------|--------------------------|------------------------|--------|
| Accuracy | 66.2% | **81.8%** | +15.6 |
| Macro-F1 | 54.0% | **59.5%** | +5.5 |
| AUC-ROC | 67.9% | **70.4%** | +2.5 |

---

## Architecture Overview

### 1. Heterogeneous Graph (User-Review-Product)

The system builds a **typed heterogeneous graph** with three node types and five edge relations:

```
User ──writes──> Review ──about──> Product
User <─written_by─ Review <─rev_by── Product
                Review ──similar_text──> Review
```

![Graph Statistics](docs/graph_statistics.png)

| Component | Count |
|-----------|-------|
| **Review nodes** | 50,000 |
| **User nodes** | 39,350 |
| **Product nodes** | 3,735 |
| **writes / written_by edges** | 100,000 (50K each direction) |
| **about / rev_by edges** | 100,000 (50K each direction) |
| **similar_text edges** | 402 (cosine > 0.9) |

### 2. Feature Engineering

Three separate feature vectors for each node type:

![Feature Breakdown](docs/feature_breakdown.png)

**Review features (394-d)**:

| Feature | Dimensions | Description |
|---------|------------|-------------|
| SBERT text embedding | 384 | Sentence-BERT `all-MiniLM-L6-v2` semantic embedding |
| Rating | 1 | Normalized star rating |
| Review length | 1 | Character count (normalized) |
| Word count | 1 | Token count (normalized) |
| User avg rating | 1 | Mean rating of the review's author |
| Product avg rating | 1 | Mean rating of the reviewed product |
| Rating deviation | 1 | How much this rating deviates from product average |
| Temporal features | 4 | reviews_in_last_24h, reviews_in_last_week (user), time_since_last_review, reviews_in_last_week (product) |

**User features (7-d)**: total_reviews, avg_rating, rating_std, rating_entropy, positive_ratio, avg_gap_between_reviews, review_frequency

**Product features (5-d)**: total_reviews, avg_rating, rating_variance, rating_entropy, review_velocity

### 3. Model Architecture (HeteroConv + Focal Loss)

```
Per-type Linear Projections (review: 394→128, user: 7→128, product: 5→128)
    ↓
HeteroConv Layer 1 (SAGEConv per relation, 128→128) + BatchNorm + ReLU + Dropout
    ↓
HeteroConv Layer 2 (SAGEConv per relation, 128→128) + BatchNorm + ReLU + Dropout
    ↓
Review-only Classifier Head (128 → 64 → 2)
    ↓
Output: [P(Real), P(Fake)]     Loss: Focal Loss (alpha=0.75, gamma=2.0)
```

**Key design choices**:
- **HeteroConv** wraps a separate SAGEConv per edge type, enabling typed message passing
- **Focal Loss** (alpha=0.75, gamma=2.0) down-weights easy-to-classify real reviews, focusing gradients on hard fake examples
- Only **review nodes** are classified; user and product nodes provide context via message passing

### 4. Training Strategy

![Dataset Overview](docs/dataset_overview.png)

| Setting | Value |
|---------|-------|
| Split | 60% train / 20% val / 20% test |
| Class distribution | 86.8% real, 13.2% fake |
| Loss function | Focal Loss (alpha=0.75, gamma=2.0) |
| Optimizer | Adam (lr=0.001, weight_decay=5e-4) |
| Scheduler | ReduceLROnPlateau (patience=7, factor=0.5) |
| Early stopping | patience=15 on validation macro-F1 |
| Training duration | 76 epochs (early stopped) |
| Parameters | 391,234 |

---

## Quick Start

### Installation

```bash
git clone <repo-url>
cd ml_innovative
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This will:
1. Load 50,000 reviews from `Yelp-Dataset/yelpzip.csv`
2. Compute SBERT embeddings (~20 min on CPU)
3. Build heterogeneous graph with temporal, behavioral, and similarity features
4. Train the HeteroGNN for up to 100 epochs (~5 min on CPU)
5. Launch the web app at **http://localhost:5000**

### Run Individual Steps

```bash
# Preprocess (SBERT + graph construction)
python src/preprocess.py --data_dir Yelp-Dataset --output_dir processed --max_samples 50000

# Train
python src/train.py --data_dir processed --output_dir checkpoints --epochs 100

# Launch web app
python app/app.py

# Generate charts
python visualize_metrics.py
```

---

## Web Application

The Flask web app provides an interactive interface to:

1. **Submit a review** (text + rating + optional user/product IDs)
2. **Get predictions** (Real/Fake with confidence scores)
3. **Visualize the graph** (see how the review connects to neighbors)
4. **Analyze patterns** (view neighbor breakdown: real vs fake)

**How prediction works**:
1. The review text is encoded with SBERT into a 384-d embedding
2. Scalar + temporal features are appended to form a 394-d review vector
3. The new review node is connected to the heterogeneous graph:
   - To the matching **user node** via `writes`/`written_by` edges (if user_id provided)
   - To the matching **product node** via `about`/`rev_by` edges (if product_id provided)
   - To nearest review neighbours via cosine similarity fallback
4. The HeteroGNN propagates information through all node types
5. Output: Probability distribution [P(Real), P(Fake)]

---

## Project Structure

```
ml_innovative/
├── Yelp-Dataset/              # Dataset files (not in repo)
│   ├── yelpzip.csv            # Main dataset (608K reviews)
│   └── YelpNYC/               # NYC-specific data
│
├── src/
│   ├── preprocess.py          # SBERT embeddings, hetero graph, user/product/temporal features
│   ├── model.py               # FraudHeteroGNN (HeteroConv) + FocalLoss
│   └── train.py               # Training with focal loss, PR-AUC, confusion matrix
│
├── app/
│   ├── app.py                 # Flask server (SBERT inference, HeteroGNN prediction)
│   └── templates/index.html   # Frontend (Tailwind + vis.js + Chart.js)
│
├── processed/                 # Generated by preprocess.py
│   ├── graph_data.pt          # HeteroData (review/user/product nodes + 5 edge types)
│   ├── transformers.pkl       # Scalers + SBERT model name
│   ├── review_meta.csv        # Lightweight metadata
│   └── group_maps.pkl         # uid/pid mappings + group indices
│
├── checkpoints/               # Generated by train.py
│   ├── best_model.pt          # Trained HeteroGNN weights + config
│   └── metrics.json           # All metrics + confusion matrix
│
├── docs/                      # Generated by visualize_metrics.py
│   ├── performance_metrics.png
│   ├── classification_details.png
│   ├── baseline_comparison.png
│   ├── dataset_overview.png
│   ├── graph_statistics.png
│   └── feature_breakdown.png
│
├── run_pipeline.py            # One-command full pipeline
├── visualize_metrics.py       # Generate all performance charts
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Configuration

### Preprocessing

```bash
python src/preprocess.py \
    --data_dir Yelp-Dataset \
    --output_dir processed \
    --max_samples 200000 \
    --sim_threshold 0.9
```

### Training

```bash
python src/train.py \
    --data_dir processed \
    --output_dir checkpoints \
    --hidden 256 \
    --dropout 0.4 \
    --lr 0.0005 \
    --epochs 200 \
    --patience 20 \
    --focal_alpha 0.75 \
    --focal_gamma 2.0
```

### Pipeline

```bash
python run_pipeline.py --max_samples 200000 --epochs 200
python run_pipeline.py --skip_train
python run_pipeline.py --app_only
```

---

## Further Improvements

| Area | Current | Potential |
|------|---------|-----------|
| Data | 50K reviews | Full 608K (`--max_samples 608000`) |
| Text | SBERT `all-MiniLM-L6-v2` | Fine-tuned domain-specific BERT |
| Loss | Focal Loss | Focal + adversarial training |
| Model | 2-layer HeteroConv | Deeper / GAT attention / GIN |
| Edges | cosine > 0.9 similarity | Temporal proximity, rating-pattern edges |
| Imbalance | Focal Loss alpha=0.75 | SMOTE, class-balanced sampling |

---

## Dataset

**Source**: Yelp review dataset with ground-truth fraud labels

- `yelpzip.csv`: 608,458 reviews, 260,239 users, 5,044 products
- Labels: `1` (real), `-1` (fake) — 86.8% / 13.2% split
- Current usage: 50,000 reviews (stratified sample)

---

## Technical Stack

- **Deep Learning**: PyTorch 2.10, PyTorch Geometric 2.6
- **NLP**: sentence-transformers (all-MiniLM-L6-v2)
- **ML/Data**: scikit-learn, pandas, numpy, scipy
- **Web**: Flask 3.1, Tailwind CSS, vis.js, Chart.js
- **Visualization**: matplotlib, seaborn