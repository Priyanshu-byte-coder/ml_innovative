# Techniques Used for Improving Fake Review Detection

This document catalogues every technique applied to improve the heterogeneous GNN fraud detection system, organized by category.

---

## 1. Decision Threshold Tuning

**Problem**: Default threshold of 0.5 biases predictions toward the majority class (real reviews).

**Solution**: Lower the decision threshold so fewer confident signals are needed to flag a review as fake.

- Default threshold set to **0.35** (configurable via `--fake_threshold`)
- After training, a **threshold sweep** is performed on the validation set over `[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]`
- The threshold maximizing **validation Fake-F1** is selected automatically
- Test metrics are reported at the best threshold AND at 0.5 for comparison

**Files**: `train.py` (sweep_thresholds, evaluate), `app.py` (FAKE_THRESHOLD), `run_pipeline.py`

---

## 2. Class-Balanced Mini-Batch Sampling

**Problem**: With ~87% real and ~13% fake reviews, standard full-batch training starves the model of fake examples.

**Solution**: Each training mini-batch is composed of approximately **50% fake and 50% real** reviews.

- `BalancedSampler` class maintains separate index arrays for fake and real training nodes
- Each batch samples `batch_size/2` from each class (with replacement)
- 8 balanced mini-batches per epoch (configurable)
- All review nodes remain connected to the full graph during message passing (only the loss is computed on the balanced subset)

**Files**: `train.py` (BalancedSampler, train_epoch_balanced)

**CLI**: `--balanced_sampling True`

---

## 3. Hard Example Mining

**Problem**: Some fake reviews are consistently misclassified; uniform sampling wastes capacity on easy examples.

**Solution**: Track frequently misclassified samples and increase their sampling probability.

- Every 5 epochs, the model is evaluated on the training set
- **False Negatives** (fake reviews predicted as real) and **False Positives** (real reviews predicted as fake) are identified
- Their sampling weights in `BalancedSampler` are boosted by a factor of **2.0×**
- Weights are re-normalized after each update
- This causes the model to see hard examples more frequently in subsequent batches

**Files**: `train.py` (find_hard_examples, BalancedSampler.update_hard_examples)

**CLI**: `--hard_example_mining True`

---

## 4. Burst Detection Temporal Features

**Problem**: Fraud campaigns often appear as bursts of reviews in short time windows.

**Solution**: Add fine-grained temporal burst features at multiple time scales.

**New features per review node** (7 total):
| Feature | Description |
|---------|-------------|
| `reviews_last_1h_user` | Reviews by same user in last 1 hour |
| `reviews_last_6h_user` | Reviews by same user in last 6 hours |
| `reviews_last_24h_user` | Reviews by same user in last 24 hours |
| `reviews_in_last_week_for_user` | Reviews by same user in last 7 days |
| `time_since_last_user_review` | Days since user's previous review |
| `reviews_last_24h_product` | Reviews on same product in last 24 hours |
| `reviews_in_last_week_for_product` | Reviews on same product in last 7 days |

**Rationale**: Multi-scale windows (1h/6h/24h) capture different burst patterns:
- 1h: Extremely rapid posting (bot-like)
- 6h: Short campaign bursts
- 24h: Daily activity spikes

**Files**: `preprocess.py` (compute_temporal_features)

---

## 5. Increased Review Similarity Edges

**Problem**: With threshold=0.9, only 402 similarity edges were created — too sparse to help.

**Solution**: Lower cosine similarity threshold from **0.9 → 0.8**.

- More `(review) --similar_text--> (review)` edges are created
- Helps the GNN detect **coordinated spam campaigns** where multiple reviews use similar language
- Bidirectional edges ensure message passing flows both ways

**Files**: `preprocess.py` (build_similarity_edges, sim_threshold default)

**CLI**: `--sim_threshold 0.8`

---

## 6. Focal Loss (retained from previous version)

**Problem**: Standard cross-entropy treats all samples equally regardless of difficulty.

**Solution**: Focal Loss down-weights easy-to-classify samples (mostly real reviews).

- **alpha = 0.75**: Weight for the positive (fake) class
- **gamma = 2.0**: Focusing parameter — higher values focus more on hard examples
- Formula: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`

**Files**: `model.py` (FocalLoss)

**CLI**: `--focal_alpha 0.75 --focal_gamma 2.0`

---

## 7. Larger Dataset Training

**Problem**: Training on 50K of 608K reviews limits the model's exposure to diverse fraud patterns.

**Solution**: Default dataset size increased to **200,000 reviews**.

- Full dataset training supported: `--max_samples 608000`
- Preprocessing remains memory-efficient (chunked similarity computation)

**Files**: `preprocess.py`, `run_pipeline.py`

**CLI**: `--max_samples 200000`

---

## 8. Expanded Evaluation Metrics

**Metrics computed and saved to `checkpoints/metrics.json`**:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Macro-F1 | Balanced F1 across both classes |
| Fake Precision | Of predicted fakes, how many are actually fake |
| Fake Recall | Of actual fakes, how many were detected |
| Fake F1 | Harmonic mean of fake precision and recall |
| AUC-ROC | Area under ROC curve |
| PR-AUC | Area under Precision-Recall curve |
| Confusion Matrix | 2×2 TP/FP/FN/TN counts |

**Additional data saved for visualization**:
- Full **Precision-Recall curve** (precision, recall arrays)
- Full **ROC curve** (FPR, TPR arrays)
- **Threshold scan**: Fake precision, recall, F1 at thresholds 0.05–0.95
- **Techniques metadata**: Which techniques were enabled for this run

**Files**: `train.py` (evaluate, main)

---

## 9. Visualization Improvements

**New charts generated by `visualize_metrics.py`**:

| Chart | File | Description |
|-------|------|-------------|
| PR Curve | `docs/pr_curve.png` | Precision-Recall curve with AUC |
| ROC Curve | `docs/roc_curve.png` | ROC curve with AUC |
| Threshold Analysis | `docs/threshold_analysis.png` | Fake recall/precision/F1 vs threshold |
| Confusion Matrix | `docs/classification_details.png` | Heatmap + per-class bar chart |
| Performance | `docs/performance_metrics.png` | All metrics at best threshold |
| Baseline Comparison | `docs/baseline_comparison.png` | Old vs new model |

---

## Configuration Summary

All techniques are controlled via CLI arguments:

```bash
python run_pipeline.py \
    --max_samples 200000 \
    --sim_threshold 0.8 \
    --fake_threshold 0.35 \
    --balanced_sampling True \
    --hard_example_mining True \
    --epochs 100
```

Or individually:

```bash
python src/preprocess.py --max_samples 200000 --sim_threshold 0.8
python src/train.py --fake_threshold 0.35 --balanced_sampling True --hard_example_mining True
```

---

## Expected Impact

| Technique | Primary Benefit |
|-----------|----------------|
| Threshold tuning | Higher fake recall at cost of some precision |
| Balanced sampling | Model sees equal fake/real during training |
| Hard example mining | Focus on difficult borderline cases |
| Burst temporal features | Capture spam campaign timing patterns |
| More similarity edges | Detect coordinated review language |
| Focal loss | Down-weight easy majority-class samples |
| Larger dataset | More diverse fraud patterns to learn from |

**Target improvements** (vs previous baseline):
- Fake Recall: 30% → 50%+
- Fake Precision: 29% → 40%+
- Fake F1: 29% → 45%+
