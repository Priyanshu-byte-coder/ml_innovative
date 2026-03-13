import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, average_precision_score,
                              precision_score, recall_score,
                              confusion_matrix, precision_recall_curve,
                              roc_curve)
import numpy as np
import json
from model import FraudHeteroGNN, FocalLoss


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _x_dict(data):
    return {k: data[k].x for k in ('review', 'user', 'product')}


def _ei_dict(data):
    return {et: data[et].edge_index for et in data.edge_types}


# ------------------------------------------------------------------
# Balanced mini-batch sampling
# ------------------------------------------------------------------
class BalancedSampler:
    """Yields balanced index masks so each 'batch' has ~50% fake / 50% real."""

    def __init__(self, labels, mask, batch_size=4096, rng_seed=42):
        idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
        self.fake_idx = idx[labels[idx] == 1]
        self.real_idx = idx[labels[idx] == 0]
        self.batch_size = batch_size
        self.rng = np.random.RandomState(rng_seed)
        # sample weights (for hard-example mining)
        self.fake_weights = np.ones(len(self.fake_idx), dtype=np.float64)
        self.real_weights = np.ones(len(self.real_idx), dtype=np.float64)
        self._normalise()

    def _normalise(self):
        self.fake_weights /= self.fake_weights.sum() + 1e-12
        self.real_weights /= self.real_weights.sum() + 1e-12

    def update_hard_examples(self, hard_fake_idx, hard_real_idx, boost=3.0):
        """Increase sampling probability for hard examples."""
        fake_set = set(hard_fake_idx)
        real_set = set(hard_real_idx)
        for i, idx in enumerate(self.fake_idx):
            if idx in fake_set:
                self.fake_weights[i] *= boost
        for i, idx in enumerate(self.real_idx):
            if idx in real_set:
                self.real_weights[i] *= boost
        self._normalise()

    def sample(self):
        half = self.batch_size // 2
        n_fake = min(half, len(self.fake_idx))
        n_real = min(half, len(self.real_idx))
        fi = self.rng.choice(len(self.fake_idx), size=n_fake, replace=True,
                             p=self.fake_weights)
        ri = self.rng.choice(len(self.real_idx), size=n_real, replace=True,
                             p=self.real_weights)
        return np.concatenate([self.fake_idx[fi], self.real_idx[ri]])


# ------------------------------------------------------------------
# Training step with balanced sampling
# ------------------------------------------------------------------
def train_epoch_balanced(model, data, optimizer, criterion, sampler,
                         device, n_batches=8):
    model.train()
    total_loss = 0.0
    x_d = _x_dict(data)
    ei_d = _ei_dict(data)
    for _ in range(n_batches):
        optimizer.zero_grad()
        out = model(x_d, ei_d)
        batch_idx = sampler.sample()
        batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=device)
        loss = criterion(out[batch_idx_t], data['review'].y[batch_idx_t])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / n_batches


def train_epoch_full(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(_x_dict(data), _ei_dict(data))
    mask = data['review'].train_mask
    loss = criterion(out[mask], data['review'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# ------------------------------------------------------------------
# Evaluation (with configurable threshold)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, mask, threshold=0.5):
    model.eval()
    out = model(_x_dict(data), _ei_dict(data))
    logits = out[mask]
    probs = torch.softmax(logits, dim=1)[:, 1]

    y_true = data['review'].y[mask].cpu().numpy()
    y_prob = probs.cpu().numpy()
    y_pred = (y_prob >= threshold).astype(int)

    acc      = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    fake_f1  = f1_score(y_true, y_pred, pos_label=1, average='binary',
                        zero_division=0)
    fake_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    fake_rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc    = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        auc = pr_auc = 0.0

    return dict(acc=acc, macro_f1=macro_f1, fake_f1=fake_f1,
                fake_prec=fake_prec, fake_rec=fake_rec,
                auc=auc, pr_auc=pr_auc,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)


# ------------------------------------------------------------------
# Find hard examples (FN fakes + FP reals) for mining
# ------------------------------------------------------------------
def find_hard_examples(model, data, train_mask, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(_x_dict(data), _ei_dict(data))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    y = data['review'].y.cpu().numpy()
    idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()

    preds = (probs[idx] >= threshold).astype(int)
    truth = y[idx]

    hard_fake = idx[(truth == 1) & (preds == 0)]  # FN
    hard_real = idx[(truth == 0) & (preds == 1)]  # FP
    return hard_fake, hard_real


# ------------------------------------------------------------------
# Threshold sweep — pick best threshold on validation fake-F1
# ------------------------------------------------------------------
def sweep_thresholds(model, data, val_mask, thresholds):
    best_t, best_ff1 = 0.5, 0.0
    print("\n  Threshold sweep on validation set:")
    for t in thresholds:
        res = evaluate(model, data, val_mask, threshold=t)
        tag = ""
        if res['fake_f1'] > best_ff1:
            best_ff1 = res['fake_f1']
            best_t = t
            tag = " <-- best"
        print(f"    t={t:.2f}  fakeP={res['fake_prec']:.3f}  "
              f"fakeR={res['fake_rec']:.3f}  fakeF1={res['fake_f1']:.3f}  "
              f"macF1={res['macro_f1']:.3f}{tag}")
    return best_t, best_ff1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Fraud-HeteroGNN")
    parser.add_argument('--data_dir',   default='processed')
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--hidden',   type=int,   default=128)
    parser.add_argument('--dropout',  type=float, default=0.3)
    parser.add_argument('--lr',       type=float, default=0.001)
    parser.add_argument('--epochs',   type=int,   default=100)
    parser.add_argument('--patience', type=int,   default=15)
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--fake_threshold', type=float, default=0.35)
    parser.add_argument('--balanced_sampling', type=str, default='True')
    parser.add_argument('--hard_example_mining', type=str, default='True')
    args = parser.parse_args()

    use_balanced = args.balanced_sampling.lower() in ('true', '1', 'yes')
    use_hard_mining = args.hard_example_mining.lower() in ('true', '1', 'yes')

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Balanced sampling: {use_balanced}")
    print(f"Hard example mining: {use_hard_mining}")
    print(f"Initial fake threshold: {args.fake_threshold}")

    # ---- Load hetero data ----
    print("Loading HeteroData ...")
    data = torch.load(os.path.join(args.data_dir, 'graph_data.pt'),
                       weights_only=False)
    data = data.to(device)

    review_dim  = data['review'].x.shape[1]
    user_dim    = data['user'].x.shape[1]
    product_dim = data['product'].x.shape[1]
    metadata    = data.metadata()

    print(f"  Review  : {data['review'].x.shape}")
    print(f"  User    : {data['user'].x.shape}")
    print(f"  Product : {data['product'].x.shape}")
    for et in data.edge_types:
        print(f"  {et}: {data[et].edge_index.shape[1]} edges")

    # ---- Model ----
    model = FraudHeteroGNN(
        metadata=metadata,
        review_dim=review_dim,
        user_dim=user_dim,
        product_dim=product_dim,
        hidden_channels=args.hidden,
        out_channels=2,
        dropout=args.dropout,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=7, factor=0.5)

    # ---- Balanced sampler ----
    sampler = None
    if use_balanced:
        labels_np = data['review'].y.cpu().numpy()
        sampler = BalancedSampler(labels_np, data['review'].train_mask,
                                  batch_size=4096)
        n_fake = (labels_np[data['review'].train_mask.cpu().numpy().astype(bool)] == 1).sum()
        n_real = (labels_np[data['review'].train_mask.cpu().numpy().astype(bool)] == 0).sum()
        print(f"  Train fakes: {n_fake}, reals: {n_real}")

    # ---- Training loop ----
    best_val_f1 = 0
    patience_ctr = 0
    hard_mining_interval = 5  # re-mine hard examples every N epochs

    for epoch in range(1, args.epochs + 1):
        # Train step
        if use_balanced and sampler is not None:
            loss = train_epoch_balanced(model, data, optimizer, criterion,
                                        sampler, device, n_batches=8)
        else:
            loss = train_epoch_full(model, data, optimizer, criterion)

        # Hard example mining
        if use_hard_mining and sampler is not None and epoch % hard_mining_interval == 0:
            hf, hr = find_hard_examples(model, data, data['review'].train_mask,
                                         device, threshold=args.fake_threshold)
            sampler.update_hard_examples(hf, hr, boost=2.0)
            if epoch % 10 == 0:
                print(f"  [HEM] epoch {epoch}: {len(hf)} hard fakes, "
                      f"{len(hr)} hard reals boosted")

        tr = evaluate(model, data, data['review'].train_mask,
                      threshold=args.fake_threshold)
        va = evaluate(model, data, data['review'].val_mask,
                      threshold=args.fake_threshold)
        scheduler.step(va['macro_f1'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Tr macF1 {tr['macro_f1']:.4f} AUC {tr['auc']:.4f} | "
                  f"Va macF1 {va['macro_f1']:.4f} AUC {va['auc']:.4f} "
                  f"fakeF1 {va['fake_f1']:.4f} fakeR {va['fake_rec']:.4f}")

        if va['macro_f1'] > best_val_f1:
            best_val_f1 = va['macro_f1']
            patience_ctr = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': dict(
                    metadata=metadata,
                    review_dim=review_dim,
                    user_dim=user_dim,
                    product_dim=product_dim,
                    hidden_channels=args.hidden,
                    out_channels=2,
                    dropout=args.dropout,
                ),
                'best_val_f1': best_val_f1,
            }, os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ---- Load best model ----
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # ---- Threshold tuning on validation set ----
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_threshold, best_val_fake_f1 = sweep_thresholds(
        model, data, data['review'].val_mask, thresholds)
    print(f"\n  Best threshold: {best_threshold} (val fakeF1={best_val_fake_f1:.4f})")

    # ---- Final test evaluation with best threshold ----
    te = evaluate(model, data, data['review'].test_mask, threshold=best_threshold)
    cm = confusion_matrix(te['y_true'], te['y_pred']).tolist()

    print(f"\n=== Test results (threshold={best_threshold}) ===")
    print(f"  Accuracy   : {te['acc']:.4f}")
    print(f"  Macro-F1   : {te['macro_f1']:.4f}")
    print(f"  Fake-Prec  : {te['fake_prec']:.4f}")
    print(f"  Fake-Recall: {te['fake_rec']:.4f}")
    print(f"  Fake-F1    : {te['fake_f1']:.4f}")
    print(f"  AUC-ROC    : {te['auc']:.4f}")
    print(f"  PR-AUC     : {te['pr_auc']:.4f}")
    print(f"  Confusion  : {cm}")
    print("\nClassification Report:")
    print(classification_report(te['y_true'], te['y_pred'],
                                target_names=['Real', 'Fake']))

    # ---- Also compute test at default 0.5 for comparison ----
    te_default = evaluate(model, data, data['review'].test_mask, threshold=0.5)

    # ---- Compute curves for visualization ----
    pr_prec_curve, pr_rec_curve, pr_thresholds = precision_recall_curve(
        te['y_true'], te['y_prob'])
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(te['y_true'], te['y_prob'])

    # Fake recall at each threshold for plotting
    threshold_scan = np.arange(0.05, 0.96, 0.05).tolist()
    threshold_fake_recall = []
    threshold_fake_prec = []
    threshold_fake_f1 = []
    for t in threshold_scan:
        yp = (te['y_prob'] >= t).astype(int)
        fr = recall_score(te['y_true'], yp, pos_label=1, zero_division=0)
        fp = precision_score(te['y_true'], yp, pos_label=1, zero_division=0)
        ff = f1_score(te['y_true'], yp, pos_label=1, zero_division=0,
                      average='binary')
        threshold_fake_recall.append(round(fr, 4))
        threshold_fake_prec.append(round(fp, 4))
        threshold_fake_f1.append(round(ff, 4))

    # ---- Save metrics ----
    metrics = {
        'test_accuracy':   round(float(te['acc']),       4),
        'test_f1':         round(float(te['macro_f1']),  4),
        'test_fake_f1':    round(float(te['fake_f1']),   4),
        'test_fake_prec':  round(float(te['fake_prec']), 4),
        'test_fake_rec':   round(float(te['fake_rec']),  4),
        'test_auc':        round(float(te['auc']),       4),
        'test_pr_auc':     round(float(te['pr_auc']),    4),
        'best_val_f1':     round(float(best_val_f1),     4),
        'best_threshold':  round(float(best_threshold),  2),
        'confusion_matrix': cm,
        # comparison at default threshold
        'test_default_05': {
            'accuracy':  round(float(te_default['acc']),      4),
            'macro_f1':  round(float(te_default['macro_f1']), 4),
            'fake_f1':   round(float(te_default['fake_f1']),  4),
            'fake_prec': round(float(te_default['fake_prec']),4),
            'fake_rec':  round(float(te_default['fake_rec']), 4),
        },
        # curves for visualization
        'pr_curve': {
            'precision': [round(float(x), 4) for x in pr_prec_curve.tolist()],
            'recall':    [round(float(x), 4) for x in pr_rec_curve.tolist()],
        },
        'roc_curve': {
            'fpr': [round(float(x), 4) for x in fpr_curve.tolist()],
            'tpr': [round(float(x), 4) for x in tpr_curve.tolist()],
        },
        'threshold_scan': {
            'thresholds':  [round(t, 2) for t in threshold_scan],
            'fake_recall': threshold_fake_recall,
            'fake_prec':   threshold_fake_prec,
            'fake_f1':     threshold_fake_f1,
        },
        # techniques used
        'techniques': {
            'balanced_sampling': use_balanced,
            'hard_example_mining': use_hard_mining,
            'focal_loss': {'alpha': args.focal_alpha, 'gamma': args.focal_gamma},
            'threshold_tuning': True,
            'best_threshold': round(float(best_threshold), 2),
        },
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")


if __name__ == '__main__':
    main()
