import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, average_precision_score,
                              confusion_matrix)
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


def train_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(_x_dict(data), _ei_dict(data))
    mask = data['review'].train_mask
    loss = criterion(out[mask], data['review'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(_x_dict(data), _ei_dict(data))
    logits = out[mask]
    pred  = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)[:, 1]

    y_true = data['review'].y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = probs.cpu().numpy()

    acc      = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    fake_f1  = f1_score(y_true, y_pred, pos_label=1, average='binary')

    try:
        auc    = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        auc = pr_auc = 0.0

    return dict(acc=acc, macro_f1=macro_f1, fake_f1=fake_f1,
                auc=auc, pr_auc=pr_auc,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)


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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

    # ---- Training loop ----
    best_val_f1 = 0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer, criterion)
        tr = evaluate(model, data, data['review'].train_mask)
        va = evaluate(model, data, data['review'].val_mask)
        scheduler.step(va['macro_f1'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Tr macF1 {tr['macro_f1']:.4f} AUC {tr['auc']:.4f} | "
                  f"Va macF1 {va['macro_f1']:.4f} AUC {va['auc']:.4f} "
                  f"fakeF1 {va['fake_f1']:.4f}")

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

    # ---- Final test evaluation ----
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    te = evaluate(model, data, data['review'].test_mask)
    cm = confusion_matrix(te['y_true'], te['y_pred']).tolist()

    print(f"\n=== Test results ===")
    print(f"  Accuracy  : {te['acc']:.4f}")
    print(f"  Macro-F1  : {te['macro_f1']:.4f}")
    print(f"  Fake-F1   : {te['fake_f1']:.4f}")
    print(f"  AUC-ROC   : {te['auc']:.4f}")
    print(f"  PR-AUC    : {te['pr_auc']:.4f}")
    print(f"  Confusion : {cm}")
    print("\nClassification Report:")
    print(classification_report(te['y_true'], te['y_pred'],
                                target_names=['Real', 'Fake']))

    # Save metrics
    metrics = {
        'test_accuracy': round(float(te['acc']),      4),
        'test_f1':       round(float(te['macro_f1']), 4),
        'test_fake_f1':  round(float(te['fake_f1']),  4),
        'test_auc':      round(float(te['auc']),      4),
        'test_pr_auc':   round(float(te['pr_auc']),   4),
        'best_val_f1':   round(float(best_val_f1),    4),
        'confusion_matrix': cm,
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")


if __name__ == '__main__':
    main()
