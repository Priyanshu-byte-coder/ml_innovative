import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


def load_metrics():
    metrics_path = 'checkpoints/metrics.json'
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def create_performance_chart(metrics):
    fig, ax = plt.subplots(figsize=(11, 6))

    metric_names = ['Test Accuracy', 'Macro-F1', 'Fake-F1',
                    'AUC-ROC', 'PR-AUC', 'Val Macro-F1']
    values = [
        metrics['test_accuracy'] * 100,
        metrics['test_f1'] * 100,
        metrics['test_fake_f1'] * 100,
        metrics['test_auc'] * 100,
        metrics['test_pr_auc'] * 100,
        metrics['best_val_f1'] * 100,
    ]
    colors = ['#6366f1', '#8b5cf6', '#ef4444', '#ec4899', '#f59e0b', '#10b981']

    bars = ax.barh(metric_names, values, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 1, i, f'{val:.1f}%', va='center',
                fontweight='bold', fontsize=11)

    ax.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heterogeneous GNN - Performance Metrics', fontsize=14,
                 fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/performance_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: docs/performance_metrics.png")
    plt.close()


def create_confusion_matrix_viz(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cm = np.array(metrics.get('confusion_matrix',
                               [[7802, 916], [903, 379]]))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax1.set_title('Confusion Matrix (Test Set)', fontsize=13,
                  fontweight='bold', pad=15)
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

    # Derive per-class metrics from the confusion matrix
    tn, fp = cm[0]
    fn, tp = cm[1]
    prec_real = tn / (tn + fn + 1e-8)
    rec_real  = tn / (tn + fp + 1e-8)
    f1_real   = 2 * prec_real * rec_real / (prec_real + rec_real + 1e-8)
    prec_fake = tp / (tp + fp + 1e-8)
    rec_fake  = tp / (tp + fn + 1e-8)
    f1_fake   = 2 * prec_fake * rec_fake / (prec_fake + rec_fake + 1e-8)

    classes = ['Real Reviews', 'Fake Reviews']
    precision = [prec_real, prec_fake]
    recall = [rec_real, rec_fake]
    f1 = [f1_real, f1_fake]

    x = np.arange(len(classes))
    width = 0.25
    ax2.bar(x - width, precision, width, label='Precision',
            color='#6366f1', alpha=0.8, edgecolor='black')
    ax2.bar(x, recall, width, label='Recall',
            color='#8b5cf6', alpha=0.8, edgecolor='black')
    ax2.bar(x + width, f1, width, label='F1-Score',
            color='#ec4899', alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Per-Class Performance', fontsize=13,
                  fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=10)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/classification_details.png', dpi=300,
                bbox_inches='tight')
    print("Saved: docs/classification_details.png")
    plt.close()


def create_dataset_overview():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    splits = ['Training\n(60%)', 'Validation\n(20%)', 'Test\n(20%)']
    sizes = [30000, 10000, 10000]
    colors_split = ['#6366f1', '#8b5cf6', '#ec4899']
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=splits, autopct='%1.0f%%', colors=colors_split,
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    for a in autotexts:
        a.set_color('white'); a.set_fontsize(12)
    ax1.set_title('Dataset Split (50K Reviews)', fontsize=13,
                  fontweight='bold', pad=20)

    labels = ['Real Reviews\n(86.8%)', 'Fake Reviews\n(13.2%)']
    label_sizes = [43390, 6610]
    colors_label = ['#10b981', '#ef4444']
    wedges2, texts2, autotexts2 = ax2.pie(
        label_sizes, labels=labels, autopct='%1.1f%%', colors=colors_label,
        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    for a in autotexts2:
        a.set_color('white'); a.set_fontsize(12)
    ax2.set_title('Label Distribution', fontsize=13,
                  fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('docs/dataset_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: docs/dataset_overview.png")
    plt.close()


def create_graph_stats():
    fig, ax = plt.subplots(figsize=(12, 6))

    stats = {
        'Review\nNodes': 50000,
        'User\nNodes': 39350,
        'Product\nNodes': 3735,
        'writes /\nwritten_by': 100000,
        'about /\nrev_by': 100000,
        'similar_text\nEdges': 402,
    }

    names = list(stats.keys())
    values = list(stats.values())
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#ef4444']

    log_values = [np.log10(v + 1) for v in values]
    bars = ax.bar(names, log_values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        if val >= 1000:
            label = f'{val/1000:.0f}K' if val < 1000000 else f'{val/1e6:.2f}M'
        else:
            label = str(val)
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.08,
                label, ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('log10(count)', fontsize=11, fontweight='bold')
    ax.set_title('Heterogeneous Graph Structure', fontsize=14,
                 fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(fontsize=9)

    plt.tight_layout()
    plt.savefig('docs/graph_statistics.png', dpi=300, bbox_inches='tight')
    print("Saved: docs/graph_statistics.png")
    plt.close()


def create_feature_breakdown():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # -- Review features (394-d) --
    rev_feats = [
        'SBERT Text\nEmbedding', 'Rating', 'Review Len',
        'Word Count', 'User Avg', 'Prod Avg',
        'Rating Dev', 'Temporal\n(4 feats)'
    ]
    rev_dims = [384, 1, 1, 1, 1, 1, 1, 4]
    rev_colors = ['#6366f1'] + ['#8b5cf6'] * 6 + ['#f59e0b']
    bars = axes[0].barh(rev_feats, rev_dims, color=rev_colors, alpha=0.85,
                        edgecolor='black', linewidth=1)
    for bar, d in zip(bars, rev_dims):
        axes[0].text(d + 2, bar.get_y() + bar.get_height() / 2,
                     f'{d}', va='center', fontweight='bold', fontsize=9)
    axes[0].set_title('Review Features (394-d)', fontsize=12,
                      fontweight='bold', pad=12)
    axes[0].set_xlim(0, 420)
    axes[0].grid(axis='x', alpha=0.3)

    # -- User features (7-d) --
    usr_feats = ['total_reviews', 'avg_rating', 'rating_std',
                 'rating_entropy', 'positive_ratio',
                 'avg_gap', 'review_freq']
    usr_dims = [1] * 7
    bars = axes[1].barh(usr_feats, usr_dims, color='#10b981', alpha=0.85,
                        edgecolor='black', linewidth=1)
    axes[1].set_title('User Features (7-d)', fontsize=12,
                      fontweight='bold', pad=12)
    axes[1].set_xlim(0, 2)
    axes[1].grid(axis='x', alpha=0.3)

    # -- Product features (5-d) --
    prd_feats = ['total_reviews', 'avg_rating', 'rating_var',
                 'rating_entropy', 'review_velocity']
    prd_dims = [1] * 5
    bars = axes[2].barh(prd_feats, prd_dims, color='#ec4899', alpha=0.85,
                        edgecolor='black', linewidth=1)
    axes[2].set_title('Product Features (5-d)', fontsize=12,
                      fontweight='bold', pad=12)
    axes[2].set_xlim(0, 2)
    axes[2].grid(axis='x', alpha=0.3)

    plt.suptitle('Node Feature Breakdown (Heterogeneous Graph)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('docs/feature_breakdown.png', dpi=300, bbox_inches='tight')
    print("Saved: docs/feature_breakdown.png")
    plt.close()


def create_baseline_comparison(metrics):
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ['Old: GraphSAGE\n(TF-IDF, homogeneous)',
              'New: HeteroConv\n(SBERT, heterogeneous)']
    old_auc, new_auc = 67.9, metrics['test_auc'] * 100
    old_f1, new_f1 = 54.0, metrics['test_f1'] * 100
    old_ff1, new_ff1 = 30.0, metrics['test_fake_f1'] * 100
    old_acc, new_acc = 66.2, metrics['test_accuracy'] * 100

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2,
           [old_acc, old_f1, old_ff1, old_auc],
           width, label='Old (GraphSAGE)', color='#94a3b8', edgecolor='black')
    ax.bar(x + width / 2,
           [new_acc, new_f1, new_ff1, new_auc],
           width, label='New (HeteroGNN)', color='#6366f1', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Macro-F1', 'Fake-F1', 'AUC-ROC'],
                       fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Baseline vs Upgraded Model', fontsize=14,
                 fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('docs/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: docs/baseline_comparison.png")
    plt.close()


def main():
    os.makedirs('docs', exist_ok=True)

    print("\n" + "=" * 60)
    print("  Generating Performance Visualizations")
    print("=" * 60 + "\n")

    metrics = load_metrics()

    if metrics:
        create_performance_chart(metrics)
        create_confusion_matrix_viz(metrics)
        create_baseline_comparison(metrics)
    else:
        print("Skipping metric-dependent charts (metrics not found)")

    create_dataset_overview()
    create_graph_stats()
    create_feature_breakdown()

    print("\n" + "=" * 60)
    print("  All charts generated in docs/ folder")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
