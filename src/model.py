import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, BatchNorm


# ======================================================================
# Focal Loss
# ======================================================================
class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=logits.size(1)).float()

        pt = (probs * targets_oh).sum(dim=1)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())

        loss = -alpha_t * ((1 - pt) ** self.gamma) * pt.clamp(min=1e-8).log()
        return loss.mean()


# ======================================================================
# Heterogeneous GNN  (HeteroConv wrapping SAGEConv per relation)
# ======================================================================
class FraudHeteroGNN(nn.Module):
    """Two-layer heterogeneous GNN for fraud detection on review nodes."""

    def __init__(self, metadata, review_dim, user_dim, product_dim,
                 hidden_channels=128, out_channels=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Per-type input projections to a shared hidden dim
        self.proj = nn.ModuleDict({
            'review':  nn.Linear(review_dim,  hidden_channels),
            'user':    nn.Linear(user_dim,    hidden_channels),
            'product': nn.Linear(product_dim, hidden_channels),
        })

        # Two HeteroConv layers
        self.conv1 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.bn1 = nn.ModuleDict({nt: BatchNorm(hidden_channels) for nt in metadata[0]})
        self.bn2 = nn.ModuleDict({nt: BatchNorm(hidden_channels) for nt in metadata[0]})

        # Classifier head (only applied to review nodes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x_dict, edge_index_dict):
        # Project to shared dim
        h = {k: F.relu(self.proj[k](x)) for k, x in x_dict.items()}

        # Layer 1
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(self.bn1[k](v)) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        # Layer 2
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(self.bn2[k](v)) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        return self.classifier(h['review'])

    def get_review_embeddings(self, x_dict, edge_index_dict):
        """Return review node embeddings before the classifier."""
        h = {k: F.relu(self.proj[k](x)) for k, x in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(self.bn1[k](v)) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(self.bn2[k](v)) for k, v in h.items()}
        return h['review']
