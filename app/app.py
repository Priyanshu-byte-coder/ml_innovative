import sys, os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import json
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from model import FraudHeteroGNN

app = Flask(__name__)

# ---- Globals (loaded once) ----
model = None
data = None
transformers = None
meta = None
group_maps = None
sbert = None
device = torch.device('cpu')
BASE = os.path.join(os.path.dirname(__file__), '..')


def load_model():
    global model, data, transformers, meta, group_maps, sbert

    with open(os.path.join(BASE, 'processed', 'transformers.pkl'), 'rb') as f:
        transformers = pickle.load(f)

    sbert = SentenceTransformer(transformers['sbert_model'])

    data_obj = torch.load(os.path.join(BASE, 'processed', 'graph_data.pt'),
                          map_location=device, weights_only=False)
    data_obj = data_obj.to(device)

    ckpt = torch.load(os.path.join(BASE, 'checkpoints', 'best_model.pt'),
                      map_location=device, weights_only=False)
    cfg = ckpt['model_config']
    mdl = FraudHeteroGNN(**cfg).to(device)
    mdl.load_state_dict(ckpt['model_state_dict'])
    mdl.eval()

    model = mdl
    data = data_obj
    meta = pd.read_csv(os.path.join(BASE, 'processed', 'review_meta.csv'))

    with open(os.path.join(BASE, 'processed', 'group_maps.pkl'), 'rb') as f:
        group_maps = pickle.load(f)

    try:
        with open(os.path.join(BASE, 'checkpoints', 'metrics.json')) as f:
            app.config['metrics'] = json.load(f)
    except Exception:
        app.config['metrics'] = {}

    print("Model loaded successfully.")


def featurize_review(text, rating):
    """Build a review feature vector compatible with the heterogeneous graph."""
    review_scaler = transformers['review_scaler']
    text_emb = sbert.encode([text], convert_to_numpy=True)    # (1, 384)

    rating_norm  = rating / 5.0
    review_len   = min(len(text) / 10000.0, 1.0)
    word_count   = min(len(text.split()) / 1000.0, 1.0)
    user_avg     = rating / 5.0      # best guess for unknown user
    prod_avg     = 3.0 / 5.0         # neutral default
    rating_dev   = (rating - 3.0) / 5.0

    # Temporal defaults for a new review
    temporal = [0.0, 0.0, 0.0, 0.0]

    scalar = np.array([[rating_norm, review_len, word_count,
                         user_avg, prod_avg, rating_dev]])
    vec = np.hstack([text_emb, scalar, [temporal]])
    return review_scaler.transform(vec)


def _x_dict(d):
    return {k: d[k].x for k in ('review', 'user', 'product')}


def _ei_dict(d):
    return {et: d[et].edge_index for et in d.edge_types}


# ---- Routes ----

@app.route('/')
def index():
    return render_template('index.html', metrics=app.config.get('metrics', {}))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form.get('review_text', '')
        rating      = float(request.form.get('rating', 3))
        user_id     = request.form.get('user_id', '').strip()
        product_id  = request.form.get('product_id', '').strip()

        features = featurize_review(review_text, rating)
        new_review = torch.tensor(features, dtype=torch.float).to(device)

        # Append new review node
        aug_review_x = torch.cat([data['review'].x, new_review], dim=0)
        new_idx = aug_review_x.shape[0] - 1

        # Build augmented x_dict
        x_dict = {
            'review':  aug_review_x,
            'user':    data['user'].x,
            'product': data['product'].x,
        }

        # Start with existing edges
        ei_dict = _ei_dict(data)

        # Connect new review to a user node
        uid = None
        if user_id:
            try:
                uid_orig = int(user_id)
                uid = group_maps.get('uid_map', {}).get(uid_orig)
            except ValueError:
                pass

        if uid is not None:
            writes = torch.tensor([[uid], [new_idx]], dtype=torch.long, device=device)
            wby    = torch.tensor([[new_idx], [uid]], dtype=torch.long, device=device)
            ei_dict[('user', 'writes', 'review')] = torch.cat(
                [ei_dict[('user', 'writes', 'review')], writes], dim=1)
            ei_dict[('review', 'written_by', 'user')] = torch.cat(
                [ei_dict[('review', 'written_by', 'user')], wby], dim=1)

        # Connect new review to a product node
        pid = None
        if product_id:
            try:
                pid_orig = int(product_id)
                pid = group_maps.get('pid_map', {}).get(pid_orig)
            except ValueError:
                pass

        if pid is not None:
            about  = torch.tensor([[new_idx], [pid]], dtype=torch.long, device=device)
            rev_by = torch.tensor([[pid], [new_idx]], dtype=torch.long, device=device)
            ei_dict[('review', 'about', 'product')] = torch.cat(
                [ei_dict[('review', 'about', 'product')], about], dim=1)
            ei_dict[('product', 'rev_by', 'review')] = torch.cat(
                [ei_dict[('product', 'rev_by', 'review')], rev_by], dim=1)

        # Predict
        with torch.no_grad():
            out   = model(x_dict, ei_dict)
            probs = torch.softmax(out[new_idx], dim=0)
            pred  = probs.argmax().item()
            fake_p = probs[1].item()

        # Gather neighbor info for visualisation
        vis_nodes = []
        neighbor_review_ids = set()
        if uid is not None and uid in group_maps.get('user_groups', {}):
            neighbor_review_ids.update(group_maps['user_groups'][uid][:15])
        if pid is not None and pid in group_maps.get('prod_groups', {}):
            neighbor_review_ids.update(group_maps['prod_groups'][pid][:15])
        if len(neighbor_review_ids) < 5:
            # cosine fallback
            cos = F.cosine_similarity(
                data['review'].x, new_review.expand(data['review'].x.shape[0], -1))
            topk = cos.topk(15).indices.tolist()
            neighbor_review_ids.update(topk)

        for ni in list(neighbor_review_ids)[:25]:
            if ni >= len(meta):
                continue
            row = meta.iloc[ni]
            vis_nodes.append({
                'id':      int(ni),
                'rating':  float(row['rating']),
                'label':   'Fake' if int(row['label_binary']) == 1 else 'Real',
                'is_fake': bool(int(row['label_binary']) == 1),
            })

        return jsonify({
            'prediction':       'Fake' if pred == 1 else 'Real',
            'confidence':       round(max(fake_p, 1 - fake_p) * 100, 1),
            'fake_probability':  round(fake_p * 100, 1),
            'real_probability':  round((1 - fake_p) * 100, 1),
            'review_text':       review_text[:300],
            'rating':            rating,
            'neighbors':         vis_nodes,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    return jsonify({
        'review_nodes':  int(data['review'].x.shape[0]),
        'user_nodes':    int(data['user'].x.shape[0]),
        'product_nodes': int(data['product'].x.shape[0]),
        'total_edges':   total_edges,
        'metrics':       app.config.get('metrics', {}),
    })


if __name__ == '__main__':
    load_model()
    app.run(debug=False, port=5000)
