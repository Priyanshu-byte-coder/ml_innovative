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
FAKE_THRESHOLD = 0.35  # default; overridden from metrics.json if available


def load_model():
    global model, data, transformers, meta, group_maps, sbert, FAKE_THRESHOLD

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
            if 'best_threshold' in app.config['metrics']:
                FAKE_THRESHOLD = app.config['metrics']['best_threshold']
    except Exception:
        app.config['metrics'] = {}

    print(f"Model loaded. Fake threshold = {FAKE_THRESHOLD}")


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

    # Temporal defaults for a new review (7 burst features)
    # reviews_last_1h_user, reviews_last_6h_user, reviews_last_24h_user,
    # reviews_in_last_week_for_user, time_since_last_user_review,
    # reviews_last_24h_product, reviews_in_last_week_for_product
    temporal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
            fake_p = probs[1].item()
            pred  = 1 if fake_p >= FAKE_THRESHOLD else 0

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


@app.route('/full_graph')
def full_graph():
    """Return full graph data for visualization (sampled for performance)"""
    try:
        max_nodes = 5000  # Limit for browser performance
        
        # Sample review nodes
        n_reviews = data['review'].x.shape[0]
        if n_reviews > max_nodes:
            sample_idx = np.random.choice(n_reviews, max_nodes, replace=False)
        else:
            sample_idx = np.arange(n_reviews)
        
        nodes = []
        edges = []
        
        # Add review nodes
        for idx in sample_idx:
            label = int(data['review'].y[idx].item())
            nodes.append({
                'id': f'r{idx}',
                'label': f'R{idx}',
                'group': 'fake' if label == 1 else 'real',
                'title': f'Review {idx} - {"Fake" if label == 1 else "Real"}',
                'type': 'review'
            })
        
        # Add user nodes (sample)
        n_users = min(500, data['user'].x.shape[0])
        user_sample = np.random.choice(data['user'].x.shape[0], n_users, replace=False)
        for idx in user_sample:
            nodes.append({
                'id': f'u{idx}',
                'label': f'U{idx}',
                'group': 'user',
                'title': f'User {idx}',
                'type': 'user'
            })
        
        # Add product nodes (sample)
        n_products = min(200, data['product'].x.shape[0])
        product_sample = np.random.choice(data['product'].x.shape[0], n_products, replace=False)
        for idx in product_sample:
            nodes.append({
                'id': f'p{idx}',
                'label': f'P{idx}',
                'group': 'product',
                'title': f'Product {idx}',
                'type': 'product'
            })
        
        # Add edges (sample from each type)
        sampled_reviews = set(sample_idx)
        sampled_users = set(user_sample)
        sampled_products = set(product_sample)
        
        # writes edges (user -> review)
        if ('user', 'writes', 'review') in data.edge_types:
            ei = data[('user', 'writes', 'review')].edge_index
            for i in range(min(5000, ei.shape[1])):
                u_idx = int(ei[0, i].item())
                r_idx = int(ei[1, i].item())
                if u_idx in sampled_users and r_idx in sampled_reviews:
                    edges.append({'from': f'u{u_idx}', 'to': f'r{r_idx}', 'arrows': 'to'})
        
        # about edges (review -> product)
        if ('review', 'about', 'product') in data.edge_types:
            ei = data[('review', 'about', 'product')].edge_index
            for i in range(min(5000, ei.shape[1])):
                r_idx = int(ei[0, i].item())
                p_idx = int(ei[1, i].item())
                if r_idx in sampled_reviews and p_idx in sampled_products:
                    edges.append({'from': f'r{r_idx}', 'to': f'p{p_idx}', 'arrows': 'to'})
        
        # similar_text edges (review -> review)
        if ('review', 'similar_text', 'review') in data.edge_types:
            ei = data[('review', 'similar_text', 'review')].edge_index
            for i in range(min(2000, ei.shape[1])):
                r1_idx = int(ei[0, i].item())
                r2_idx = int(ei[1, i].item())
                if r1_idx in sampled_reviews and r2_idx in sampled_reviews:
                    edges.append({'from': f'r{r1_idx}', 'to': f'r{r2_idx}', 'dashes': True})
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_reviews': int(n_reviews),
                'total_users': int(data['user'].x.shape[0]),
                'total_products': int(data['product'].x.shape[0]),
                'sampled_reviews': len(sample_idx),
                'sampled_users': n_users,
                'sampled_products': n_products,
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=False, port=5000)
