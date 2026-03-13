import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from scipy.stats import entropy as sp_entropy
import torch
from torch_geometric.data import HeteroData
import pickle
import os
import time


class YelpGraphBuilder:
    """Build a heterogeneous User-Review-Product graph from Yelp data."""

    def __init__(self, data_dir, output_dir, max_samples=None,
                 sbert_model='all-MiniLM-L6-v2', sim_threshold=0.8,
                 sbert_batch=512):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.sbert_model_name = sbert_model
        self.sim_threshold = sim_threshold
        self.sbert_batch = sbert_batch
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    def load_data(self):
        print("Loading yelpzip.csv ...")
        df = pd.read_csv(os.path.join(self.data_dir, 'yelpzip.csv'))
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        df['label_binary'] = (df['label'] == -1).astype(int)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        if self.max_samples and self.max_samples < len(df):
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1,
                                         train_size=self.max_samples,
                                         random_state=42)
            idx, _ = next(sss.split(df, df['label_binary']))
            df = df.iloc[idx].reset_index(drop=True)

        # Contiguous integer IDs for users & products
        df['uid'] = df['user_id'].astype('category').cat.codes.values
        df['pid'] = df['prod_id'].astype('category').cat.codes.values
        self.uid_map = dict(zip(df['user_id'], df['uid']))
        self.pid_map = dict(zip(df['prod_id'], df['pid']))
        self.n_users = df['uid'].nunique()
        self.n_prods = df['pid'].nunique()

        print(f"  Reviews : {len(df)}")
        print(f"  Users   : {self.n_users}")
        print(f"  Products: {self.n_prods}")
        print(f"  Fake    : {df['label_binary'].sum()} "
              f"({df['label_binary'].mean()*100:.1f}%)")
        self.df = df
        return df

    # ------------------------------------------------------------------
    # 2. Sentence-BERT text embeddings (384-d)
    # ------------------------------------------------------------------
    def compute_text_embeddings(self):
        print("Computing SBERT embeddings ...")
        t0 = time.time()
        sbert = SentenceTransformer(self.sbert_model_name)
        texts = self.df['text'].fillna('').tolist()
        self.text_emb = sbert.encode(texts,
                                     batch_size=self.sbert_batch,
                                     show_progress_bar=True,
                                     convert_to_numpy=True)
        print(f"  SBERT dim: {self.text_emb.shape[1]}  ({time.time()-t0:.1f}s)")
        self.sbert = sbert

    # ------------------------------------------------------------------
    # 3. Temporal features (per review)
    # ------------------------------------------------------------------
    def compute_temporal_features(self):
        print("Computing temporal features ...")
        df = self.df.copy()
        df = df.sort_values('date')

        # Per-user temporal
        user_dates = df.groupby('uid')['date']
        df['time_since_last_user_review'] = (
            user_dates.diff().dt.total_seconds().fillna(0) / 86400.0)

        def rolling_count_hours(group, hours):
            """Count reviews within `hours` hours before each review."""
            out = np.zeros(len(group))
            dates = group.values
            delta = np.timedelta64(hours, 'h')
            for i in range(len(dates)):
                out[i] = np.sum((dates[i] - dates[:i]) <= delta)
            return pd.Series(out, index=group.index)

        # Burst detection: 1h, 6h, 24h windows for user
        df['reviews_last_1h_user'] = (
            user_dates.apply(lambda g: rolling_count_hours(g, 1))
            .droplevel(0).reindex(df.index).fillna(0))
        df['reviews_last_6h_user'] = (
            user_dates.apply(lambda g: rolling_count_hours(g, 6))
            .droplevel(0).reindex(df.index).fillna(0))
        df['reviews_last_24h_user'] = (
            user_dates.apply(lambda g: rolling_count_hours(g, 24))
            .droplevel(0).reindex(df.index).fillna(0))
        df['reviews_in_last_week_for_user'] = (
            user_dates.apply(lambda g: rolling_count_hours(g, 168))
            .droplevel(0).reindex(df.index).fillna(0))

        # Per-product temporal: 24h and 7-day windows
        prod_dates = df.groupby('pid')['date']
        df['reviews_last_24h_product'] = (
            prod_dates.apply(lambda g: rolling_count_hours(g, 24))
            .droplevel(0).reindex(df.index).fillna(0))
        df['reviews_in_last_week_for_product'] = (
            prod_dates.apply(lambda g: rolling_count_hours(g, 168))
            .droplevel(0).reindex(df.index).fillna(0))

        # Re-align to original index order
        df = df.sort_index()
        self.temporal_cols = [
            'reviews_last_1h_user',
            'reviews_last_6h_user',
            'reviews_last_24h_user',
            'reviews_in_last_week_for_user',
            'time_since_last_user_review',
            'reviews_last_24h_product',
            'reviews_in_last_week_for_product',
        ]
        for c in self.temporal_cols:
            self.df[c] = df[c].values

    # ------------------------------------------------------------------
    # 4. Review node features
    # ------------------------------------------------------------------
    def build_review_features(self):
        print("Building review node features ...")
        df = self.df
        user_avg = df.groupby('uid')['rating'].transform('mean') / 5.0
        prod_avg = df.groupby('pid')['rating'].transform('mean') / 5.0

        scalar = np.column_stack([
            df['rating'].values / 5.0,
            df['text'].fillna('').str.len().values
                / (df['text'].fillna('').str.len().max() + 1e-8),
            df['text'].fillna('').str.split().str.len().values
                / (df['text'].fillna('').str.split().str.len().max() + 1e-8),
            user_avg.values,
            prod_avg.values,
            (df['rating'].values - prod_avg.values * 5.0) / 5.0,
        ])
        temporal = df[self.temporal_cols].values.astype(float)

        combined = np.hstack([self.text_emb, scalar, temporal])
        self.review_scaler = StandardScaler()
        self.review_features = self.review_scaler.fit_transform(combined)
        print(f"  Review feature dim: {self.review_features.shape[1]}")

    # ------------------------------------------------------------------
    # 5. User node features
    # ------------------------------------------------------------------
    def build_user_features(self):
        print("Building user node features ...")
        df = self.df
        uf = df.groupby('uid').agg(
            total_reviews=('rating', 'size'),
            avg_rating=('rating', 'mean'),
            rating_std=('rating', 'std'),
        ).fillna(0)

        # Rating entropy per user
        def rating_entropy(ratings):
            counts = np.bincount(ratings.astype(int).clip(1, 5), minlength=6)[1:]
            p = counts / (counts.sum() + 1e-8)
            return sp_entropy(p + 1e-8)

        uf['rating_entropy'] = df.groupby('uid')['rating'].apply(rating_entropy)

        # Positive ratio
        uf['positive_ratio'] = df.groupby('uid')['rating'].apply(
            lambda r: (r >= 4).mean())

        # Average time between reviews & frequency
        def time_stats(g):
            g = g.sort_values()
            diffs = g.diff().dt.total_seconds().dropna() / 86400.0
            avg_gap = diffs.mean() if len(diffs) else 0
            span = (g.max() - g.min()).total_seconds() / (86400 * 30) + 1e-8
            freq = len(g) / span
            return pd.Series({'avg_gap': avg_gap, 'review_freq': freq})

        ts = df.groupby('uid')['date'].apply(time_stats).unstack().fillna(0)
        uf = uf.join(ts)

        scaler = StandardScaler()
        cols = ['total_reviews', 'avg_rating', 'rating_std',
                'rating_entropy', 'positive_ratio', 'avg_gap', 'review_freq']
        uf_arr = scaler.fit_transform(uf[cols].values)

        # Ensure aligned to 0..n_users-1
        self.user_features = np.zeros((self.n_users, uf_arr.shape[1]))
        self.user_features[uf.index.values] = uf_arr
        self.user_scaler = scaler
        print(f"  User feature dim: {self.user_features.shape[1]}")

    # ------------------------------------------------------------------
    # 6. Product node features
    # ------------------------------------------------------------------
    def build_product_features(self):
        print("Building product node features ...")
        df = self.df
        pf = df.groupby('pid').agg(
            total_reviews=('rating', 'size'),
            avg_rating=('rating', 'mean'),
            rating_var=('rating', 'var'),
        ).fillna(0)

        def rating_entropy(ratings):
            counts = np.bincount(ratings.astype(int).clip(1, 5), minlength=6)[1:]
            p = counts / (counts.sum() + 1e-8)
            return sp_entropy(p + 1e-8)

        pf['rating_entropy'] = df.groupby('pid')['rating'].apply(rating_entropy)

        # Review velocity (reviews per week)
        def velocity(g):
            span_weeks = (g.max() - g.min()).total_seconds() / (86400 * 7) + 1e-8
            return len(g) / span_weeks

        pf['review_velocity'] = df.groupby('pid')['date'].apply(velocity)

        scaler = StandardScaler()
        cols = ['total_reviews', 'avg_rating', 'rating_var',
                'rating_entropy', 'review_velocity']
        pf_arr = scaler.fit_transform(pf[cols].values)

        self.product_features = np.zeros((self.n_prods, pf_arr.shape[1]))
        self.product_features[pf.index.values] = pf_arr
        self.product_scaler = scaler
        print(f"  Product feature dim: {self.product_features.shape[1]}")

    # ------------------------------------------------------------------
    # 7. Build heterogeneous edges
    # ------------------------------------------------------------------
    def build_hetero_edges(self):
        print("Building heterogeneous edges ...")
        df = self.df
        review_idx = torch.arange(len(df), dtype=torch.long)
        user_idx   = torch.tensor(df['uid'].values, dtype=torch.long)
        prod_idx   = torch.tensor(df['pid'].values, dtype=torch.long)

        self.edge_index_dict = {
            ('user', 'writes', 'review'):      torch.stack([user_idx, review_idx]),
            ('review', 'written_by', 'user'):  torch.stack([review_idx, user_idx]),
            ('review', 'about', 'product'):    torch.stack([review_idx, prod_idx]),
            ('product', 'rev_by', 'review'):   torch.stack([prod_idx, review_idx]),
        }

        for k, v in self.edge_index_dict.items():
            print(f"  {k}: {v.shape[1]} edges")

    # ------------------------------------------------------------------
    # 8. Review similarity edges (cosine > threshold)
    # ------------------------------------------------------------------
    def build_similarity_edges(self):
        print(f"Building similarity edges (threshold={self.sim_threshold}) ...")
        t0 = time.time()
        emb = torch.tensor(self.text_emb, dtype=torch.float)
        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        emb_n = emb / norms

        src_list, dst_list = [], []
        n = emb_n.shape[0]
        chunk = 2000
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            sim = emb_n[i:end] @ emb_n.T          # (chunk, n)
            sim[:, :end] = sim[:, :end].tril(-1)   # avoid duplicates & self
            if i > 0:
                pass  # upper-left already handled
            rows, cols = (sim > self.sim_threshold).nonzero(as_tuple=True)
            src_list.append(rows + i)
            dst_list.append(cols)

        if src_list:
            src = torch.cat(src_list)
            dst = torch.cat(dst_list)
            # bidirectional
            ei = torch.stack([torch.cat([src, dst]),
                              torch.cat([dst, src])])
            self.edge_index_dict[
                ('review', 'similar_text', 'review')] = ei
            print(f"  similar_text edges: {ei.shape[1]}  ({time.time()-t0:.1f}s)")
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
            self.edge_index_dict[
                ('review', 'similar_text', 'review')] = ei
            print(f"  similar_text edges: 0  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # 9. Assemble HeteroData
    # ------------------------------------------------------------------
    def create_hetero_data(self):
        print("Assembling HeteroData ...")
        data = HeteroData()

        data['review'].x = torch.tensor(self.review_features, dtype=torch.float)
        data['user'].x   = torch.tensor(self.user_features,   dtype=torch.float)
        data['product'].x = torch.tensor(self.product_features, dtype=torch.float)

        data['review'].y = torch.tensor(
            self.df['label_binary'].values, dtype=torch.long)

        for key, ei in self.edge_index_dict.items():
            data[key].edge_index = ei

        # Train / val / test masks on review nodes
        n = len(self.df)
        perm = np.random.RandomState(42).permutation(n)
        tr, va = int(0.6 * n), int(0.8 * n)
        for name, sl in [('train_mask', slice(None, tr)),
                         ('val_mask',   slice(tr, va)),
                         ('test_mask',  slice(va, None))]:
            m = torch.zeros(n, dtype=torch.bool)
            m[perm[sl]] = True
            data['review'][name] = m

        self.data = data
        return data

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    def save(self):
        print("Saving ...")
        torch.save(self.data, os.path.join(self.output_dir, 'graph_data.pt'))

        with open(os.path.join(self.output_dir, 'transformers.pkl'), 'wb') as f:
            pickle.dump({
                'sbert_model': self.sbert_model_name,
                'review_scaler': self.review_scaler,
                'user_scaler': self.user_scaler,
                'product_scaler': self.product_scaler,
                'review_feat_dim': self.review_features.shape[1],
                'user_feat_dim': self.user_features.shape[1],
                'product_feat_dim': self.product_features.shape[1],
                'temporal_cols': self.temporal_cols,
            }, f)

        meta = self.df[['user_id', 'prod_id', 'uid', 'pid',
                         'rating', 'label_binary', 'date']].copy()
        meta['date'] = meta['date'].astype(str)
        meta.to_csv(os.path.join(self.output_dir, 'review_meta.csv'), index=False)

        with open(os.path.join(self.output_dir, 'group_maps.pkl'), 'wb') as f:
            pickle.dump({
                'user_groups': {int(k): v.tolist()
                                for k, v in self.df.groupby('uid').indices.items()},
                'prod_groups': {int(k): v.tolist()
                                for k, v in self.df.groupby('pid').indices.items()},
                'uid_map': {int(k): int(v) for k, v in self.uid_map.items()},
                'pid_map': {int(k): int(v) for k, v in self.pid_map.items()},
            }, f)
        print(f"  Saved to {self.output_dir}/")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run(self):
        self.load_data()
        self.compute_text_embeddings()
        self.compute_temporal_features()
        self.build_review_features()
        self.build_user_features()
        self.build_product_features()
        self.build_hetero_edges()
        self.build_similarity_edges()
        self.create_hetero_data()
        self.save()

        d = self.data
        print(f"\n=== Heterogeneous graph summary ===")
        print(f"  Review nodes : {d['review'].x.shape[0]}  "
              f"(features={d['review'].x.shape[1]})")
        print(f"  User nodes   : {d['user'].x.shape[0]}  "
              f"(features={d['user'].x.shape[1]})")
        print(f"  Product nodes: {d['product'].x.shape[0]}  "
              f"(features={d['product'].x.shape[1]})")
        for k in d.edge_types:
            print(f"  {k}: {d[k].edge_index.shape[1]} edges")
        print(f"  Train : {d['review'].train_mask.sum().item()}")
        print(f"  Val   : {d['review'].val_mask.sum().item()}")
        print(f"  Test  : {d['review'].test_mask.sum().item()}")
        return d


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Preprocess Yelp → HeteroData")
    p.add_argument('--data_dir',      default='Yelp-Dataset')
    p.add_argument('--output_dir',    default='processed')
    p.add_argument('--max_samples',   type=int, default=200000)
    p.add_argument('--sim_threshold', type=float, default=0.8)
    args = p.parse_args()

    YelpGraphBuilder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        sim_threshold=args.sim_threshold,
    ).run()
