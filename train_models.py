"""
=============================================================================
  BollyWise Recommender Engine — Offline Model Training Pipeline
=============================================================================
This script trains:
1. Baseline Popularity Model (models/popularity_model.pkl)
2. Personalized Matrix Factorization SVD (models/svd_model.pkl)
   - 32 Latent Factors
   - Regularized Funk-SVD with User and Item Biases
   - Stochastic Gradient Descent (SGD)

Usage:
    python train_models.py
=============================================================================
"""

import os
import sys
import time
import types
import pickle
import numpy as np
import pandas as pd

# Root Directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Register stub module so models can be unpickled everywhere seamlessly
src_mod = types.ModuleType("src")
training_mod = types.ModuleType("training")
models_mod = types.ModuleType("models")
sys.modules["src"] = src_mod
sys.modules["src.training"] = training_mod
sys.modules["src.training.models"] = models_mod


# =============================================================================
# 1. Baseline Popularity Recommender
# =============================================================================
class PopularityRecommender:
    """
    Baseline non-personalized recommender ranking items by rating volume and mean score.
    Score = rating_count * mean_rating
    """
    def __init__(self):
        self.item_scores = {}
        self.ranked_items = []
        self.user_history = {}
        self.global_mean = 3.529

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        for uid, grp in train_df.groupby("user_id"):
            self.user_history[int(uid)] = set(grp["item_id"].astype(int))

        stats = train_df.groupby("item_id").agg(count=("rating", "count"), mean=("rating", "mean"))
        scores = {int(iid): float(row["count"] * row["mean"]) for iid, row in stats.iterrows()}
        self.item_scores = scores
        self.ranked_items = [iid for iid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        return float(self.global_mean)

    def recommend(self, user_id: int, top_k: int = 10) -> list:
        history = self.user_history.get(user_id, set())
        return [iid for iid in self.ranked_items if iid not in history][:top_k]


# =============================================================================
# 2. Personalized Matrix Factorization (Funk-SVD)
# =============================================================================
class MatrixFactorizationSVD:
    """
    Latent Factor Collaborative Filtering with User & Item Biases (Funk-SVD).
    
    Optimization Objective:
        min sum (r_ui - (mu + b_u + b_i + P_u . Q_i))^2 + lambda * (||P_u||^2 + ||Q_i||^2 + b_u^2 + b_i^2)
    
    Parameters:
        n_factors: int = 32 (Latent dimension k)
        lr: float = 0.005 (SGD Learning rate gamma)
        reg: float = 0.02 (L2 Regularization parameter lambda)
        n_epochs: int = 15 (Number of full passes over training ratings)
        random_state: int = 42
    """
    def __init__(self, n_factors: int = 32, lr: float = 0.005, reg: float = 0.02, n_epochs: int = 15, random_state: int = 42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.global_mean = 3.529
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_history = {}
        self.bu = None
        self.bi = None
        self.P = None
        self.Q = None

    def fit(self, train_df: pd.DataFrame, items_df: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        self.global_mean = float(train_df["rating"].mean())

        for uid, grp in train_df.groupby("user_id"):
            self.user_history[int(uid)] = set(grp["item_id"].astype(int))

        users = sorted(train_df["user_id"].unique())
        items = sorted(items_df["item_id"].unique())
        self.user_to_idx = {uid: idx for idx, uid in enumerate(users)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(items)}
        self.idx_to_item = {idx: iid for idx, iid in enumerate(items)}

        n_u = len(users)
        n_i = len(items)

        # Initialize baseline biases and latent factor matrices
        self.bu = np.zeros(n_u, dtype=np.float32)
        self.bi = np.zeros(n_i, dtype=np.float32)
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors)).astype(np.float32)

        u_arr = np.array([self.user_to_idx[uid] for uid in train_df["user_id"]], dtype=np.int32)
        i_arr = np.array([self.item_to_idx[iid] for iid in train_df["item_id"]], dtype=np.int32)
        r_arr = train_df["rating"].values.astype(np.float32)
        n_samples = len(r_arr)

        print(f"      Users in training set: {n_u:,}")
        print(f"      Items in catalog:      {n_i:,}")
        print(f"      Total ratings:         {n_samples:,}")
        print(f"      Global rating mean mu: {self.global_mean:.4f}")
        print(f"      Latent factors k:      {self.n_factors}")
        print(f"      Regularization lambda: {self.reg}")
        print(f"      Learning rate gamma:   {self.lr}")
        print(f"      Total SGD epochs:      {self.n_epochs}")
        print("-" * 65)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            sse = 0.0

            # Shuffle ratings each epoch for stable SGD convergence
            indices = rng.permutation(n_samples)

            for idx in indices:
                u = u_arr[idx]
                i = i_arr[idx]
                r = r_arr[idx]

                # 1. Forward pass prediction
                pred = self.global_mean + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                sse += err * err

                # 2. Stochastic Gradient Descent Parameter Updates
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])

            rmse = np.sqrt(sse / n_samples)
            elapsed = time.time() - t0
            print(f"      Epoch {epoch:2d}/{self.n_epochs:2d} | Train RMSE: {rmse:.4f} | Time: {elapsed:.2f}s")

        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id in self.user_to_idx and item_id in self.item_to_idx:
            u = self.user_to_idx[user_id]
            i = self.item_to_idx[item_id]
            pred = self.global_mean + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
        else:
            pred = self.global_mean
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, top_k: int = 10) -> list:
        if user_id not in self.user_to_idx:
            return list(self.idx_to_item.values())[:top_k]
        u = self.user_to_idx[user_id]
        history = self.user_history.get(user_id, set())

        scores = self.global_mean + self.bu[u] + self.bi + np.dot(self.Q, self.P[u])
        candidates = [(self.idx_to_item[i], score) for i, score in enumerate(scores) if self.idx_to_item[i] not in history]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in candidates[:top_k]]


# Attach to module for pickle serialization compatibility
models_mod.MatrixFactorizationSVD = MatrixFactorizationSVD
models_mod.PopularityRecommender = PopularityRecommender


# =============================================================================
# 3. Main Training Orchestration
# =============================================================================
def train_all_models():
    print("=" * 65)
    print("  BollyWise Recommender System — Model Training Pipeline")
    print("=" * 65)

    train_path = os.path.join(DATA_DIR, "train_interactions.parquet")
    items_path = os.path.join(DATA_DIR, "cleaned_items.parquet")

    if not os.path.exists(train_path) or not os.path.exists(items_path):
        raise FileNotFoundError(f"Missing data files: {train_path} or {items_path}")

    print("\n[Step 1/3] Loading Processed Training Data...")
    train_df = pd.read_parquet(train_path)
    items_df = pd.read_parquet(items_path)
    print(f"           Loaded {len(train_df):,} interaction rows and {len(items_df):,} items.")

    print("\n[Step 2/3] Training Baseline Popularity Model...")
    pop_model = PopularityRecommender().fit(train_df, items_df)
    pop_path = os.path.join(MODELS_DIR, "popularity_model.pkl")
    with open(pop_path, "wb") as f:
        pickle.dump(pop_model, f)
    print(f"           Popularity model saved -> {pop_path}")

    print("\n[Step 3/3] Training Personalized Matrix Factorization (Funk-SVD)...")
    svd_model = MatrixFactorizationSVD(
        n_factors=32,
        lr=0.005,
        reg=0.02,
        n_epochs=15,
        random_state=42
    ).fit(train_df, items_df)

    svd_path = os.path.join(MODELS_DIR, "svd_model.pkl")
    with open(svd_path, "wb") as f:
        pickle.dump(svd_model, f)

    print("-" * 65)
    print(f"           SVD model saved -> {svd_path}")
    print(f"           Item Latent Matrix Q shape: {svd_model.Q.shape}")
    print(f"           User Latent Matrix P shape: {svd_model.P.shape}")
    print("=" * 65)
    print("  Training Successfully Finished! Artifacts ready in models/")
    print("=" * 65)


if __name__ == "__main__":
    train_all_models()
