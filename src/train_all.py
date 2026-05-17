"""
Offline Training Script — AIE425 IRS
======================================
Run this ONCE to train and save all models.
The Flask server.py only loads pre-trained models for fast inference.

Usage:
  python src/train_all.py --data data/processed --output models/saved

Pipeline:
  1. Load preprocessed data
  2. Train User-Based CF (Pearson)          → ubcf.pkl
  3. Train Item-Based CF (Adjusted Cosine)  → ibcf.pkl
  4. Train Matrix Factorization (GD)        → mf.pkl
  5. Train NCF (PyTorch)                    → ncf_model.pt
  6. Fit TF-IDF Recommender                 → tfidf.pkl
  7. Fit CNN Semantic Recommender            → cnn.pkl
  8. Fit Knowledge-Based Recommender        → kb.pkl
  9. Compute evaluation metrics             → eval_results.pkl
 10. Save summary                           → training_summary.pkl
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

from collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization
from ncf_model                import NCFRecommender, train_ncf
from content_based            import TFIDFRecommender, CNNRecommender
from knowledge_based          import KnowledgeBasedRecommender
from evaluation               import RecommenderEvaluator, mae, rmse, binarise, \
                                      precision_score, recall_score, f_score

import torch


def main(data_dir: str, output_dir: str, quick_mode: bool = False):
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n📂  Loading preprocessed data ...")
    df_cf      = pd.read_parquet(data_dir / "cf_interactions.parquet")
    items_df   = pd.read_parquet(data_dir / "items_full.parquet")
    rating_mat = pd.read_parquet(data_dir / "rating_matrix.parquet")

    with open(data_dir / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    n_users = mappings["n_users"]
    n_items = mappings["n_items"]
    print(f"  {n_users} users, {n_items} CF items, {len(df_cf):,} interactions")

    # ── Train/test split for evaluation ───────────────────────────────────────
    test_df = df_cf.sample(frac=0.2, random_state=42)
    train_df = df_cf.drop(test_df.index)

    # ── 1. User-Based CF (Pearson) ─────────────────────────────────────────────
    print("\n🔨  Training User-Based CF (Pearson) ...")
    # Build rating matrix from train split
    train_mat = train_df.pivot_table(
        index="user_idx", columns="item_idx", values="rating", aggfunc="mean"
    )
    ubcf = UserBasedCF(n_neighbors=30)
    ubcf.fit(train_mat)
    ubcf.save(output_dir / "ubcf.pkl")
    print("  ✅  Saved ubcf.pkl")

    # ── 2. Item-Based CF (Adjusted Cosine) ────────────────────────────────────
    print("\n🔨  Training Item-Based CF (Adjusted Cosine) ...")
    ibcf = ItemBasedCF(n_neighbors=30)
    ibcf.fit(train_mat)
    ibcf.save(output_dir / "ibcf.pkl")
    print("  ✅  Saved ibcf.pkl")

    # ── 3. Matrix Factorization ────────────────────────────────────────────────
    print("\n🔨  Training Matrix Factorization (Gradient Descent) ...")
    n_epochs_mf = 20 if quick_mode else 50
    mf = MatrixFactorization(n_factors=20, lr=0.01, reg=0.1, n_epochs=n_epochs_mf)
    mf.fit(train_df, n_users=n_users, n_items=n_items)
    mf.save(output_dir / "mf.pkl")
    print("  ✅  Saved mf.pkl")

    # ── 4. Neural CF ──────────────────────────────────────────────────────────
    print("\n🔨  Training Neural Collaborative Filtering (NCF) ...")
    ncf_path   = str(output_dir / "ncf_model.pt")
    n_epochs_ncf = 10 if quick_mode else 20
    ncf_model, ncf_history = train_ncf(
        train_df, n_users=n_users, n_items=n_items,
        emb_dim=64, lr=0.001, n_epochs=n_epochs_ncf,
        batch_size=1024, save_path=ncf_path,
    )
    ncf_rec = NCFRecommender(ncf_model, n_users, n_items)
    print("  ✅  Saved ncf_model.pt")

    # ── Helper for Content-Based and Knowledge-Based predict_fn ───────────────
    # Pre-calculate user_ratings_dict for all users in train_df for efficiency
    user_train_ratings = {}
    for user_id, group in train_df.groupby("user_id"):
        user_train_ratings[user_id] = dict(zip(group["parent_asin"], group["rating"]))

    # N_FOR_EVAL for content/knowledge-based models to determine "recommended"
    N_FOR_EVAL = 10

    # --- TF-IDF predict_fn ---
    def tfidf_predict_fn(user_idx_cf, item_idx_cf):
        user_id = mappings["idx2user"][user_idx_cf]
        parent_asin = mappings["cf_idx2item"][item_idx_cf]
        
        if user_id not in user_train_ratings:
            # If user has no training history, cannot make personalized content-based rec
            # Fallback to a neutral prediction
            return 1.0 

        user_ratings_dict = user_train_ratings[user_id]
        # TF-IDF recommender expects parent_asin, not item_idx
        recs = tfidf_rec.recommend(user_ratings_dict, n=N_FOR_EVAL, exclude_rated=False)
        
        # Check if the item is in the recommended list
        for asin, score, expl in recs:
            if asin == parent_asin:
                return 5.0 # Recommended
        return 1.0 # Not recommended

    # --- CNN predict_fn ---
    def cnn_predict_fn(user_idx_cf, item_idx_cf):
        user_id = mappings["idx2user"][user_idx_cf]
        parent_asin = mappings["cf_idx2item"][item_idx_cf]

        if user_id not in user_train_ratings:
            return 1.0

        user_ratings_dict = user_train_ratings[user_id]
        recs = cnn_rec.recommend(user_ratings_dict, n=N_FOR_EVAL, exclude_rated=False)
        
        for asin, score, expl in recs:
            if asin == parent_asin:
                return 5.0
        return 1.0

    # --- Knowledge-Based predict_fn ---
    def kb_predict_fn(user_idx_cf, item_idx_cf):
        user_id = mappings["idx2user"][user_idx_cf]
        parent_asin = mappings["cf_idx2item"][item_idx_cf]

        # If user has no training history, use general constraints for KB
        if user_id not in user_train_ratings:
            recs = kb_rec.recommend(
                max_price=60.0, # Default max price
                genre_keyword=None, # No specific genre for cold-start
                min_rating=3.0, # Default min rating
                min_reviews=5,  # Default min reviews
                n=N_FOR_EVAL,
                exclude_asins=set()
            )
            for asin, score, expl in recs:
                if asin == parent_asin:
                    return 5.0
            return 1.0

        user_ratings_dict = user_train_ratings[user_id]
        
        # Derive pseudo-constraints from user's liked items in train_df
        liked_items_asins = [asin for asin, r in user_ratings_dict.items() if r >= 4.0]
        
        derived_max_price = None
        derived_genre_kw = None
        
        if liked_items_asins:
            liked_items_meta = items_df[items_df["parent_asin"].isin(liked_items_asins)]
            if not liked_items_meta.empty:
                derived_max_price = liked_items_meta["price"].max() * 1.2 # Allow slightly higher price
                
                # Most frequent category/genre from liked items
                all_categories = liked_items_meta["categories"].apply(lambda x: x.split(',') if isinstance(x, str) else []).explode()
                all_categories = all_categories.str.strip().str.lower()
                if not all_categories.empty:
                    derived_genre_kw = all_categories.mode()
        
        recs = kb_rec.recommend(
            max_price=derived_max_price,
            genre_keyword=derived_genre_kw,
            min_rating=3.0, # Fixed minimum rating for evaluation
            min_reviews=5,  # Fixed minimum reviews for evaluation
            n=N_FOR_EVAL,
            exclude_asins=set(user_ratings_dict.keys()) # Exclude items already rated by user
        )
        
        for asin, score, expl in recs:
            if asin == parent_asin:
                return 5.0
        return 1.0

    # ── 5. TF-IDF Recommender ─────────────────────────────────────────────────
    print("\n🔨  Fitting TF-IDF Content-Based Recommender ...")
    tfidf_rec = TFIDFRecommender(max_features=10_000)
    tfidf_rec.fit(items_df)
    tfidf_rec.save(output_dir / "tfidf.pkl")
    print("  ✅  Saved tfidf.pkl")

    # ── 6. CNN Semantic Recommender ───────────────────────────────────────────
    print("\n🔨  Fitting CNN Semantic Content-Based Recommender ...")
    cnn_rec = CNNRecommender(n_components=128)
    cnn_rec.fit(items_df)
    cnn_rec.save(output_dir / "cnn.pkl")
    print("  ✅  Saved cnn.pkl")

    # ── 7. Knowledge-Based Recommender ────────────────────────────────────────
    print("\n🔨  Fitting Knowledge-Based Recommender ...")
    kb_rec = KnowledgeBasedRecommender()
    kb_rec.fit(items_df, ratings_df=df_cf.merge(
        items_df[["parent_asin"]], left_on="parent_asin", right_on="parent_asin", how="inner"
    ) if "parent_asin" in df_cf.columns else None)
    kb_rec.save(output_dir / "kb.pkl")
    print("  ✅  Saved kb.pkl")
   
    # ── 8. Evaluation ─────────────────────────────────────────────────────────
    print("\n📊  Evaluating all 4 CF models on test set ...")
    print("\n📊  Evaluating all 7 models across 3 approaches (CF, Content, Knowledge) ...")

    evaluator = RecommenderEvaluator(test_df)

    def ncf_predict_rating(u, i):
        u_t = torch.tensor([u], dtype=torch.long)
        i_t = torch.tensor([i], dtype=torch.long)
        with torch.no_grad():
            score = ncf_model(u_t, i_t).item()
        return 1 + score * 4   # map (0,1) → (1,5)

    eval_models = [
        {"name": "User-Based CF (Pearson)",      "predict_fn": ubcf.predict},
        {"name": "Item-Based CF (Adj. Cosine)",  "predict_fn": ibcf.predict},
        {"name": "Matrix Factorization (GD)",    "predict_fn": mf.predict},
        {"name": "Neural CF (NCF)",              "predict_fn": ncf_predict_rating},
        {"name": "TF-IDF Content-Based",         "predict_fn": tfidf_predict_fn},
        {"name": "CNN Semantic Content-Based",   "predict_fn": cnn_predict_fn},
        {"name": "Knowledge-Based",              "predict_fn": kb_predict_fn},
    ]

    eval_results = evaluator.compare_models(eval_models)
    print("\n" + eval_results.to_string())

    with open(output_dir / "eval_results.pkl", "wb") as f:
        pickle.dump(eval_results, f)
    print("  ✅  Saved eval_results.pkl")

    # ── 9. Save summary ───────────────────────────────────────────────────────
    summary = {
        "n_users":        n_users,
        "n_items":        n_items,
        "n_interactions": len(df_cf),
        "n_all_items":    len(items_df),
        "mappings":       mappings,
        "ncf_history":    ncf_history,
        "mf_losses":      mf.train_losses,
    }
    with open(output_dir / "training_summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print("\n✅  All models trained and saved successfully!")
    print(f"    Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/processed", help="Processed data directory")
    parser.add_argument("--output", default="models/saved",   help="Model output directory")
    parser.add_argument("--quick",  action="store_true",       help="Quick training (fewer epochs)")
    args = parser.parse_args()
    main(args.data, args.output, quick_mode=args.quick)
