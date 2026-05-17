"""
Data Preprocessing Module — AIE425 Intelligent Recommender System
Amazon Video Games Dataset (Reviews.json + Meta.json)

Steps:
  1. Load JSON Lines files
  2. Merge on parent_asin
  3. K-Core filtering (min 5 interactions per user and item)
  4. Feature engineering: description/features/categories → text_content
  5. Price imputation with median
  6. Generate contiguous user_idx / item_idx mappings
  7. Save processed artefacts for training & inference
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSON-Lines file robustly (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)


def flatten_list_col(series: pd.Series) -> pd.Series:
    """Convert list/nested-list column to a flat concatenated string."""
    def _flatten(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, list):
                    parts.extend([str(x) for x in item])
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(val)
    return series.apply(_flatten)


def kcore_filter(df: pd.DataFrame, user_col: str, item_col: str, k: int = 5) -> pd.DataFrame:
    """
    Iteratively remove users and items with fewer than k interactions
    until convergence (standard K-Core procedure from lecture).
    """
    while True:
        prev_len = len(df)
        user_counts = df[user_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= k].index)]
        item_counts = df[item_col].value_counts()
        df = df[df[item_col].isin(item_counts[item_counts >= k].index)]
        if len(df) == prev_len:
            break
    return df.reset_index(drop=True)


# ─── main preprocessing pipeline ──────────────────────────────────────────────

def run_preprocessing(
    reviews_path: str,
    meta_path: str,
    output_dir: str,
    k: int = 5,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict of key processed DataFrames and mappings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1 — Loading Reviews.json ...")
    reviews = load_jsonl(reviews_path)
    print(f"  Reviews loaded: {len(reviews):,} rows, cols={list(reviews.columns)}")

    print("Step 2 — Loading Meta.json ...")
    meta = load_jsonl(meta_path)
    print(f"  Meta loaded:    {len(meta):,} rows, cols={list(meta.columns)}")

    # ── normalise column names ────────────────────────────────────────────────
    # Reviews must have: user_id, parent_asin, rating
    for col in ["user_id", "parent_asin", "rating"]:
        if col not in reviews.columns:
            raise ValueError(f"Reviews file missing expected column: '{col}'")

    # keep only essential review columns
    keep_rev = ["user_id", "parent_asin", "rating", "timestamp"]
    keep_rev = [c for c in keep_rev if c in reviews.columns]
    reviews = reviews[keep_rev].copy()
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews.dropna(subset=["rating", "user_id", "parent_asin"], inplace=True)

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    print("Step 3 — Merging on parent_asin ...")
    meta_key = "parent_asin" if "parent_asin" in meta.columns else "asin"
    if meta_key not in meta.columns:
        raise ValueError("Meta file has no 'parent_asin' or 'asin' column.")
    meta = meta.rename(columns={meta_key: "parent_asin"})
    df = reviews.merge(meta, on="parent_asin", how="left")
    print(f"  Merged shape: {df.shape}")

    # ── Step 4: K-Core Filtering ──────────────────────────────────────────────
    print(f"Step 4 — K-Core filtering (k={k}) ...")
    df_cf = kcore_filter(df.copy(), "user_id", "parent_asin", k=k)
    print(f"  After K-Core: {len(df_cf):,} interactions, "
          f"{df_cf['user_id'].nunique():,} users, "
          f"{df_cf['parent_asin'].nunique():,} items")

    # ── Step 5: Feature Engineering ───────────────────────────────────────────
    print("Step 5 — Feature engineering ...")

    # flatten list columns
    for col in ["description", "features", "categories"]:
        if col in df.columns:
            df[col] = flatten_list_col(df[col])
        else:
            df[col] = ""

    # title
    if "title" not in df.columns:
        df["title"] = ""
    df["title"] = df["title"].fillna("").astype(str)

    # text_content = title + description + features
    df["text_content"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["features"].fillna("")
    ).str.strip()

    # price imputation with median (as per project spec)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        median_price = df["price"].median()
        df["price"] = df["price"].fillna(median_price)
        print(f"  Median price used for imputation: {median_price:.2f}")
    else:
        df["price"] = 29.99   # default placeholder

    # ── Step 6: ID Mapping ────────────────────────────────────────────────────
    print("Step 6 — Generating contiguous ID mappings ...")

    # full dataset (for content/knowledge-based)
    all_items = df.drop_duplicates("parent_asin")[["parent_asin", "title", "text_content",
                                                    "description", "features", "categories",
                                                    "price"]].copy()
    item2idx = {asin: idx for idx, asin in enumerate(all_items["parent_asin"])}
    idx2item = {v: k for k, v in item2idx.items()}

    # k-core dataset (for CF models)
    cf_users = sorted(df_cf["user_id"].unique())
    cf_items = sorted(df_cf["parent_asin"].unique())
    user2idx = {u: i for i, u in enumerate(cf_users)}
    cf_item2idx = {p: i for i, p in enumerate(cf_items)}
    idx2user = {v: k for k, v in user2idx.items()}
    cf_idx2item = {v: k for k, v in cf_item2idx.items()}

    df_cf = df_cf.copy()
    df_cf["user_idx"] = df_cf["user_id"].map(user2idx)
    df_cf["item_idx"] = df_cf["parent_asin"].map(cf_item2idx)

    n_users = len(cf_users)
    n_items = len(cf_items)
    print(f"  CF matrix: {n_users} users × {n_items} items")

    # ── Build user–item rating matrix (sparse-friendly) ───────────────────────
    print("Step 7 — Building rating matrix ...")
    rating_matrix = df_cf.pivot_table(
        index="user_idx", columns="item_idx", values="rating", aggfunc="mean"
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("Step 8 — Saving artefacts ...")

    df_cf.to_parquet(output_dir / "cf_interactions.parquet", index=False)
    all_items.to_parquet(output_dir / "items_meta.parquet", index=False)
    df[["parent_asin", "title", "text_content", "description",
        "features", "categories", "price"]].drop_duplicates("parent_asin").to_parquet(
        output_dir / "items_full.parquet", index=False
    )
    rating_matrix.to_parquet(output_dir / "rating_matrix.parquet")

    mappings = {
        "user2idx": user2idx,
        "idx2user": idx2user,
        "cf_item2idx": cf_item2idx,
        "cf_idx2item": cf_idx2item,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "n_users": n_users,
        "n_items": n_items,
        "n_all_items": len(all_items),
    }
    with open(output_dir / "mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)

    print(f"\n✅  Preprocessing complete. Artefacts saved to: {output_dir}")
    print(f"    CF interactions : {len(df_cf):,}")
    print(f"    Unique users    : {n_users:,}")
    print(f"    Unique CF items : {n_items:,}")
    print(f"    All items (meta): {len(all_items):,}")

    return {
        "df_cf": df_cf,
        "all_items": all_items,
        "rating_matrix": rating_matrix,
        "mappings": mappings,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Amazon Video Games dataset")
    parser.add_argument("--reviews", required=True, help="Path to Reviews.json (JSONL)")
    parser.add_argument("--meta",    required=True, help="Path to Meta.json (JSONL)")
    parser.add_argument("--output",  default="data/processed", help="Output directory")
    parser.add_argument("--k",       type=int, default=5, help="K-Core threshold")
    args = parser.parse_args()

    run_preprocessing(args.reviews, args.meta, args.output, k=args.k)
