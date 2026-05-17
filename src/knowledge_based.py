"""
Knowledge-Based Recommendation Module — AIE425 IRS
===================================================
Constraint-based system where the user explicitly inputs filters:
  • Maximum price
  • Genre / Category
  • Minimum rating
  • Publisher / Developer (optional)

Items that satisfy ALL constraints are ranked by average community rating.
Explanation indicates exactly which constraints were matched.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Optional


class KnowledgeBasedRecommender:
    """
    Constraint-Based Knowledge Recommender (Lecture 9 / project spec).

    Users explicitly state what they want; the system filters and ranks.
    No user history needed — works well for cold-start situations.
    """

    def __init__(self):
        self.items_df: pd.DataFrame = None

    def fit(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        """
        items_df : must have parent_asin, title, price, categories, text_content
        ratings_df: optional — used to compute avg_rating per item
        """
        df = items_df.drop_duplicates("parent_asin").copy()

        if ratings_df is not None and "parent_asin" in ratings_df.columns:
            avg_ratings = (
                ratings_df.groupby("parent_asin")["rating"]
                .agg(avg_rating="mean", n_ratings="count")
                .reset_index()
            )
            df = df.merge(avg_ratings, on="parent_asin", how="left")
        else:
            df["avg_rating"] = 0.0
            df["n_ratings"]  = 0

        df["avg_rating"] = df["avg_rating"].fillna(0.0)
        df["n_ratings"]  = df["n_ratings"].fillna(0).astype(int)

        # price
        if "price" not in df.columns:
            df["price"] = 29.99
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(29.99)

        # categories as lowercase string
        if "categories" in df.columns:
            df["categories_str"] = df["categories"].fillna("").astype(str).str.lower()
        else:
            df["categories_str"] = ""

        # text for keyword search
        df["text_lower"] = df["text_content"].fillna("").astype(str).str.lower()

        self.items_df = df.reset_index(drop=True)
        return self

    def recommend(
        self,
        max_price: Optional[float]  = None,
        genre_keyword: Optional[str] = None,
        min_rating: float            = 0.0,
        min_reviews: int             = 0,
        n: int                       = 10,
        exclude_asins: set           = None,
    ) -> list:
        """
        Filter items by constraints, rank by avg_rating * log(1+n_ratings).
        Returns list of (parent_asin, score, explanation).
        """
        df = self.items_df.copy()
        applied_constraints = []

        if exclude_asins:
            df = df[~df["parent_asin"].isin(exclude_asins)]

        # ── Price filter ──────────────────────────────────────────────────────
        if max_price is not None:
            df = df[df["price"] <= max_price]
            applied_constraints.append(f"price ≤ ${max_price:.2f}")

        # ── Genre / category keyword filter ───────────────────────────────────
        if genre_keyword and genre_keyword.strip():
            kw = genre_keyword.strip().lower()
            mask = (
                df["categories_str"].str.contains(kw, na=False) |
                df["text_lower"].str.contains(kw, na=False) |
                df["title"].str.lower().str.contains(kw, na=False)
            )
            df = df[mask]
            applied_constraints.append(f"genre/keyword = '{genre_keyword}'")

        # ── Minimum community rating ──────────────────────────────────────────
        if min_rating > 0:
            df = df[df["avg_rating"] >= min_rating]
            applied_constraints.append(f"community rating ≥ {min_rating:.1f}")

        # ── Minimum number of reviews ─────────────────────────────────────────
        if min_reviews > 0:
            df = df[df["n_ratings"] >= min_reviews]
            applied_constraints.append(f"reviews ≥ {min_reviews}")

        if df.empty:
            return []

        # ── Rank: Bayesian-style score ────────────────────────────────────────
        df["score"] = df["avg_rating"] * np.log1p(df["n_ratings"])

        # normalise to [0,1]
        sc_min, sc_max = df["score"].min(), df["score"].max()
        if sc_max > sc_min:
            df["score"] = (df["score"] - sc_min) / (sc_max - sc_min)

        df = df.sort_values("score", ascending=False).head(n)

        constraint_str = " AND ".join(applied_constraints) if applied_constraints else "no specific constraints"

        results = []
        for _, row in df.iterrows():
            expl = (
                f"Recommended based on your selected requirements "
                f"({constraint_str}). "
                f"It meets your specific budget and genre preferences with a "
                f"community rating of {row['avg_rating']:.1f}/5."
            )
            results.append((row["parent_asin"], float(row["score"]), expl))

        return results

    def get_genres(self) -> list:
        """Return sorted list of unique genre keywords for the UI."""
        if self.items_df is None:
            return []
        all_cats = self.items_df["categories_str"].str.split().explode().dropna()
        genres = sorted(set(
            w.strip("[]',") for w in all_cats
            if len(w) > 3 and w not in {"and", "the", "for", "with", "game", "games"}
        ))
        return genres[:100]   # top 100 for dropdown

    def save(self, path: str):
        """Save model metadata and calculated stats, excluding the full raw dataframe to save memory."""
        if self.items_df is None:
            return
            
        # Keep only derived/essential columns to reduce pickle size significantly
        cols_to_keep = ["parent_asin", "avg_rating", "n_ratings", "categories_str", "text_lower"]
        actual_cols = [c for c in cols_to_keep if c in self.items_df.columns]
        
        original_df = self.items_df
        self.items_df = original_df[actual_cols].copy()
        
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
        self.items_df = original_df # Restore for current session

    @classmethod
    def load(cls, path: str, items_df: pd.DataFrame = None) -> "KnowledgeBasedRecommender":
        """Load the model and optionally re-attach the full items metadata."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        
        if items_df is not None and obj.items_df is not None:
            # Reconstruct the full items_df by merging saved stats with the passed metadata
            # We only merge the stats columns to avoid duplicating title/price
            stats_df = obj.items_df
            
            # Prevent x/y suffixes by dropping overlapping columns from stats_df
            overlap_cols = [c for c in stats_df.columns if c in items_df.columns and c != "parent_asin"]
            if overlap_cols:
                stats_df = stats_df.drop(columns=overlap_cols)
                
            obj.items_df = items_df.merge(stats_df, on="parent_asin", how="left")
            obj.items_df["avg_rating"] = obj.items_df["avg_rating"].fillna(0.0)
            obj.items_df["n_ratings"] = obj.items_df["n_ratings"].fillna(0).astype(int)
        return obj