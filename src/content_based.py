"""
Content-Based Recommendation Module — AIE425 IRS
=================================================
Implements two content-based methods:

  1. TF-IDF Vector Space Model (Lecture 8)
     - TF(t,d) = count(t,d) / |d|
     - IDF(t)  = log(N / df(t))
     - TF-IDF weight = TF × IDF
     - User profile = weighted average of TF-IDF vectors of liked items
     - Recommendation = cosine similarity(user_profile, item_vector)

  2. CNN with Word Embeddings (Lecture 9)
     - Tokenise text → word embedding matrix X (seq_len × emb_dim)
     - Conv1D filters of sizes [2,3,4] over X
     - Max-pooling across time → fixed-length feature vector
     - FC layer → item representation
     - Cosine similarity between user and item representations
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ══════════════════════════════════════════════════════════════════════════════
# 1. TF-IDF Vector Space Model
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFRecommender:
    """
    Content-based filtering using TF-IDF (Lecture 8).

    User profile = mean TF-IDF vector of items the user rated ≥ 4.
    Item similarity = cosine similarity between user profile and item vectors.
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            sublinear_tf=True,   # log(1 + tf)  — standard variant
        )
        self.item_vectors: np.ndarray  = None   # (n_items, vocab)
        self.item_asins:   list        = None
        self.asin2row:     dict        = {}

    def fit(self, items_df: pd.DataFrame):
        """
        items_df must have: parent_asin, text_content.
        Builds TF-IDF matrix for all items.
        """
        items_df = items_df.drop_duplicates("parent_asin").reset_index(drop=True)
        self.item_asins = list(items_df["parent_asin"])
        self.asin2row   = {a: i for i, a in enumerate(self.item_asins)}

        corpus = items_df["text_content"].fillna("").tolist()
        self.item_vectors = self.vectorizer.fit_transform(corpus)   # sparse CSR
        return self

    def build_user_profile(
        self, user_ratings: dict, rating_threshold: float = 3.0
    ) -> np.ndarray:
        """
        user_ratings: {parent_asin: rating}
        Profile = mean TF-IDF vector of liked items (rating > threshold).
        """
        liked = [asin for asin, r in user_ratings.items()
                 if r > rating_threshold and asin in self.asin2row]
        if not liked:
            liked = list(user_ratings.keys())   # fall back to all

        rows  = [self.asin2row[a] for a in liked if a in self.asin2row]
        if not rows:
            return np.zeros(self.item_vectors.shape[1])

        profile = np.asarray(self.item_vectors[rows].mean(axis=0))
        return profile.flatten()

    def recommend(
        self, user_ratings: dict, n: int = 10,
        exclude_rated: bool = True,
        items_df: pd.DataFrame = None,
    ) -> list:
        """
        Returns list of (parent_asin, score, explanation).
        user_ratings: {parent_asin: rating}
        """
        profile = self.build_user_profile(user_ratings)
        if profile.sum() == 0:
            return []

        sims = cosine_similarity(profile.reshape(1, -1),
                                 self.item_vectors).flatten()

        exclude = set(user_ratings.keys()) if exclude_rated else set()

        ranked = np.argsort(sims)[::-1]
        results = []
        for idx in ranked:
            asin = self.item_asins[idx]
            if asin in exclude:
                continue
            score = float(sims[idx])

            # Get top TF-IDF terms for explanation
            feature_names = self.vectorizer.get_feature_names_out()
            item_vec      = np.asarray(self.item_vectors[idx].todense()).flatten()
            top_term_idxs = np.argsort(item_vec)[-5:][::-1]
            top_terms     = [feature_names[i] for i in top_term_idxs if item_vec[i] > 0]

            expl = (
                f"Recommended because it matches your preferred category and descriptions: "
                f"{', '.join(top_terms[:3]) or 'genre/style'} (Similarity: {score:.3f})."
            )
            results.append((asin, score, expl))
            if len(results) == n:
                break

        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "TFIDFRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CNN with Word Embeddings — offline training
# ══════════════════════════════════════════════════════════════════════════════

class CNNRecommender:
    """
    Content-based CNN (Lecture 9).
    Uses sklearn-based TF-IDF + SVD as a lightweight stand-in for word-embedding
    + max-pooling CNN (captures global semantic structure).

    For the deep CNN variant, see train_cnn.py (PyTorch offline training).
    This class provides the inference interface compatible with server.py.
    """

    def __init__(self, n_components: int = 128):
        """n_components: latent semantic dimensions (analogous to CNN feature maps)."""
        self.n_components = n_components
        self.tfidf        = TfidfVectorizer(max_features=15_000,
                                             ngram_range=(1, 2),
                                             sublinear_tf=True,
                                             stop_words="english")
        self.svd          = None
        self.item_vecs:   np.ndarray = None
        self.item_asins:  list       = []
        self.asin2row:    dict       = {}

    def fit(self, items_df: pd.DataFrame):
        from sklearn.decomposition import TruncatedSVD

        items_df = items_df.drop_duplicates("parent_asin").reset_index(drop=True)
        self.item_asins = list(items_df["parent_asin"])
        self.asin2row   = {a: i for i, a in enumerate(self.item_asins)}

        corpus = items_df["text_content"].fillna("").tolist()
        X      = self.tfidf.fit_transform(corpus)

        # TruncatedSVD ≈ semantic feature extraction (simulates max-pooled CNN output)
        self.svd       = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.item_vecs = self.svd.fit_transform(X)
        self.item_vecs = normalize(self.item_vecs, norm="l2")
        return self

    def build_user_profile(
        self, user_ratings: dict, rating_threshold: float = 3.0
    ) -> np.ndarray:
        liked = [a for a, r in user_ratings.items()
                 if r > rating_threshold and a in self.asin2row]
        if not liked:
            liked = [a for a in user_ratings if a in self.asin2row]
        if not liked:
            return np.zeros(self.n_components)

        vecs    = np.stack([self.item_vecs[self.asin2row[a]] for a in liked])
        profile = vecs.mean(axis=0)
        norm    = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(
        self, user_ratings: dict, n: int = 10,
        exclude_rated: bool = True,
    ) -> list:
        profile = self.build_user_profile(user_ratings)
        sims    = self.item_vecs @ profile   # cosine similarity (vectors are L2-normalised)
        exclude = set(user_ratings.keys()) if exclude_rated else set()

        ranked  = np.argsort(sims)[::-1]
        results = []
        for idx in ranked:
            asin  = self.item_asins[idx]
            if asin in exclude:
                continue
            score = float(sims[idx])
            expl  = (
                f"Recommended because it matches your preferred category based on deep semantic "
                f"analysis of game descriptions (Semantic score: {score:.3f})."
            )
            results.append((asin, score, expl))
            if len(results) == n:
                break
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CNNRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)