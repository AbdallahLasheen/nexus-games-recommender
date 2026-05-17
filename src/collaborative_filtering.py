"""
Collaborative Filtering Module — AIE425 IRS
============================================
Implements four CF methods exactly as taught in the lectures:

  1. User-Based Nearest Neighbor  → Pearson Correlation (Lecture 2, Formula 2.1)
     pred(a,p) = r̄_a + Σ sim(a,b)*(r_{b,p} - r̄_b) / Σ|sim(a,b)|

  2. Item-Based Nearest Neighbor  → Adjusted Cosine Similarity (Lecture 4)
     sim(i,j) = Σ_u (r_{u,i}-r̄_u)(r_{u,j}-r̄_u) /
                sqrt(Σ_u(r_{u,i}-r̄_u)²) * sqrt(Σ_u(r_{u,j}-r̄_u)²)
     pred(u,i) = Σ_j sim(i,j)*r_{u,j} / Σ|sim(i,j)|

  3. Matrix Factorization via Gradient Descent (Lecture 6)
     R̂_{ui} = P_u · Q_i^T
     update: P_uf += α*(2*e_{ui}*Q_if - λ*P_uf)
             Q_if += α*(2*e_{ui}*P_uf - λ*Q_if)

  4. Neural Collaborative Filtering (Lecture 7)
     Architecture: Embedding → Concatenate → FC(ReLU) × 3 → FC(Sigmoid)
     Trained offline and saved as .pt file.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 1. User-Based Nearest Neighbor (Pearson Correlation)
# ══════════════════════════════════════════════════════════════════════════════

class UserBasedCF:
    """
    User-Based Nearest Neighbor Recommendation.
    Similarity: Pearson Correlation Coefficient (Lecture 2, Formula 2.1).
    Prediction: weighted deviation from mean (Lecture 2, slide 17).
    """

    def __init__(self, n_neighbors: int = 30):
        self.n_neighbors = n_neighbors
        self.rating_matrix: pd.DataFrame = None
        self.user_means: pd.Series = None
        self.user_norms: np.ndarray = None

    def fit(self, rating_matrix: pd.DataFrame):
        self.rating_matrix = rating_matrix.copy().astype(float)
        self.user_means = self.rating_matrix.mean(axis=1)
        self._calculate_norms()
        return self

    def _calculate_norms(self):
        """حساب المعايير (norms) اللازمة لتشابه بيرسون دون استهلاك ذاكرة عالي."""
        n_users = len(self.rating_matrix)
        norms = np.zeros(n_users)
        # استخراج القيم كمصفوفة numpy لتسريع المعالجة اليدوية
        matrix_values = self.rating_matrix.values
        means_values = self.user_means.values
        
        for i in range(n_users):
            row = matrix_values[i]
            mask = ~np.isnan(row)
            valid_data = row[mask]
            if valid_data.size > 0:
                norms[i] = np.sqrt(np.sum((valid_data - means_values[i])**2))
        self.user_norms = norms
        self.user_norms[self.user_norms == 0] = 1e-9
    def _ensure_norms(self):
        """تأكد من وجود user_norms (للتوافق مع الملفات القديمة)."""
        if not hasattr(self, 'user_norms') or self.user_norms is None:
            self._calculate_norms()

    def predict(self, user_a: int, item_p: int) -> float:
        if user_a not in self.rating_matrix.index or item_p not in self.rating_matrix.columns:
            return self.user_means.mean()

        # تأكد من التوافق مع الموديل القديم
        self._ensure_norms()

        # Memory efficient similarity: only center the intersection of ratings
        user_ratings = self.rating_matrix.loc[user_a].dropna()
        item_col = self.rating_matrix[item_p].dropna()
        neighbor_ids = item_col.index[item_col.index != user_a]
        
        if len(neighbor_ids) == 0: return self.user_means[user_a]

        # Compute similarity between user_a and candidate neighbors
        sub_matrix = self.rating_matrix.loc[neighbor_ids, user_ratings.index]
        sub_centered = sub_matrix.sub(self.user_means.loc[neighbor_ids], axis=0).fillna(0).values
        u_vec = (user_ratings - self.user_means[user_a]).values
        
        dot_prods = sub_centered @ u_vec
        u_idx = self.rating_matrix.index.get_loc(user_a)
        neighbor_idxs = self.rating_matrix.index.get_indexer(neighbor_ids)
        sims = dot_prods / (self.user_norms[neighbor_idxs] * self.user_norms[u_idx] + 1e-9)
        
        # Get top N
        top_local = np.argsort(sims)[-self.n_neighbors:]
        best_sims = sims[top_local]
        best_neighbor_ids = neighbor_ids[top_local]
        
        # Weighted average of deviations
        devs = item_col.loc[best_neighbor_ids].values - self.user_means.loc[best_neighbor_ids].values
        sum_abs_sim = np.sum(np.abs(best_sims))
        
        if sum_abs_sim == 0: return self.user_means[user_a]
        return float(self.user_means[user_a] + np.sum(best_sims * devs) / sum_abs_sim)

    def recommend(self, user_a: int, n: int = 10, exclude_rated: bool = True) -> list:
        if user_a not in self.rating_matrix.index:
            return []

        # تأكد من التوافق مع الموديل القديم
        self._ensure_norms()

        # 1. Efficient Similarity Calculation
        user_ratings = self.rating_matrix.loc[user_a].dropna()
        u_centered_vals = (user_ratings - self.user_means[user_a]).values
        
        # بدلاً من إنشاء مصفوفة فرعية ضخمة، نحسب الضرب النقطي عموداً بعمود
        # هذا يمنع تخصيص الـ 2.8 جيجابايت تماماً
        dot_prods = np.zeros(len(self.rating_matrix))
        user_means_vals = self.user_means.values
        
        for asin, weight in zip(user_ratings.index, u_centered_vals):
            # نأخذ عمود واحد فقط في الذاكرة
            col = self.rating_matrix[asin].values
            mask = ~np.isnan(col)
            # نجمع مساهمة هذا العمود في التشابه
            dot_prods[mask] += (col[mask] - user_means_vals[mask]) * weight

        u_idx = self.rating_matrix.index.get_loc(user_a)
        user_sims = dot_prods / (self.user_norms * self.user_norms[u_idx] + 1e-9)

        # 2. Get top neighbors
        neighbor_idx = np.argsort(user_sims)[::-1]
        neighbor_idx = neighbor_idx[neighbor_idx != u_idx]
        top_n_idx = neighbor_idx[:self.n_neighbors]
        
        # 3. Predict for all items at once
        weights = user_sims[top_n_idx]
        neighbor_ratings = self.rating_matrix.iloc[top_n_idx]
        neighbor_devs = neighbor_ratings.sub(self.user_means.iloc[top_n_idx], axis=0).fillna(0).values
        
        weighted_sum = weights @ neighbor_devs
        abs_sim_sum = np.abs(weights) @ neighbor_ratings.notna().astype(float).values
        abs_sim_sum[abs_sim_sum == 0] = 1e-9
        
        all_preds = self.user_means[user_a] + (weighted_sum / abs_sim_sum)
        preds_ser = pd.Series(all_preds, index=self.rating_matrix.columns)

        if exclude_rated:
            preds_ser = preds_ser.drop(self.rating_matrix.loc[user_a].dropna().index, errors='ignore')

        top_recs = preds_ser.sort_values(ascending=False).head(n)
        results = []
        for item_idx, score in top_recs.items():
            expl = f"Recommended because similar users with patterns like yours also purchased and enjoyed this item (Predicted rating: {score:.2f}/5)."
            results.append((int(item_idx), float(score), expl))
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "UserBasedCF":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Item-Based Nearest Neighbor (Adjusted Cosine Similarity)
# ══════════════════════════════════════════════════════════════════════════════

class ItemBasedCF:
    """
    Item-Based Nearest Neighbor using Adjusted Cosine Similarity (Lecture 4).

    Adjusted cosine accounts for individual user rating scales by subtracting
    each user's mean before computing cosine:
      sim(i,j) = Σ_u(r_{u,i}-r̄_u)(r_{u,j}-r̄_u)
                 ──────────────────────────────────────────────────────────
                 √Σ_u(r_{u,i}-r̄_u)² · √Σ_u(r_{u,j}-r̄_u)²
    where the sum is over users who rated BOTH items i and j.
    """

    def __init__(self, n_neighbors: int = 30):
        self.n_neighbors = n_neighbors
        self.rating_matrix: pd.DataFrame = None   # users × items
        self.item_sim: pd.DataFrame      = None   # items × items similarity matrix
        self.user_means: pd.Series       = None

    def fit(self, rating_matrix: pd.DataFrame):
        # نحتاج لنسخة واحدة فقط ونعمل عليها مباشرة (In-place) لتوفير الذاكرة
        self.rating_matrix = rating_matrix.astype(float)
        self.user_means    = self.rating_matrix.mean(axis=1)
        user_means_vals    = self.user_means.values
        
        n_items = self.rating_matrix.shape[1]
        item_norms = np.zeros(n_items)
        
        # 1. طرح المتوسطات في مكانها (In-place) لتجنب نسخ المصفوفة
        val = self.rating_matrix.values
        for i in range(len(val)):
            row = val[i]
            mask = ~np.isnan(row)
            row[mask] -= user_means_vals[i]
        
        # 2. حساب المعايير (norms) للأعمدة بعد الطرح
        for j in range(n_items):
            col = val[:, j]
            valid_mask = ~np.isnan(col)
            if np.any(valid_mask):
                item_norms[j] = np.sqrt(np.sum(col[valid_mask]**2))
        item_norms[item_norms == 0] = 1e-9

        # 3. تحويل NaNs إلى أصفار وتطبيع الأعمدة في مكانها
        np.nan_to_num(val, copy=False, nan=0.0)
        val /= item_norms # تقسيم كل عمود على المعيار الخاص به (Broadcasting)
        
        # 4. حساب مصفوفة التشابه (Similarity Matrix)
        # نستخدم np.dot(val.T, val) مباشرة لتجنب إنشاء نسخ وسيطة كبيرة
        self.item_sim = pd.DataFrame(np.dot(val.T, val), 
                                     index=self.rating_matrix.columns, 
                                     columns=self.rating_matrix.columns)
        return self

    def predict(self, user_u: int, item_i: int) -> float:
        if user_u not in self.rating_matrix.index or item_i not in self.item_sim.columns:
            return self.user_means.mean()

        user_ratings = self.rating_matrix.loc[user_u].dropna()
        sims = self.item_sim.loc[item_i, user_ratings.index]
        
        num = (sims * user_ratings).sum()
        denom = np.abs(sims).sum()
        return float(num / denom if denom != 0 else self.user_means[user_u])

    def recommend(self, user_u: int, n: int = 10, exclude_rated: bool = True) -> list:
        if user_u not in self.rating_matrix.index:
            return []

        user_ratings = self.rating_matrix.loc[user_u].dropna()
        sim_matrix = self.item_sim.loc[user_ratings.index]
        
        num = user_ratings.values @ sim_matrix.values
        denom = np.abs(sim_matrix.values).sum(axis=0)
        denom[denom == 0] = 1e-9
        
        preds_ser = pd.Series(num / denom, index=self.item_sim.columns)
        if exclude_rated:
            preds_ser = preds_ser.drop(user_ratings.index, errors='ignore')
            
        top_recs = preds_ser.sort_values(ascending=False).head(n)
        results = []
        for item_idx, score in top_recs.items():
            expl = f"Recommended because this item is similar to others you have rated highly in the past (Predicted rating: {score:.2f}/5)."
            results.append((int(item_idx), float(score), expl))
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ItemBasedCF":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Matrix Factorization via Gradient Descent
# ══════════════════════════════════════════════════════════════════════════════

class MatrixFactorization:
    """
    Matrix Factorization with Gradient Descent (Lecture 6).

    R ≈ P · Q^T   where P ∈ R^{n_users × k}, Q ∈ R^{n_items × k}

    Update rules per observed rating r_{ui}:
      ê_{ui}  = r_{ui} - P_u · Q_i
      P_uf   += α * (2 * ê_{ui} * Q_if - λ * P_uf)
      Q_if   += α * (2 * ê_{ui} * P_uf - λ * Q_if)

    Parameters (from Lecture 6):
      α = 0.01  (learning rate)
      λ = 0.1   (regularisation)
      k = 20    (latent features, increased for real data)
    """

    def __init__(
        self,
        n_factors: int = 20,
        lr: float = 0.01,
        reg: float = 0.1,
        n_epochs: int = 50,
        random_state: int = 42,
    ):
        self.n_factors    = n_factors
        self.lr           = lr
        self.reg          = reg
        self.n_epochs     = n_epochs
        self.random_state = random_state

        self.P: np.ndarray = None   # user factor matrix
        self.Q: np.ndarray = None   # item factor matrix
        self.n_users: int  = 0
        self.n_items: int  = 0
        self.global_mean: float = 0.0
        self.train_losses: list = []

    def fit(self, df: pd.DataFrame, n_users: int, n_items: int):
        """
        df must have columns: user_idx (int), item_idx (int), rating (float).
        """
        self.n_users     = n_users
        self.n_items     = n_items
        self.global_mean = df["rating"].mean()

        rng = np.random.default_rng(self.random_state)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        records = df[["user_idx", "item_idx", "rating"]].values

        for epoch in range(self.n_epochs):
            np.random.shuffle(records)
            total_loss = 0.0
            for u, i, r in records:
                u, i = int(u), int(i)
                pred    = self.P[u] @ self.Q[i]
                err     = r - pred                     # ê_{ui}
                total_loss += err ** 2

                P_u_old = self.P[u].copy()
                # Gradient descent update (Lecture 6)
                self.P[u] += self.lr * (2 * err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (2 * err * P_u_old  - self.reg * self.Q[i])

            rmse = np.sqrt(total_loss / len(records))
            self.train_losses.append(rmse)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.n_epochs}  RMSE={rmse:.4f}")

        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        if user_idx >= self.n_users or item_idx >= self.n_items:
            return self.global_mean
        return float(self.P[user_idx] @ self.Q[item_idx])

    def recommend(
        self, user_idx: int, n: int = 10,
        rated_items: set = None,
        cf_idx2item: dict = None,
    ) -> list:
        if user_idx >= self.n_users:
            return []

        scores = self.P[user_idx] @ self.Q.T   # shape (n_items,)
        ranked = np.argsort(scores)[::-1]

        results = []
        for i in ranked:
            if rated_items and i in rated_items:
                continue
            pred = float(scores[i])
            expl = (
                f"Recommended because similar users purchased this item; our latent factor analysis "
                f"suggests it fits your profile (Predicted score: {pred:.2f})."
            )
            results.append((i, pred, expl))
            if len(results) == n:
                break
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MatrixFactorization":
        with open(path, "rb") as f:
            return pickle.load(f)
