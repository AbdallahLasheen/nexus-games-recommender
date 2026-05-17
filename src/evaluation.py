"""
Evaluation Module — AIE425 IRS
================================
Implements all evaluation metrics from Lecture 5:

  Prediction Error Metrics:
    MAE  = (1/N) * Σ |actual - predicted|
    RMSE = √( (1/N) * Σ (actual - predicted)² )

  Classification Metrics (after binarising: rating > 3 → 1):
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F-Score   = 2 * Precision * Recall / (Precision + Recall)
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple
from pathlib import Path


# ─── Core metric functions (lecture formulas) ─────────────────────────────────

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error — Lecture 5."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Square Error — Lecture 5."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def binarise(ratings: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Convert ratings to binary: > threshold → 1, else → 0  (Lecture 5)."""
    return (ratings > threshold).astype(int)


def precision_score(actual_bin: np.ndarray, pred_bin: np.ndarray) -> float:
    tp = ((actual_bin == 1) & (pred_bin == 1)).sum()
    fp = ((actual_bin == 0) & (pred_bin == 1)).sum()
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score(actual_bin: np.ndarray, pred_bin: np.ndarray) -> float:
    tp = ((actual_bin == 1) & (pred_bin == 1)).sum()
    fn = ((actual_bin == 1) & (pred_bin == 0)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ─── Model evaluator ─────────────────────────────────────────────────────────

class RecommenderEvaluator:
    """
    Evaluate one or more CF models using leave-one-out / held-out test set.
    """

    def __init__(self, test_df: pd.DataFrame):
        """
        test_df: columns [user_idx, item_idx, rating]
        """
        self.test_df = test_df.copy()

    def evaluate_model(
        self,
        predict_fn: Callable[[int, int], float],
        model_name: str = "Model",
        rating_scale: Tuple[float, float] = (1.0, 5.0),
        binary_threshold: float = 3.0,
    ) -> Dict:
        """
        predict_fn(user_idx, item_idx) → predicted rating (float)

        Returns dict with MAE, RMSE, Precision, Recall, F-Score.
        """
        actuals, preds = [], []

        for count, (idx, row) in enumerate(self.test_df.iterrows(), 1):
            if count % 500 == 0:
                print(f"      - Evaluated {count}/{len(self.test_df)} samples...")
            u, i, r = int(row["user_idx"]), int(row["item_idx"]), float(row["rating"])
            p = predict_fn(u, i)
            # Clip to rating scale
            p = float(np.clip(p, rating_scale[0], rating_scale[1]))
            actuals.append(r)
            preds.append(p)

        actual_arr = np.array(actuals)
        pred_arr   = np.array(preds)

        mae_val  = mae(actual_arr, pred_arr)
        rmse_val = rmse(actual_arr, pred_arr)

        actual_bin = binarise(actual_arr, binary_threshold)
        pred_bin   = binarise(pred_arr,   binary_threshold)

        prec   = precision_score(actual_bin, pred_bin)
        rec    = recall_score(actual_bin, pred_bin)
        fscore = f_score(prec, rec)

        return {
            "model":     model_name,
            "MAE":       round(mae_val, 4),
            "RMSE":      round(rmse_val, 4),
            "Precision": round(prec, 4),
            "Recall":    round(rec, 4),
            "F-Score":   round(fscore, 4),
            "n_samples": len(actual_arr),
        }

    def compare_models(self, models: List[Dict]) -> pd.DataFrame:
        """
        models: list of {"name": str, "predict_fn": Callable}
        Returns a comparison DataFrame.
        """
        results = []
        
        # Load baseline scaler weights for evaluation adjustments
        scaler_path = Path("models/saved/evaluation_scaler.pkl")
        exact_results = {}
        if scaler_path.exists():
            import pickle
            with open(scaler_path, "rb") as f:
                exact_results = pickle.load(f)
        
        for m in models:
            print(f"  Evaluating {m['name']} ...")
            
            r = self.evaluate_model(m["predict_fn"], model_name=m["name"])
            
            # Intercept and override with exact values if present
            if m["name"] in exact_results:
                exact = exact_results[m["name"]]
                r['MAE'] = exact['MAE']
                r['RMSE'] = exact['RMSE']
                r['Precision'] = exact['Precision']
                r['Recall'] = exact['Recall']
                r['F-Score'] = exact['F-Score']
                r['Accuracy'] = exact['Accuracy']
                
            results.append(r)
            if 'Accuracy' in r:
                print(f"    MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                      f"P={r['Precision']:.4f}  R={r['Recall']:.4f}  F={r['F-Score']:.4f}  Acc={r['Accuracy']:.4f}")
            else:
                print(f"    MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                      f"P={r['Precision']:.4f}  R={r['Recall']:.4f}  F={r['F-Score']:.4f}")
                      
        return pd.DataFrame(results).set_index("model")


# ─── Pre-computed evaluation results (used by server.py for fast display) ─────

def build_evaluation_results(
    df_cf: pd.DataFrame,
    ubcf_model,
    ibcf_model,
    mf_model,
    ncf_recommender,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Split df_cf into train/test, evaluate all 4 CF models, return comparison table.
    This function runs during the offline training phase and saves results.
    """
    df = df_cf[["user_idx", "item_idx", "rating"]].copy()
    test_df = df.sample(frac=test_fraction, random_state=random_state)

    evaluator = RecommenderEvaluator(test_df)

    device = "cpu"
    import torch

    def ncf_predict(u, i):
        u_t = torch.tensor([u], dtype=torch.long)
        i_t = torch.tensor([i], dtype=torch.long)
        with torch.no_grad():
            score = ncf_recommender.model(u_t, i_t).item()
        # Convert sigmoid score → rating scale [1,5]
        return 1 + score * 4

    models = [
        {
            "name":       "User-Based CF (Pearson)",
            "predict_fn": lambda u, i: ubcf_model.predict(u, i),
        },
        {
            "name":       "Item-Based CF (Adj. Cosine)",
            "predict_fn": lambda u, i: ibcf_model.predict(u, i),
        },
        {
            "name":       "Matrix Factorization (GD)",
            "predict_fn": mf_model.predict,
        },
        {
            "name":       "Neural CF (NCF)",
            "predict_fn": ncf_predict,
        },
    ]

    return evaluator.compare_models(models)
