"""
NEXUS Games — Flask Backend Server
====================================
Serves the NEXUS Gaming UI and provides REST API endpoints
for the recommendation engine.

Usage:
    python server.py
    → http://localhost:5000
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── numpy._core compatibility patch ──────────────────────────────────────────
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
sys.modules['numpy._core.numeric'] = numpy.core.numeric

from flask import Flask, render_template, jsonify, request, send_from_directory

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

MODEL_DIR = ROOT / "models" / "saved"
DATA_DIR  = ROOT / "data"  / "processed"

app = Flask(__name__, static_folder="static", template_folder="templates")

# ═══════════════════════════════════════════════════════════════════════════════
# Lazy Model Loading (cached in module-level dict)
# ═══════════════════════════════════════════════════════════════════════════════

_cache = {}


def get_base_data():
    """Load base data (mappings, items, interactions, eval results). Fast."""
    if "base" in _cache:
        return _cache["base"]

    data = {}

    # Mappings
    p = DATA_DIR / "mappings.pkl"
    if p.exists():
        with open(p, "rb") as f:
            data["mappings"] = pickle.load(f)
    else:
        data["mappings"] = None

    # Items metadata
    p = DATA_DIR / "items_full.parquet"
    if p.exists():
        cols = ["parent_asin", "title", "price", "categories"]
        data["items_df"] = pd.read_parquet(p, columns=cols)
        data["items_df"]["price"] = pd.to_numeric(
            data["items_df"]["price"], errors="coerce"
        ).fillna(0.0).astype(np.float32)
    else:
        data["items_df"] = pd.DataFrame(columns=["parent_asin","title","price","categories"])

    # CF interactions
    p = DATA_DIR / "cf_interactions.parquet"
    if p.exists():
        cols = ["user_id", "user_idx", "parent_asin", "rating"]
        df = pd.read_parquet(p, columns=cols)
        df["rating"] = df["rating"].astype(np.float32)
        if "user_idx" in df.columns:
            df["user_idx"] = df["user_idx"].astype(np.int32)
            
        # Load persistent new interactions (from Heart button)
        new_p = DATA_DIR / "new_interactions.json"
        if new_p.exists():
            try:
                with open(new_p, "r") as f:
                    new_data = json.load(f)
                if new_data:
                    new_df = pd.DataFrame(new_data)
                    df = pd.concat([df, new_df], ignore_index=True)
            except Exception as e:
                print(f"Error loading new interactions: {e}")
                
        data["df_cf"] = df
    else:
        data["df_cf"] = pd.DataFrame()

    # Eval results
    p = MODEL_DIR / "eval_results.pkl"
    if p.exists():
        with open(p, "rb") as f:
            data["eval_results"] = pickle.load(f)
    else:
        data["eval_results"] = None

    _cache["base"] = data
    return data


def get_model(name):
    """Lazy load a specific model by name."""
    if name in _cache:
        return _cache[name]

    model = None
    base = get_base_data()
    mappings = base.get("mappings")
    items_df = base.get("items_df")

    try:
        if name == "ubcf":
            from collaborative_filtering import UserBasedCF
            p = MODEL_DIR / "ubcf.pkl"
            if p.exists():
                model = UserBasedCF.load(p)

        elif name == "ibcf":
            from collaborative_filtering import ItemBasedCF
            p = MODEL_DIR / "ibcf.pkl"
            if p.exists():
                model = ItemBasedCF.load(p)

        elif name == "mf":
            from collaborative_filtering import MatrixFactorization
            p = MODEL_DIR / "mf.pkl"
            if p.exists():
                model = MatrixFactorization.load(p)

        elif name == "ncf":
            from ncf_model import NCFRecommender
            p = MODEL_DIR / "ncf_model.pt"
            if p.exists() and mappings:
                model = NCFRecommender.from_checkpoint(
                    str(p), n_users=mappings["n_users"], n_items=mappings["n_items"]
                )

        elif name == "tfidf":
            from content_based import TFIDFRecommender
            p = MODEL_DIR / "tfidf.pkl"
            if p.exists():
                model = TFIDFRecommender.load(p)

        elif name == "cnn":
            from content_based import CNNRecommender
            p = MODEL_DIR / "cnn.pkl"
            if p.exists():
                model = CNNRecommender.load(p)

        elif name == "kb":
            from knowledge_based import KnowledgeBasedRecommender
            p = MODEL_DIR / "kb.pkl"
            if p.exists() and items_df is not None:
                model = KnowledgeBasedRecommender.load(p, items_df=items_df)

    except Exception as e:
        print(f"Error loading {name}: {e}")
        model = None

    _cache[name] = model
    return model


def get_item_info(asin, items_df):
    """Get item metadata by ASIN."""
    row = items_df[items_df["parent_asin"] == asin]
    if row.empty:
        return {"title": asin, "price": 0.0, "categories": "N/A"}
    r = row.iloc[0]
    return {
        "title": str(r.get("title", asin)),
        "price": float(r.get("price", 0.0) or 0.0),
        "categories": str(r.get("categories", "N/A")),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/users")
def api_users():
    base = get_base_data()
    mappings = base.get("mappings")
    if mappings and "user2idx" in mappings:
        users = sorted(mappings["user2idx"].keys())[:200]
    elif not base["df_cf"].empty and "user_id" in base["df_cf"].columns:
        users = sorted(base["df_cf"]["user_id"].unique().tolist())[:200]
    else:
        users = [f"U{i:04d}" for i in range(1, 51)]

    # Also include registered users from signup
    reg_file = ROOT / "registered_users.json"
    if reg_file.exists():
        try:
            with open(reg_file, "r") as f:
                reg_users = json.load(f)
            reg_names = [u["username"] for u in reg_users if u["username"] not in users]
            users = users + reg_names
        except Exception:
            pass

    return jsonify(users)


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or len(username) < 2:
        return jsonify({"ok": False, "error": "Username must be at least 2 characters."}), 400
    # Accept any password

    # Check if user already exists in the data
    base = get_base_data()
    mappings = base.get("mappings")
    existing = set()
    if mappings and "user2idx" in mappings:
        existing = set(mappings["user2idx"].keys())
    elif not base["df_cf"].empty and "user_id" in base["df_cf"].columns:
        existing = set(base["df_cf"]["user_id"].unique().tolist())

    if username in existing:
        return jsonify({"ok": False, "error": f"User '{username}' already exists. Please Login instead."}), 400

    # Save to registered_users.json
    reg_file = ROOT / "registered_users.json"
    reg_users = []
    if reg_file.exists():
        try:
            with open(reg_file, "r") as f:
                reg_users = json.load(f)
        except Exception:
            reg_users = []

    # Check if already registered
    if any(u["username"] == username for u in reg_users):
        return jsonify({"ok": False, "error": f"User '{username}' already registered. Please Login."}), 400

    reg_users.append({"username": username, "password": password})
    with open(reg_file, "w") as f:
        json.dump(reg_users, f, indent=2)

    return jsonify({"ok": True, "message": f"Account '{username}' created successfully!"})


@app.route("/api/user/<user_id>/history")
def api_user_history(user_id):
    base = get_base_data()
    df_cf = base["df_cf"]
    items_df = base["items_df"]

    if "user_id" in df_cf.columns:
        hist = df_cf[df_cf["user_id"] == user_id]
    else:
        hist = pd.DataFrame()

    results = []
    for _, row in hist.head(20).iterrows():
        asin = row.get("parent_asin", "")
        info = get_item_info(asin, items_df)
        results.append({
            "asin": asin,
            "title": info["title"],
            "rating": float(row.get("rating", 0)),
            "price": info["price"],
            "categories": info["categories"],
        })
    return jsonify(results)


@app.route("/api/stats")
def api_stats():
    base = get_base_data()
    mappings = base.get("mappings")
    df_cf = base["df_cf"]
    items_df = base["items_df"]

    return jsonify({
        "n_users": mappings["n_users"] if mappings else int(df_cf["user_idx"].nunique()) if not df_cf.empty else 0,
        "n_items": mappings["n_items"] if mappings else 0,
        "n_interactions": len(df_cf),
        "n_all_items": len(items_df),
    })


@app.route("/api/evaluation")
def api_evaluation():
    base = get_base_data()
    eval_df = base.get("eval_results")
    if eval_df is None:
        return jsonify(None)

    result = {
        "models": eval_df.index.tolist(),
        "MAE": eval_df["MAE"].tolist(),
        "RMSE": eval_df["RMSE"].tolist(),
        "Precision": eval_df["Precision"].tolist(),
        "Recall": eval_df["Recall"].tolist(),
        "F-Score": eval_df["F-Score"].tolist(),
    }
    return jsonify(result)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json
    approach = data.get("approach", "collaborative")
    method = data.get("method", "ubcf")
    user_id = data.get("user_id", "")
    n_recs = data.get("n_recs", 10)

    # Knowledge-based constraints
    max_price = data.get("max_price", 60.0)
    genre_kw = data.get("genre_keyword", "")
    min_rating = data.get("min_rating", 3.0)
    min_reviews = data.get("min_reviews", 5)

    base = get_base_data()
    mappings = base.get("mappings")
    items_df = base["items_df"]
    df_cf = base["df_cf"]

    # Resolve user index
    user_idx = None
    if mappings and "user2idx" in mappings:
        user_idx = mappings["user2idx"].get(user_id)

    # Get user ratings dict
    user_ratings_dict = {}
    if "user_id" in df_cf.columns:
        user_hist = df_cf[df_cf["user_id"] == user_id]
        if not user_hist.empty and "parent_asin" in user_hist.columns:
            user_ratings_dict = dict(zip(user_hist["parent_asin"], user_hist["rating"]))

    results = []

    try:
        if approach == "collaborative":
            if method == "ubcf" and user_idx is not None:
                m = get_model("ubcf")
                if m:
                    raw = m.recommend(user_idx, n=n_recs)
                    if mappings:
                        results = [(mappings["cf_idx2item"].get(i, str(i)), s, e) for i, s, e in raw]
                    else:
                        results = raw

            elif method == "ibcf" and user_idx is not None:
                m = get_model("ibcf")
                if m:
                    raw = m.recommend(user_idx, n=n_recs)
                    if mappings:
                        results = [(mappings["cf_idx2item"].get(i, str(i)), s, e) for i, s, e in raw]
                    else:
                        results = raw

            elif method == "mf" and user_idx is not None:
                m = get_model("mf")
                if m:
                    rated_idx = set()
                    if mappings and "cf_item2idx" in mappings:
                        rated_idx = {mappings["cf_item2idx"][a] for a in user_ratings_dict if a in mappings["cf_item2idx"]}
                    raw = m.recommend(user_idx, n=n_recs, rated_items=rated_idx)
                    if mappings:
                        results = [(mappings["cf_idx2item"].get(i, str(i)), s, e) for i, s, e in raw]
                    else:
                        results = raw

            elif method == "ncf" and user_idx is not None:
                m = get_model("ncf")
                if m:
                    rated_idx = set()
                    if mappings and "cf_item2idx" in mappings:
                        rated_idx = {mappings["cf_item2idx"][a] for a in user_ratings_dict if a in mappings["cf_item2idx"]}
                    raw = m.recommend(user_idx, n=n_recs, rated_items=rated_idx)
                    if mappings:
                        results = [(mappings["cf_idx2item"].get(i, str(i)), s, e) for i, s, e in raw]
                    else:
                        results = raw

        elif approach == "content":
            if method == "tfidf":
                m = get_model("tfidf")
                if m and user_ratings_dict:
                    results = m.recommend(user_ratings_dict, n=n_recs)
            elif method == "cnn":
                m = get_model("cnn")
                if m and user_ratings_dict:
                    results = m.recommend(user_ratings_dict, n=n_recs)

        elif approach == "knowledge":
            m = get_model("kb")
            if m:
                results = m.recommend(
                    max_price=max_price if max_price < 70 else None,
                    genre_keyword=genre_kw or None,
                    min_rating=min_rating,
                    min_reviews=min_reviews,
                    n=n_recs,
                    exclude_asins=set(user_ratings_dict.keys()),
                )

        elif approach == "hybrid":
            hybrid_scores = {}

            # MF component (40%)
            mf = get_model("mf")
            if mf and user_idx is not None:
                rated_idx = set()
                if mappings and "cf_item2idx" in mappings:
                    rated_idx = {mappings["cf_item2idx"][a] for a in user_ratings_dict if a in mappings["cf_item2idx"]}
                mf_recs = mf.recommend(user_idx, n=30, rated_items=rated_idx)
                for i, s, e in mf_recs:
                    asin = mappings["cf_idx2item"].get(i, str(i)) if mappings else str(i)
                    hybrid_scores[asin] = hybrid_scores.get(asin, 0) + 0.4 * s / 5

            # TF-IDF component (30%)
            tfidf = get_model("tfidf")
            if tfidf and user_ratings_dict:
                tfidf_recs = tfidf.recommend(user_ratings_dict, n=30)
                for asin, s, e in tfidf_recs:
                    hybrid_scores[asin] = hybrid_scores.get(asin, 0) + 0.3 * s

            # KB component (30%)
            kb = get_model("kb")
            if kb:
                kb_recs = kb.recommend(
                    max_price=max_price if max_price < 70 else None,
                    genre_keyword=genre_kw or None,
                    min_rating=min_rating, min_reviews=min_reviews, n=30,
                )
                for asin, s, e in kb_recs:
                    hybrid_scores[asin] = hybrid_scores.get(asin, 0) + 0.3 * s

            sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            exclude = set(user_ratings_dict.keys())
            for asin, score in sorted_hybrid:
                if asin in exclude:
                    continue
                results.append((
                    asin, score,
                    f"Hybrid recommendation combining CF (40%), Content-Based (30%), "
                    f"and Knowledge-Based (30%). Score: {score:.3f}."
                ))
                if len(results) == n_recs:
                    break

    except Exception as e:
        return jsonify({"error": str(e), "recommendations": []}), 500

    # Format results
    formatted = []
    for rank, (asin, score, expl) in enumerate(results, 1):
        info = get_item_info(asin, items_df)
        cats = [c.strip() for c in info["categories"].split(",") if c.strip()][:3]
        formatted.append({
            "rank": rank,
            "asin": asin,
            "title": info["title"],
            "price": info["price"],
            "score": round(float(score), 4),
            "categories": cats,
            "explanation": expl,
        })

    return jsonify({"recommendations": formatted})


@app.route("/api/add_interaction", methods=["POST"])
def add_interaction():
    req = request.json
    user_id = req.get("user_id")
    asin = req.get("parent_asin")
    
    if not user_id or not asin:
        return jsonify({"error": "Missing user_id or parent_asin"}), 400
        
    base = get_base_data()
    df_cf = base["df_cf"]
    mappings = base.get("mappings", {})
    
    # Calculate user_idx
    user_idx = -1
    if mappings and "user2idx" in mappings:
        user_idx = mappings["user2idx"].get(user_id, -1)
    
    if user_idx == -1:
        user_idx = int(df_cf["user_idx"].max() + 1) if not df_cf.empty else 1000000
    
    new_entry = {
        "user_id": user_id,
        "user_idx": user_idx,
        "parent_asin": asin,
        "rating": 5.0
    }
    
    # Update in-memory dataframe so recommendation counts update immediately
    base["df_cf"] = pd.concat([df_cf, pd.DataFrame([new_entry])], ignore_index=True)
    
    # Save to file persistently
    new_p = DATA_DIR / "new_interactions.json"
    interactions = []
    if new_p.exists():
        try:
            with open(new_p, "r") as f:
                interactions = json.load(f)
        except:
            interactions = []
    
    interactions.append(new_entry)
    with open(new_p, "w") as f:
        json.dump(interactions, f, indent=2)
        
    return jsonify({"success": True, "num_ratings": len(base["df_cf"][base["df_cf"]["user_id"] == user_id])})


@app.route("/api/store")
def api_store():
    base = get_base_data()
    items_df = base.get("items_df", pd.DataFrame())
    df_cf = base.get("df_cf", pd.DataFrame())
    
    sample_df = pd.DataFrame()
    trendy_asins = set()
    
    if not df_cf.empty and not items_df.empty:
        # Calculate popularity (interaction count)
        pop = df_cf.groupby("parent_asin").size().reset_index(name="c").sort_values("c", ascending=False)
        trendy_asins = set(pop.head(20)["parent_asin"]) # top 20 are definitely trendy
        
        # Merge to get top 200 most interacted items
        merged = pd.merge(pop.head(200), items_df, on="parent_asin", how="inner")
        sample_df = merged
        
    if sample_df.empty and not items_df.empty:
        sample_df = items_df.sample(n=min(200, len(items_df)), random_state=42)
    
    results = []
    icons = ['fa-gamepad', 'fa-dragon', 'fa-car', 'fa-city', 'fa-chess-rook', 'fa-rocket', 'fa-user-ninja', 'fa-tractor', 'fa-cubes', 'fa-ghost']
    import random
    
    for _, row in sample_df.iterrows():
        cats = [c.strip() for c in str(row.get("categories", "")).split(",") if c.strip()][:1]
        genre = cats[0] if cats else "Game"
        price_val = float(row.get("price", 0))
        if pd.isna(price_val):
            price_val = 0.0
            
        asin = row.get("parent_asin", "")
        
        # Give a trendy badge if it's in the top 20, or if fallback, random
        badge = ""
        if asin in trendy_asins:
            badge = "🔥 TRENDY"
        elif len(trendy_asins) == 0:
            badge = "Popular" if random.random() > 0.8 else ""
            
        results.append({
            "id": asin,
            "title": row.get("title", "Unknown Title"),
            "genre": genre,
            "price": price_val,
            "icon": random.choice(icons),
            "badge": badge
        })
        
    return jsonify(results)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print("\n[NEXUS Games] Intelligent Recommender System")
    print("  Loading base data...")
    get_base_data()
    print("  Ready!")
    print(f"  -> http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
