<div align="center">
  <img src="static/images/logo.png" alt="NEXUS Games Logo" width="120"/>
  <h1>NEXUS Games | AI Recommender System</h1>
  <p><strong>Advanced Machine Learning Recommender System with an Interactive Glassmorphism UI</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
</div>

<hr/>

## 📖 Overview

**NEXUS Games** is a state-of-the-art recommendation engine designed to provide highly personalized video game recommendations. Built to compare and showcase various algorithmic approaches, this platform features an interactive frontend utilizing a dynamic, dark-mode glassmorphism design.

Whether you're a new user looking for quick genre-based suggestions, or a veteran with a massive library needing deep collaborative filtering, NEXUS delivers.

## ✨ Key Features

*   **Interactive Gaming Dashboard:** A seamless single-page application (SPA) with smooth micro-animations, functional shopping cart, premium upgrade modals, and date/platform filtering.
*   **Real-time Recommendations:** Fetch personalized game suggestions via robust RESTful APIs.
*   **Analytics & Evaluation:** Built-in comparison tools utilizing metrics like **RMSE, MAE, Precision@k, Recall@k, and F1-Score**.
*   **Responsive Design:** Fully fluid UI that adapts flawlessly to desktop and mobile devices.

## 🧠 Algorithmic Approaches

The system implements multiple recommendation paradigms, each optimized for different scenarios:

1.  **Content-Based Filtering (CBF):**
    *   **TF-IDF & Cosine Similarity:** Analyzes game metadata (title, genre, tags) to find similar games.
    *   **CNN-Based Embeddings:** Utilizes Convolutional Neural Networks for advanced feature extraction.
2.  **Collaborative Filtering (CF):**
    *   **User-Based (UBCF):** Finds similar users based on interaction history (Pearson Correlation).
    *   **Item-Based (IBCF):** Identifies games frequently bought together (Cosine/Adjusted Cosine).
3.  **Matrix Factorization:**
    *   **SVD (Singular Value Decomposition):** Decomposes the user-item interaction matrix to uncover latent features.
4.  **Neural Collaborative Filtering (NCF):**
    *   **Deep Learning Model (PyTorch):** Learns non-linear interactions between users and items using embedding layers and Multi-Layer Perceptrons (MLP).
5.  **Knowledge-Based Filtering:**
    *   **Rule-Based Engine:** Best for "cold-start" users; filters by price, genre, ratings, and review counts based on explicit user input.
6.  **Hybrid Approach:**
    *   **Ensemble Engine:** Combines Matrix Factorization, TF-IDF, and Knowledge-Based outputs using a weighted scoring system for the ultimate personalized experience.

## 📁 Project Architecture

```text
videogame_recommender/
├── server.py                 # Main Flask application and API routes
├── src/                      # Core Machine Learning modules
│   ├── train_all.py          # Unified training pipeline script
│   ├── collaborative_filtering.py
│   ├── content_based.py
│   ├── ncf_model.py
│   ├── matrix_factorization.py
│   ├── knowledge_based.py
│   ├── hybrid_model.py
│   └── evaluation.py         # Metrics calculation (RMSE, Precision, etc.)
├── static/                   # Frontend assets
│   ├── css/style.css         # UI Styling (Glassmorphism, animations)
│   ├── js/app.js             # Frontend logic (API fetching, DOM manipulation)
│   └── images/               # Game covers and UI graphics
├── templates/
│   └── index.html            # Main SPA layout
├── data/                     # Datasets (Raw & Processed) - *Ignored in Git*
└── models/saved/             # Serialized model weights (.pkl, .pt) - *Ignored in Git*
```

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/videogame_recommender.git
cd videogame_recommender
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 3. Generate Models & Datasets (Crucial Step)
> ⚠️ **Note:** Due to GitHub's strict file size limitations (100MB per file), the pre-trained models (which exceed 8GB total) and raw datasets are **not** included in this repository.

To generate the required `.pkl`, `.parquet`, and `.pt` files, you must run the unified training script once:
```bash
python src/train_all.py
```
*(Depending on your hardware, model training and serialization may take several minutes).*

### 4. Run the Application
Start the Flask server:
```bash
python server.py
```
Open your web browser and navigate to: `http://localhost:5000`

## ☁️ Deployment (Hugging Face Spaces)

This project is fully containerized and ready for deployment on **Hugging Face Spaces** using Docker. 
A `Dockerfile` is included in the root directory to automatically build the environment, install dependencies, and run the Flask app on port `7860`.

---
<div align="center">
  <i>Developed for Advanced Recommender Systems Analysis</i>
</div>
