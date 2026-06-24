<div align="center">
  <h1>🎮 NEXUS Games | Advanced AI Recommender System</h1>
  <p><strong>A State-of-the-Art Machine Learning Recommendation Engine with Interactive Glassmorphism UI</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/Abdallah-12/nexus-games-app)

  <h3>🚀 <a href="https://huggingface.co/spaces/Abdallah-12/nexus-games-app">Play Live Demo Here</a> 🚀</h3>
</div>

<hr/>

## 📖 Overview

**NEXUS Games** is a production-ready, highly personalized video game recommendation system. It was built to demonstrate how multiple machine learning paradigms—from traditional Matrix Factorization to Deep Neural Collaborative Filtering—can be orchestrated together in a unified, scalable web application.

The project features a **Tri-Split Cloud Architecture** utilizing Hugging Face Hub, where the Application, Machine Learning Models, and Datasets are stored in distinct repositories and dynamically linked at runtime. This allows handling massive models (>8 GB) without exceeding repository size limitations.

---

## ✨ Key Features

*   **Multi-Algorithm Ensemble:** Simultaneously runs Content-Based, Collaborative Filtering, Matrix Factorization, and PyTorch Deep Learning models.
*   **Tri-Split Deployment:** Dynamically downloads datasets and `.pt`/`.pkl` binaries via `huggingface_hub` into a Dockerized Flask environment.
*   **Interactive Gaming Dashboard:** A seamless Single Page Application (SPA) utilizing smooth micro-animations, real-time fetching, and a premium Dark-Mode Glassmorphism design.
*   **Analytics & Evaluation:** Built-in comparison scripts calculating metrics such as RMSE, MAE, Precision@k, Recall@k, and F1-Score.

---

## 🧠 Algorithmic Architecture

The system evaluates user interactions and outputs personalized game recommendations by employing the following distinct algorithms:

### 1. Content-Based Filtering (CBF)
*   **TF-IDF & Cosine Similarity:** Extracts features from game titles, genres, and metadata to find visually and contextually similar games.
*   **CNN Embeddings:** Leverages deep learning feature extraction on game attributes.

### 2. Collaborative Filtering (CF)
*   **User-Based (UBCF):** Finds similar users based on interaction history using Pearson Correlation.
*   **Item-Based (IBCF):** Identifies items frequently bought/played together using Adjusted Cosine Similarity.

### 3. Matrix Factorization
*   **SVD (Singular Value Decomposition):** Reduces the dimensionality of the user-item interaction matrix, predicting missing ratings by learning latent user preferences and game attributes.

### 4. Neural Collaborative Filtering (NCF)
*   **PyTorch Deep Learning Model:** Replaces the inner product of Matrix Factorization with a neural architecture (Multi-Layer Perceptron) to learn complex, non-linear interactions between users and games.

### 5. Knowledge-Based Engine
*   **Rule-Based Logic:** Designed to combat the "Cold-Start Problem" for new users by filtering the catalog based on explicit constraints (price limits, genre preferences, rating thresholds).

### 6. Hybrid Ensemble Model
*   A weighted scoring system that merges the outputs of the aforementioned models. It normalizes confidence scores and creates a definitive rank, delivering the most robust and personalized recommendation list possible.

---

## 📁 Repository Structure

```text
videogame_recommender/
├── server.py                 # Core Flask backend and API routing
├── requirements.txt          # Python dependencies for Docker environment
├── Dockerfile                # Instructions for Hugging Face Spaces deployment
├── split_hf_deployment.py    # Automation script for Tri-Split Deployment
├── README.md                 # Project documentation and HF Space configuration
├── src/                      # Machine Learning Training Modules
│   ├── train_all.py          # Unified pipeline execution
│   ├── ncf_model.py          # PyTorch NCF Architecture
│   ├── hybrid_model.py       # Ensemble Logic
│   └── evaluation.py         # RMSE/Precision Evaluation metrics
├── static/                   # Frontend Web Assets
│   ├── css/style.css         # Glassmorphism UI styling
│   ├── js/app.js             # Async API fetching and DOM handling
│   └── images/               # Web application graphics
└── templates/
    └── index.html            # Main User Interface
```
*(Note: `data/` and `models/` directories are intentionally ignored by Git. They are hosted on external Hugging Face Dataset/Model repositories).*

---

## 🚀 Local Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/yourusername/videogame_recommender.git
cd videogame_recommender
pip install -r requirements.txt
```

### 2. Download Models & Data
Since the trained models and datasets are hosted separately, the `server.py` script uses `huggingface_hub` to download them automatically upon starting.
Ensure you have the following Environment Variables set (either locally or in a `.env` file):
*   `HF_MODEL_REPO`: The Hugging Face repo ID for your models (e.g., `username/nexus-games-models`)
*   `HF_DATASET_REPO`: The Hugging Face repo ID for your datasets (e.g., `username/nexus-games-data`)

### 3. Start the Backend Server
```bash
python server.py
```
Navigate to `http://localhost:5000` in your browser.

---

## ☁️ Hugging Face Cloud Deployment

This project utilizes a **Tri-Split Hugging Face Architecture** to bypass the standard 50GB Space and 100MB Git file limits.

### How it Works:
1.  **Hugging Face Dataset Repo:** Hosts `.parquet` and `.csv` files containing the game catalog and interaction history.
2.  **Hugging Face Model Repo:** Hosts the massive `.pkl` and `.pt` serialized model binaries (totaling ~8GB).
3.  **Hugging Face Space (Docker):** Hosts the code in this repository. 
When the Space spins up via the `Dockerfile`, `server.py` executes `download_hf_artifacts()`, which fetches the binary files directly into the container's isolated local storage before starting the Flask server.

To deploy your own version:
1. Create a HF Space (Docker SDK).
2. Set the `HF_MODEL_REPO` and `HF_DATASET_REPO` in the Space's Secrets settings.
3. Upload this codebase using `upload_space.py`.

---
<div align="center">
  <i>Developed for Advanced Recommender Systems Analysis & Deployment</i>
</div>
