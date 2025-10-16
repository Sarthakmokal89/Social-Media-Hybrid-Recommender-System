# 📊 Social Media Hybrid Recommender System

### Overview
This project implements a **Hybrid Recommender System** designed for social media platforms. It combines **Collaborative Filtering (CF)** and **Content-Based Filtering (CBF)** to generate personalized post recommendations. The notebook demonstrates a complete end-to-end pipeline using synthetic data, suitable for experimentation and learning.

### Objectives
- Model social media user–content interactions using synthetic datasets.
- Apply **Matrix Factorization (SVD)** for Collaborative Filtering.
- Use **TF-IDF vectorization** and metadata for Content-Based Filtering.
- Blend both approaches using a tunable **hybrid weighting parameter (α)**.
- Visualize recommendation distributions and platform preferences.

### Dataset
A **synthetic dataset** is generated programmatically:
- 300 users, 800 posts.
- Metadata: `platform`, `topic`, `title`, and `caption`.
- Ratings sampled between 1–5 with topic bias to simulate user preference patterns.

### Methodology
1. **Collaborative Filtering (CF)**  
   Utilizes Truncated SVD to capture latent relationships between users and posts.
2. **Content-Based Filtering (CBF)**  
   Uses TF-IDF on text fields combined with one-hot encoded metadata.
3. **Hybrid Model**  
   Final recommendation score = `α * CF + (1 - α) * CBF`, where `α = 0.6` by default.

### Workflow
1. Data generation (`generate_synthetic_social_data`)
2. Rating matrix construction (`build_rating_matrix`)
3. CF model training (TruncatedSVD)
4. Content embedding creation (TF-IDF + OneHotEncoder)
5. Hybrid recommendation generation
6. Visualization of rating distributions and platform trends

### Key Results
- Recommender successfully identifies user-specific content themes.
- CF captures collaborative similarity; CBF adds textual interpretability.
- Visual analytics show the mix of platforms in top-N suggestions.

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib scipy
# Social-Media-Hybrid-Recommender-System
