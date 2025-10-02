# 🎬 Movie Recommendation System (MovieLens 100K)

This project implements a **Recommendation System** using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).
The system predicts movies a user is likely to enjoy based on **similarity modeling and matrix factorization techniques**.

---

## 📌 Features

* ✅ **User-Based Collaborative Filtering**

  * Builds a **user-item matrix** from ratings.
  * Computes **cosine similarity** between users.
  * Recommends top-rated **unseen movies** based on similar users’ preferences.

* ✅ **Evaluation with Precision@K**

  * Implements **Precision@5** to evaluate recommendation quality.
  * Provides performance visualization across multiple users.

---

## 🔥 Bonus Implementations

* **Item-Based Collaborative Filtering**

  * Computes similarity between movies instead of users.
  * Recommends based on how similar items are to the user’s past ratings.

* **Matrix Factorization (SVD)**

  * Applies **Truncated SVD** for latent factor extraction.
  * Generates recommendations using hidden patterns in the user-item matrix.

---

## 📊 Visualizations

* Distribution of movie ratings.
* Heatmaps of user-user and item-item similarity matrices.
* Precision@K comparison plots.
* Top-N recommendation examples.

---

## 🛠️ Tools & Libraries

* **Python**, **Pandas**, **NumPy**
* **Scikit-learn** (cosine similarity, TruncatedSVD)
* **Matplotlib / Seaborn** for visualization

---

## 🚀 Results

* **User-Based CF**: Personalized recommendations from similar users.
* **Item-Based CF**: Content-aware recommendations based on item similarity.
* **SVD**: Latent factor recommendations capturing deeper patterns.

Each approach shows trade-offs in **precision, coverage, and personalization**, with visual results provided for comparison.
