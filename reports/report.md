# MovieLens Recommendation: Study Report

Building and evaluating a movie recommender on the MovieLens (100K-scale) dataset
(Kaggle CSV distribution: `ratings.csv` + `movies.csv` + `tags.csv`; 100,004 ratings, 671 users,
9,066 movies). Two tasks: **rating prediction** (RMSE/MAE) and **top-N recommendation**
(Precision/Recall/NDCG@10).

---

## Executive summary
- **Best rating predictor: Matrix Factorization (Funk SVD) — test RMSE 0.881, MAE 0.676**, ahead of a
  bias baseline (0.897), item-item kNN (0.906), and the global-mean floor (1.058).
- **Instructive negative result for ranking:** on top-10 recommendation, a **popularity baseline
  (Precision@10 = 0.113) beats matrix factorization (0.039)**. An RMSE-optimal model is *not* a
  ranking-optimal model — the standard lesson that explicit-rating regression and top-N ranking are
  different objectives.
- **The data is extremely sparse (1.64% filled)** with a heavy long tail: 66% of movies have ≤5 ratings,
  which drives cold-start difficulty and motivates bias shrinkage and minimum-support filtering.

Pipeline: `src/eda.py` → `src/recommender.py`. A formal write-up is in [paper.md](paper.md);
companion outputs: [phase1_eda.md](phase1_eda.md), [phase4_model.md](phase4_model.md),
[metrics.json](metrics.json).

## Dataset note
The classic MovieLens 100K (`u.data`/`u.item`/`u.user`, 943 users × 1682 movies, with demographics) was not
retrievable from the accessible Kaggle mirrors; those provide the internally-consistent CSV distribution
(shared `movieId`, genres in `movies.csv`). We use that: 100,004 ratings on a 0.5–5.0 half-star scale — the
same "100K" scale, with genres for content analysis but without user demographics.

## Phase 1 — Data Analysis
- **Scale & sparsity:** 100,004 ratings, 671 users, 9,066 movies; user–item matrix **1.64%** filled.
- **Rating distribution:** positively skewed, global mean **3.54**, mode 4.0 → subtract global/user/item
  biases before modeling.
- **Long tail:** ratings/user median 71 (max 2,391); ratings/movie median 3 (max 341); **66% of movies have
  ≤5 ratings**. Sparsely-rated movies take extreme average ratings (a single 5.0 rater), motivating
  shrinkage and a minimum-support filter for neighborhood methods.
- **Genres:** 20 genres; Drama/Comedy/Action most rated; Film-Noir/War/Documentary highest average,
  Horror/Comedy lowest. Time span 1995–2016.

## Phase 2–3 — Representation
- Map users/movies to contiguous indices; build the sparse user–item rating matrix.
- Center by global mean and user/item biases; apply support-based shrinkage for the neighborhood model.
- Genres retained for content analysis and a cold-start fallback.

## Phase 4 — Models & Evaluation (`recommender.py`)
80/20 random hold-out (80,099 train / 19,905 test ratings).

**Rating prediction (lower is better):**

| model | RMSE | MAE |
|---|---|---|
| Global mean | 1.058 | 0.849 |
| Bias (μ + bᵤ + bᵢ) | 0.897 | 0.693 |
| Item–item kNN CF | 0.906 | 0.684 |
| **Matrix Factorization** | **0.881** | **0.676** |

**Top-10 ranking (relevant = held-out rating ≥ 4; rankable pool ≥ 10 ratings):**

| model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Matrix Factorization | 0.039 | 0.033 | 0.042 |
| **Popularity** | **0.113** | **0.109** | **0.130** |

- **Matrix factorization wins rating prediction** — latent factors capture user/movie interaction beyond
  additive biases, and the bias baseline already closes most of the gap from the global-mean floor.
- **Popularity wins ranking.** Held-out "liked" items skew popular, and an RMSE-trained model surfaces
  personalized-but-untested items rather than broadly-liked ones. This is expected: rating regression
  optimizes squared error, not ranking.

## Limitations & next steps
1. **Ranking needs a ranking objective** — implicit-feedback methods (BPR, WARP, ALS) optimize ordering
   directly and are the right tool to beat the popularity baseline on top-N.
2. **Cold start** — 66% of movies have ≤5 ratings; a hybrid that blends genre-content similarity would help
   the long tail that CF/MF cannot serve.
3. **No demographics** — the CSV distribution lacks user age/gender/occupation (present in classic 100K);
   demographic features could enable a stronger cold-start user model.
4. **Single random split** — a temporal or per-user leave-last-out split would better mimic deployment.

## Reproducibility
```bash
cd movielens-100k
../.venv/bin/python src/eda.py           # Phase 1 -> figures + phase1_eda.md
../.venv/bin/python src/recommender.py   # Phase 4 -> metrics.json + RMSE/ranking figures
../.venv/bin/python src/build_paper_html.py   # paper.md -> paper.html
```
Seed 42; 80/20 hold-out.
