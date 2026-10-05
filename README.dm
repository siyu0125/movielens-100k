# MovieLens Recommender Study: Rating Prediction vs. Top-N Ranking

A benchmark of rating-prediction models and top-N recommenders on the MovieLens dataset. Matrix factorization is the best rating predictor (**RMSE 0.881**), yet a non-personalized popularity baseline decisively beats it at top-10 ranking (**Precision@10 0.113 vs. 0.039**). The project analyzes why the two objectives disagree.

## Dataset

- MovieLens small dataset from [GroupLens](https://grouplens.org/datasets/movielens/): 100,004 ratings from 671 users on 9,066 movies, on a 0.5–5.0 half-star scale (1995–2016)
- The user–item matrix is only 1.64% filled, and 66% of movies have five or fewer ratings
- The raw data is not included in this repository. Download it from GroupLens and place the CSV files in `data/`.

## Approach

1. **EDA** — rating distribution (global mean 3.54), long-tail activity, popularity vs. average rating, and genre analysis
2. **Rating predictors** — global mean, regularized bias model (μ + bᵤ + bᵢ), item–item kNN collaborative filtering (cosine similarity on bias-centered ratings, top-40 neighbors, support shrinkage), and biased Funk-SVD matrix factorization (50 latent factors, SGD)
3. **Top-N recommenders** — matrix factorization ranked by predicted rating vs. a non-personalized popularity baseline
4. **Evaluation** — 80/20 random hold-out (seed 42). RMSE and MAE for rating prediction; Precision@10, Recall@10, and NDCG@10 for ranking, with held-out ratings ≥ 4 as relevant and a rankable pool of items with ≥ 10 training ratings

## Results

**Rating prediction**

| Model | RMSE | MAE |
|---|---|---|
| Global mean | 1.058 | 0.849 |
| Bias (μ + bᵤ + bᵢ) | 0.897 | 0.693 |
| Item–item kNN | 0.906 | 0.684 |
| **Matrix factorization** | **0.881** | **0.676** |

**Top-10 recommendation**

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Matrix factorization | 0.039 | 0.033 | 0.042 |
| **Popularity** | **0.113** | **0.109** | **0.130** |

**Key findings**

- Matrix factorization wins on rating prediction, but the result reverses for ranking: popularity scores roughly three times higher.
- Three mechanisms explain the gap: (1) objective mismatch, since squared error rewards calibrated scores rather than correct ordering; (2) popularity bias in the evaluation, since held-out liked items skew toward popular titles; (3) long-tail noise, since an RMSE-trained model can push poorly estimated obscure items above broadly liked ones.
- To beat popularity on ranking, the model must optimize ranking, for example with Bayesian Personalized Ranking or weighted ALS on implicit feedback.

![Rating-prediction RMSE by model](reports/figures/rmse_comparison.png)
![Top-10 ranking: popularity vs. matrix factorization](reports/figures/ranking_comparison.png)

The full write-up is in [`reports/`](reports/).

## Repository Structure

```
├── src/       # Data pipeline, models, and evaluation code
└── reports/   # Written report and figures
```

## How to Run

```bash
pip install numpy pandas scipy matplotlib
python src/<main_script>.py
```

## Limitations

- Top-N lists are ranked by predicted rating rather than by a ranking-trained model; a BPR/ALS comparison is the natural next step.
- A single random split and a rating ≥ 4 relevance threshold influence the absolute numbers; leave-last-out evaluation and other thresholds would test robustness.
- Genre information is available but unused by the collaborative models; a hybrid approach could address cold start.