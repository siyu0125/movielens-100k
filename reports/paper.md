# Matrix Factorization, Neighborhoods, and the Popularity Baseline: A Study of Rating Prediction versus Top-N Ranking on MovieLens

**Abstract**

Recommender systems are evaluated in two distinct ways — how accurately they predict a user's rating of an
item, and how well they rank items the user will actually like — and these objectives do not always agree.
Using the MovieLens 100K-scale dataset (100,004 ratings, 671 users, 9,066 movies), we build a recommendation
pipeline and benchmark four rating predictors (global mean, a regularized bias model, item–item k-nearest-
neighbor collaborative filtering, and Funk-SVD matrix factorization) and two top-N recommenders (matrix
factorization ranked by predicted rating versus a non-personalized popularity baseline). Matrix
factorization gives the best rating prediction (test RMSE 0.881, MAE 0.676), but on top-10 recommendation
the popularity baseline decisively outranks it (Precision@10 0.113 vs. 0.039). We analyze why: the dataset
is extremely sparse (1.64% filled) with a long tail in which 66% of movies have five or fewer ratings, and
an RMSE-trained model surfaces personalized-but-unvalidated long-tail items rather than the broadly liked,
popular items that dominate held-out positives. We conclude that optimizing squared error is the wrong
objective for ranking and that implicit-feedback, ranking-aware methods are required to beat popularity.

**Keywords:** recommender systems, collaborative filtering, matrix factorization, top-N ranking,
popularity baseline, MovieLens, RMSE, NDCG.

---

## 1. Introduction

Recommendation is central to modern content and commerce platforms. Two evaluation paradigms coexist. In
**rating prediction**, the system estimates the numeric score a user would give an item and is scored by
error metrics such as RMSE and MAE — the framing popularized by the Netflix Prize. In **top-N
recommendation**, the system produces a short ranked list and is scored by ranking metrics such as
Precision@K, Recall@K, and NDCG@K — closer to how recommenders are actually used. A recurring, sometimes
surprising, empirical fact is that a model excelling at the first can be mediocre at the second, and that a
trivial popularity baseline is hard to beat on the second.

Using the MovieLens 100K-scale dataset, we build an end-to-end pipeline and quantify this gap. Our
contributions are: (1) a reproducible analysis of the dataset's sparsity and long-tail structure; (2) a
benchmark of four rating predictors under a common hold-out; (3) a top-N comparison of matrix factorization
against a popularity baseline; and (4) an analysis of why the RMSE winner loses at ranking, with concrete
recommendations for closing the gap.

## 2. Related work

**Neighborhood methods.** Item–item collaborative filtering predicts a rating from a user's ratings of
similar items, with similarity from cosine or Pearson correlation; shrinkage and minimum-support thresholds
stabilize similarities on sparse data [Sarwar et al. 2001].

**Matrix factorization.** Latent-factor models represent users and items in a shared low-dimensional space;
the biased "Funk SVD" trained by stochastic gradient descent was central to the Netflix Prize and remains a
strong rating predictor [Koren et al. 2009].

**Ranking objectives.** Because rating regression is not ranking, implicit-feedback methods optimize
ordering directly — Bayesian Personalized Ranking [Rendle et al. 2009] and weighted matrix factorization
for implicit data [Hu et al. 2008]. A well-known methodological point is that non-personalized popularity
is a strong top-N baseline and that offline top-N evaluation is sensitive to protocol and popularity bias
[Cremonesi et al. 2010].

## 3. Data and exploratory analysis

The dataset comprises 100,004 ratings by 671 users on 9,066 movies, on a 0.5–5.0 half-star scale, spanning
1995–2016, with movie genres available. The user–item matrix is **1.64%** filled — extreme sparsity that
governs every modeling choice.

**Rating distribution.** Ratings skew positive with a global mean of **3.54** and a mode of 4.0
(Figure 1). This immediately implies that subtracting global, user, and item biases is a prerequisite for
any accurate model, and it sets the error floor: a global-mean predictor already achieves RMSE ≈ 1.06.

![Figure 1. Rating distribution (mean 3.54). Ratings skew positive; 4.0 is the mode.](figures/rating_distribution.png)

**Long tail.** Activity is highly uneven (Figure 2). The median user has rated 71 movies (max 2,391), while
the median movie has just 3 ratings (max 341), and **66% of movies have five or fewer ratings**. Such
sparsely-rated items take extreme average ratings — a movie with a single 5-star rating averages 5.0 —
which is visible as the funnel in Figure 3 and motivates both bias shrinkage and a minimum-support filter
for neighborhood and ranking methods.

![Figure 2. Ratings per user and per movie (log-scaled counts). A few heavy raters and a few popular movies dominate; most movies are rated only a handful of times.](figures/longtail_activity.png)

![Figure 3. Popularity versus average rating. Rarely-rated movies span the full rating range; popular movies regress toward 3.5–4.0.](figures/popularity_vs_rating.png)

**Genres and time.** Twenty genres are present; Drama, Comedy, and Action are the most-rated, while
Film-Noir, War, and Documentary earn the highest average ratings and Horror the lowest (Figure 4).

![Figure 4. Rating volume (left) and average rating (right) by genre; dashed line is the global mean.](figures/genre_analysis.png)

## 4. Methodology

**Representation.** Users and movies are mapped to contiguous indices and split 80/20 at random into train
and test ratings; unseen users/items at test time fall back to available biases.

**Rating predictors.** (i) *Global mean* — predict μ for all. (ii) *Bias model* — r̂ = μ + bᵤ + bᵢ with
regularized, alternating closed-form updates. (iii) *Item–item kNN* — cosine similarity on
bias-centered ratings, top-40 neighbors, support-based shrinkage, and a minimum-support floor. (iv) *Matrix
factorization* — biased Funk SVD, r̂ = μ + bᵤ + bᵢ + pᵤ·qᵢ, 50 latent factors trained by SGD (25 epochs,
learning rate 0.01, L2 0.05); predictions clipped to [0.5, 5].

**Top-N evaluation.** For each user we treat held-out items rated ≥ 4 as relevant, rank the rankable pool
(items with ≥ 10 training ratings, excluding already-seen items), take the top 10, and compute Precision@10,
Recall@10, and NDCG@10. We compare matrix factorization (ranked by predicted rating) against a
non-personalized popularity ranking (by training rating count). Metrics are RMSE/MAE for rating prediction.

## 5. Results

**Rating prediction (Table 1, Figure 5).** Matrix factorization is best (RMSE 0.881), with the bias model
already capturing most of the improvement over the global-mean floor; item–item kNN is competitive on MAE
but slightly worse on RMSE.

*Table 1. Rating-prediction performance on the 20% hold-out.*

| model | RMSE | MAE |
|---|---|---|
| Global mean | 1.058 | 0.849 |
| Bias (μ + bᵤ + bᵢ) | 0.897 | 0.693 |
| Item–item kNN | 0.906 | 0.684 |
| **Matrix factorization** | **0.881** | **0.676** |

![Figure 5. Rating-prediction RMSE by model (lower is better).](figures/rmse_comparison.png)

**Top-N ranking (Table 2, Figure 6).** The result reverses. The popularity baseline achieves Precision@10
0.113 and NDCG@10 0.130, roughly three times matrix factorization's 0.039 and 0.042.

*Table 2. Top-10 recommendation (relevant = held-out rating ≥ 4; rankable pool ≥ 10 ratings).*

| model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Matrix factorization | 0.039 | 0.033 | 0.042 |
| **Popularity** | **0.113** | **0.109** | **0.130** |

![Figure 6. Top-10 ranking quality: popularity versus matrix factorization.](figures/ranking_comparison.png)

## 6. Discussion

The divergence between Tables 1 and 2 is the paper's central finding, and it is not a bug — it is the
expected consequence of optimizing the wrong objective for the task. Three mechanisms combine. First,
**objective mismatch**: matrix factorization minimizes squared rating error, which rewards calibrated
scores everywhere, not correct ordering at the top of the list. Second, **popularity bias in the
evaluation**: the held-out items a user rated ≥ 4 are themselves skewed toward popular titles, so simply
recommending popular items intercepts many of them. Third, **long-tail noise**: with 66% of movies rated
five or fewer times, an RMSE-trained model can assign confident-looking high scores to obscure items with
poorly-estimated factors, pushing them above broadly-liked titles; the minimum-support filter mitigates but
does not remove this. Together these make popularity a strong, hard-to-beat baseline for offline top-N on
sparse explicit data — a well-documented phenomenon.

The practical implication is direct: to beat popularity on ranking, one must optimize ranking. Implicit-
feedback models such as Bayesian Personalized Ranking or weighted ALS learn to order items rather than
regress ratings, and typically surpass both popularity and rating-optimized matrix factorization on top-N
metrics.

## 7. Limitations

- **Ranking model.** We rank by predicted rating rather than with a ranking-trained model; a BPR/ALS
  comparison is the natural next step.
- **Evaluation protocol.** A single random split and a rating-≥4 relevance definition influence absolute
  numbers; per-user leave-last-out and alternative relevance thresholds would test robustness.
- **Cold start and side information.** This CSV distribution lacks the user demographics of the classic
  MovieLens 100K; genre content is available but unused by the collaborative models, leaving a hybrid
  cold-start approach for future work.

## 8. Conclusion

On MovieLens, matrix factorization is the best rating predictor (RMSE 0.881), but a non-personalized
popularity baseline decisively wins top-N ranking (Precision@10 0.113 vs. 0.039). Rating accuracy and
ranking quality are different objectives, and choosing an evaluation that matches the deployment goal — and
a model trained for that objective — matters more than squeezing RMSE. The pipeline and experiments are
fully reproducible from the accompanying code.

## References

- Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender algorithms on top-N
  recommendation tasks. *RecSys*.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
  *IEEE Computer*, 42(8), 30–37.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized
  ranking from implicit feedback. *UAI*.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering
  recommendation algorithms. *WWW*.
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM TiiS*, 5(4).

*Reproducibility: see `src/` and `reports/`. Seed 42; 80/20 hold-out.*
