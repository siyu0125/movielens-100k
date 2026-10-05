"""Phase 2–4 — Recommender models & evaluation for MovieLens.

Rating prediction (RMSE/MAE): GlobalMean, Bias baseline, Item-item kNN CF, Matrix Factorization (Funk SVD).
Top-N ranking (Precision/Recall/NDCG@K): Popularity vs Matrix Factorization.
Writes reports/phase4_model.md, reports/metrics.json, reports/figures/*.

Run from movielens-100k/: ../.venv/bin/python src/recommender.py
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RNG = np.random.RandomState(42)
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
K = 10  # top-N cutoff


# ---------------- data ----------------
def load_split(test_frac=0.2):
    r = pd.read_csv("data/ratings.csv")
    # contiguous indices from the FULL set (train fallback handles unseen gracefully)
    r["u"] = r.userId.astype("category").cat.codes
    r["i"] = r.movieId.astype("category").cat.codes
    mask = RNG.rand(len(r)) < (1 - test_frac)
    tr, te = r[mask].reset_index(drop=True), r[~mask].reset_index(drop=True)
    n_u, n_i = r.u.max() + 1, r.i.max() + 1
    return tr, te, n_u, n_i


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b): return float(np.mean(np.abs(a - b)))


# ---------------- rating-prediction models ----------------
def global_mean(tr, te):
    mu = tr.rating.mean()
    return np.full(len(te), mu)


def bias_model(tr, te, n_u, n_i, reg=10.0, n_iter=15):
    """r_ui = mu + b_u + b_i, alternating regularized least squares."""
    mu = tr.rating.mean()
    bu = np.zeros(n_u); bi = np.zeros(n_i)
    u, i, rt = tr.u.values, tr.i.values, tr.rating.values
    cnt_u = np.bincount(u, minlength=n_u); cnt_i = np.bincount(i, minlength=n_i)
    for _ in range(n_iter):
        bi = np.bincount(i, weights=(rt - mu - bu[u]), minlength=n_i) / (reg + cnt_i)
        bu = np.bincount(u, weights=(rt - mu - bi[i]), minlength=n_u) / (reg + cnt_u)
    pred = mu + bu[te.u.values] + bi[te.i.values]
    return np.clip(pred, 0.5, 5.0), (mu, bu, bi)


def item_knn(tr, te, n_u, n_i, k=40, min_support=5, shrink=50.0):
    """Item-item CF on mean-centered ratings, cosine similarity, top-k neighbours."""
    mu = tr.rating.mean()
    bu = tr.groupby("u").rating.mean() - mu
    bu = bu.reindex(range(n_u)).fillna(0).values
    centered = tr.rating.values - mu - bu[tr.u.values]
    M = csr_matrix((centered, (tr.i.values, tr.u.values)), shape=(n_i, n_u))  # items x users
    support = np.asarray((M != 0).sum(axis=1)).ravel()
    sim = cosine_similarity(M, dense_output=False).toarray()
    np.fill_diagonal(sim, 0.0)
    sim *= (support / (support + shrink))[None, :]  # shrink low-support items
    # user -> {item: centered rating}
    user_items = {uu: dict(zip(g.i.values, centered[g.index]))
                  for uu, g in tr.groupby("u")}
    preds = np.empty(len(te))
    for row, (uu, ii) in enumerate(zip(te.u.values, te.i.values)):
        seen = user_items.get(uu, {})
        if not seen or support[ii] < min_support:
            preds[row] = mu + bu[uu]; continue
        cand = np.array(list(seen.keys())); s = sim[ii, cand]
        top = np.argsort(-s)[:k]; s_top = s[top]
        if s_top.sum() <= 1e-8:
            preds[row] = mu + bu[uu]; continue
        r_top = np.array([seen[c] for c in cand[top]])
        preds[row] = mu + bu[uu] + np.dot(s_top, r_top) / np.abs(s_top).sum()
    return np.clip(preds, 0.5, 5.0)


def matrix_factorization(tr, te, n_u, n_i, k=50, lr=0.01, reg=0.05, epochs=25):
    """Funk SVD with biases, trained by SGD."""
    mu = tr.rating.mean()
    bu = np.zeros(n_u); bi = np.zeros(n_i)
    Pu = RNG.normal(0, 0.05, (n_u, k)); Qi = RNG.normal(0, 0.05, (n_i, k))
    u, i, rt = tr.u.values, tr.i.values, tr.rating.values
    idx = np.arange(len(tr))
    for ep in range(epochs):
        RNG.shuffle(idx)
        for j in idx:
            uu, ii = u[j], i[j]
            pred = mu + bu[uu] + bi[ii] + Pu[uu] @ Qi[ii]
            e = rt[j] - pred
            bu[uu] += lr * (e - reg * bu[uu])
            bi[ii] += lr * (e - reg * bi[ii])
            Pu[uu] += lr * (e * Qi[ii] - reg * Pu[uu])
            Qi[ii] += lr * (e * Pu[uu] - reg * Qi[ii])
    pred = mu + bu[te.u.values] + bi[te.i.values] + np.sum(Pu[te.u.values] * Qi[te.i.values], axis=1)
    return np.clip(pred, 0.5, 5.0), (mu, bu, bi, Pu, Qi)


# ---------------- top-N ranking eval ----------------
def ranking_eval(tr, te, n_u, n_i, mf_params, k=K, rel=4.0, min_support=10):
    """Precision/Recall/NDCG@k for MF vs Popularity. Relevant = held-out items rated >= rel.
    Rankable pool restricted to items with >= min_support train ratings (fair, standard: keeps
    RMSE-optimized MF from surfacing noisy 1-rating long-tail items)."""
    mu, bu, bi, Pu, Qi = mf_params
    seen = tr.groupby("u")["i"].apply(set).to_dict()
    pos = te[te.rating >= rel].groupby("u")["i"].apply(set).to_dict()
    support = np.bincount(tr.i.values, minlength=n_i).astype(float)
    pop = support.copy()                      # popularity score
    low_support = support < min_support       # excluded from ranking pool

    def metrics_for(score_fn):
        P_, R_, N_, m = 0.0, 0.0, 0.0, 0
        idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
        for uu, relset in pos.items():
            relset = {it for it in relset if not low_support[it]}
            if not relset:
                continue
            scores = score_fn(uu).copy()
            scores[low_support] = -np.inf            # restrict to rankable pool
            for it in seen.get(uu, ()):              # mask already-seen (train) items
                scores[it] = -np.inf
            topk = np.argpartition(-scores, k)[:k]
            topk = topk[np.argsort(-scores[topk])]
            hits = [1 if it in relset else 0 for it in topk]
            P_ += sum(hits) / k
            R_ += sum(hits) / len(relset)
            dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
            N_ += dcg / idcg
            m += 1
        return {"precision@%d" % k: P_/m, "recall@%d" % k: R_/m, "ndcg@%d" % k: N_/m}

    mf_score = lambda uu: mu + bu[uu] + bi + Qi @ Pu[uu]
    pop_score = lambda uu: pop
    return {"MatrixFactorization": metrics_for(mf_score),
            "Popularity": metrics_for(pop_score)}


def main():
    tr, te, n_u, n_i = load_split()
    print(f"train {len(tr):,} | test {len(te):,} | users {n_u} | movies {n_i}")

    results = {}
    results["GlobalMean"] = {"rmse": rmse(global_mean(tr, te), te.rating.values),
                             "mae": mae(global_mean(tr, te), te.rating.values)}
    p_bias, _ = bias_model(tr, te, n_u, n_i)
    results["Bias(mu+bu+bi)"] = {"rmse": rmse(p_bias, te.rating.values), "mae": mae(p_bias, te.rating.values)}
    p_knn = item_knn(tr, te, n_u, n_i)
    results["ItemKNN"] = {"rmse": rmse(p_knn, te.rating.values), "mae": mae(p_knn, te.rating.values)}
    p_mf, mf_params = matrix_factorization(tr, te, n_u, n_i)
    results["MatrixFactorization"] = {"rmse": rmse(p_mf, te.rating.values), "mae": mae(p_mf, te.rating.values)}

    print("\n=== Rating prediction (test) ===")
    for k, v in results.items():
        print(f"  {k:>22}  RMSE={v['rmse']:.4f}  MAE={v['mae']:.4f}")

    ranking = ranking_eval(tr, te, n_u, n_i, mf_params)
    print(f"\n=== Top-{K} ranking ===")
    for model, mvals in ranking.items():
        print(f"  {model:>22}  " + "  ".join(f"{kk}={vv:.4f}" for kk, vv in mvals.items()))

    # figure: RMSE comparison
    plt.figure(figsize=(6.5, 4))
    names = list(results.keys()); rmses = [results[n]["rmse"] for n in names]
    ax = sns.barplot(x=rmses, y=names, palette="viridis")
    for j, val in enumerate(rmses):
        ax.annotate(f"{val:.3f}", (val, j), va="center", ha="left", fontsize=9)
    plt.title(f"Rating-prediction RMSE (lower is better)"); plt.xlabel("test RMSE"); plt.xlim(0.8, 1.15)
    plt.tight_layout(); plt.savefig(FIG / "rmse_comparison.png", dpi=110, bbox_inches="tight"); plt.close()

    # figure: top-N comparison
    plt.figure(figsize=(6.5, 4))
    dfp = pd.DataFrame(ranking).T.reset_index().melt(id_vars="index")
    sns.barplot(data=dfp, x="value", y="variable", hue="index", palette=["#4C72B0", "#C44E52"])
    plt.title(f"Top-{K} ranking quality"); plt.xlabel("score"); plt.ylabel(""); plt.legend(title="")
    plt.tight_layout(); plt.savefig(FIG / "ranking_comparison.png", dpi=110, bbox_inches="tight"); plt.close()

    md = ["# Phase 4 — Recommender results\n",
          f"- 80/20 random hold-out; {len(tr):,} train / {len(te):,} test ratings.\n",
          "\n## Rating prediction (lower is better)\n",
          "| model | RMSE | MAE |", "|---|---|---|"]
    for k, v in results.items():
        md.append(f"| {k} | {v['rmse']:.4f} | {v['mae']:.4f} |")
    md.append("\n![rmse_comparison.png](figures/rmse_comparison.png)\n")
    md.append(f"\n## Top-{K} recommendation (relevant = held-out rating ≥ 4)\n")
    md.append("| model | Precision@%d | Recall@%d | NDCG@%d |" % (K, K, K))
    md.append("|---|---|---|---|")
    for model, mvals in ranking.items():
        md.append(f"| {model} | " + " | ".join(f"{vv:.4f}" for vv in mvals.values()) + " |")
    md.append("\n![ranking_comparison.png](figures/ranking_comparison.png)\n")
    best = min(results, key=lambda k: results[k]["rmse"])
    mf_p = ranking["MatrixFactorization"]["precision@%d" % K]
    pop_p = ranking["Popularity"]["precision@%d" % K]
    winner = "matrix factorization" if mf_p >= pop_p else "the popularity baseline"
    md.append(f"\n**Best rating predictor: `{best}` (RMSE {results[best]['rmse']:.4f}).** "
              f"On top-{K} ranking (rankable pool ≥10 ratings), {winner} leads "
              f"(MF P@{K}={mf_p:.3f} vs Popularity {pop_p:.3f}) — a reminder that RMSE-optimal "
              "models are not automatically ranking-optimal.\n")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/phase4_model.md").write_text("\n".join(md))
    json.dump({"rating_prediction": results, "ranking": ranking},
              open("reports/metrics.json", "w"), indent=2)
    print("\nWrote reports/phase4_model.md and reports/metrics.json")


if __name__ == "__main__":
    main()
