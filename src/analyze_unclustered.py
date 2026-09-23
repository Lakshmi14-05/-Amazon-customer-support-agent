"""
Phase 4 Extension: Deep Dive into Unclustered / Residual Queries (1,544 queries).
1. Samples 25 real unclustered queries with thread IDs to inspect semantic themes.
2. Evaluates KMeans across k in [8, 10, 12, 14] to see if coherent clusters emerge.
3. Quantifies reduction in cluster dispersion and residual variance.
"""

import os
import re
import json
import io
import sys
import random
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath("src"))
from explore_problems import load_all_customer_queries
from refine_clusters import AMAZON_NOISE_TERMS


def analyze_residual_and_evaluate_k(queries):
    corpus = [q["cleaned_text"] for q in queries]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        stop_words=AMAZON_NOISE_TERMS + list(TfidfVectorizer(stop_words="english").get_stop_words()),
        min_df=3,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(corpus)

    # 1. Run k=10 clustering to see how the large/diffuse cluster splits
    kmeans_10 = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_10 = kmeans_10.fit_predict(X)

    terms = vectorizer.get_feature_names_out()
    centroids = kmeans_10.cluster_centers_.argsort()[:, ::-1]

    clusters_10 = {}
    counts_10 = Counter(labels_10)

    for i in range(10):
        top_words = [terms[ind] for ind in centroids[i, :10]]
        member_indices = [idx for idx, label in enumerate(labels_10) if label == i]
        member_vectors = X[member_indices]
        sims = member_vectors.dot(kmeans_10.cluster_centers_[i])
        top_rep_indices = np.argsort(sims)[::-1][:5]
        rep_samples = [queries[member_indices[idx]] for idx in top_rep_indices]

        clusters_10[i] = {
            "cluster_id": i,
            "count": counts_10[i],
            "pct": round((counts_10[i] / len(queries)) * 100, 2),
            "top_terms": top_words,
            "rep_samples": rep_samples,
            "member_queries": [queries[idx] for idx in member_indices]
        }

    # 2. Evaluate Inertia across k in [8, 10, 12, 14]
    inertias = {}
    for k_val in [8, 10, 12, 14]:
        km = KMeans(n_clusters=k_val, random_state=42, n_init=5)
        km.fit(X)
        inertias[k_val] = round(km.inertia_, 2)

    return clusters_10, inertias


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    usable_path = "data/processed/threads_usable.json"
    partially_path = "data/processed/threads_partially_usable.json"

    queries = load_all_customer_queries(usable_path, partially_path)
    clusters_10, inertias = analyze_residual_and_evaluate_k(queries)

    print("=== INERTIA EVALUATION ACROSS K ===")
    for k_val, inr in inertias.items():
        print(f"• k={k_val:2d} | Inertia: {inr:,.2f}")

    print("\n=== K=10 CLUSTER SUMMARIES ===")
    sorted_clusters = sorted(clusters_10.values(), key=lambda x: x["count"], reverse=True)
    for c in sorted_clusters:
        print(f"\nCluster {c['cluster_id']} — Volume: {c['count']:,} queries ({c['pct']}%)")
        print(f"Top Terms: {', '.join(c['top_terms'][:8])}")
        print("Samples:")
        for s in c['rep_samples'][:2]:
            raw_s = s['raw_text'].replace('\n', ' ')
            print(f"  • [ID: {s['thread_id']}] {raw_s[:100]}...")

    # Identify the truly diffuse long-tail cluster
    diffuse_cluster = sorted_clusters[0]  # The largest residual cluster
    random.seed(42)
    sample_diffuse = random.sample(diffuse_cluster["member_queries"], 25)

    with open("data/processed/residual_25_samples.json", "w", encoding="utf-8") as f:
        json.dump([{
            "thread_id": q["thread_id"],
            "tweet_id": q["tweet_id"],
            "text": q["raw_text"]
        } for q in sample_diffuse], f, indent=2, ensure_ascii=False)

    print(f"\nSaved 25 random residual queries to data/processed/residual_25_samples.json")
