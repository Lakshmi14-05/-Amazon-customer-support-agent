"""
Refined clustering of customer problems across the 4,939 queries.
Uses custom stop words (removing 'amazon', 'help', 'hi', 'please', 'just')
to surface true functional customer problem themes.
"""

import os
import re
import json
import io
import sys
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
sys.path.append(os.path.abspath("src"))
from explore_problems import load_all_customer_queries

# Domain-specific noise terms to ignore so topical issues emerge clearly
AMAZON_NOISE_TERMS = [
    "amazon", "amazonhelp", "help", "hi", "hello", "please", "thanks", "thank",
    "just", "like", "know", "got", "want", "need", "hey", "did", "say", "said",
    "really", "guys", "im", "ive", "dont", "cant", "wont", "doesnt", "get", "let"
]

def refine_clustering(queries, n_clusters=8, random_state=42):
    corpus = [q["cleaned_text"] for q in queries]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        stop_words=AMAZON_NOISE_TERMS + list(TfidfVectorizer(stop_words="english").get_stop_words()),
        min_df=3,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(corpus)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    for q, label in zip(queries, cluster_labels):
        q["cluster_id"] = int(label)

    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    cluster_summaries = {}
    counts = Counter(cluster_labels)

    for i in range(n_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        member_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        member_vectors = X[member_indices]
        cluster_center = kmeans.cluster_centers_[i]
        sims = member_vectors.dot(cluster_center)
        top_rep_indices = np.argsort(sims)[::-1][:5]
        representative_samples = [queries[member_indices[idx]] for idx in top_rep_indices]

        cluster_summaries[f"cluster_{i}"] = {
            "cluster_id": i,
            "count": counts[i],
            "percentage": round((counts[i] / len(queries)) * 100, 2),
            "top_terms": top_words,
            "representative_samples": representative_samples
        }

    return cluster_summaries


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    usable_path = "data/processed/threads_usable.json"
    partially_path = "data/processed/threads_partially_usable.json"

    queries = load_all_customer_queries(usable_path, partially_path)
    summaries = refine_clustering(queries, n_clusters=8)

    with open("data/processed/customer_problem_clusters_refined.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\n=== REFINED FUNCTIONAL CLUSTERING (4,939 QUERIES) ===")
    sorted_clusters = sorted(summaries.values(), key=lambda x: x["count"], reverse=True)
    for c in sorted_clusters:
        print(f"\nCluster {c['cluster_id']} — Volume: {c['count']:,} queries ({c['percentage']}%)")
        print(f"Top Discriminating Terms: {', '.join(c['top_terms'][:8])}")
        print("Real Representative Samples:")
        for s in c['representative_samples'][:3]:
            raw_single = s['raw_text'].replace('\n', ' ')
            print(f"  • [Thread: {s['thread_id']} | Tweet: {s['tweet_id']}] {raw_single[:110]}...")
