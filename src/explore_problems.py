"""
Phase 4: Unsupervised Exploration & Clustering of Customer Problems.
Processes customer-initiated messages across BOTH `usable` and `partially_usable`
threads (4,980 real customer queries) using TF-IDF and KMeans clustering.
"""

import os
import re
import json
import io
import sys
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


def clean_customer_query(text: str) -> str:
    """Cleans tweet handles, URLs, and extraneous whitespace."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_all_customer_queries(usable_path: str, partially_usable_path: str):
    """Loads all root customer problem statements across usable and partially_usable tiers."""
    queries = []

    with open(usable_path, "r", encoding="utf-8") as f:
        usable = json.load(f)
        for t in usable:
            customer_msg = t["messages"][0]
            queries.append({
                "thread_id": t["thread_id"],
                "tweet_id": customer_msg["tweet_id"],
                "author_id": customer_msg.get("author_id", "customer"),
                "raw_text": customer_msg["text"],
                "cleaned_text": clean_customer_query(customer_msg["text"]),
                "tier": "usable"
            })

    with open(partially_usable_path, "r", encoding="utf-8") as f:
        partially = json.load(f)
        for t in partially:
            customer_msg = t["messages"][0]
            queries.append({
                "thread_id": t["thread_id"],
                "tweet_id": customer_msg["tweet_id"],
                "author_id": customer_msg.get("author_id", "customer"),
                "raw_text": customer_msg["text"],
                "cleaned_text": clean_customer_query(customer_msg["text"]),
                "tier": "partially_usable"
            })

    # Filter out empty or trivially short queries (< 10 chars)
    valid_queries = [q for q in queries if len(q["cleaned_text"]) >= 10]
    return valid_queries


def cluster_customer_problems(queries, n_clusters=7, random_state=42):
    """Performs TF-IDF vectorization and KMeans clustering."""
    corpus = [q["cleaned_text"] for q in queries]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        stop_words="english",
        min_df=3,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(corpus)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=512,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X)

    # Attach labels to queries
    for q, label in zip(queries, cluster_labels):
        q["cluster_id"] = int(label)

    # Compute top keywords per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    cluster_summaries = {}
    counts = Counter(cluster_labels)

    for i in range(n_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_queries = [q for q in queries if q["cluster_id"] == i]
        
        # Sort queries by closeness to cluster center for representative samples
        cluster_center = kmeans.cluster_centers_[i]
        # Calculate distances for cluster members
        member_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        member_vectors = X[member_indices]
        # Cosine similarity to center
        sims = member_vectors.dot(cluster_center)
        # Top 5 most representative indices
        top_rep_indices = np.argsort(sims)[::-1][:5]
        representative_samples = [queries[member_indices[idx]] for idx in top_rep_indices]

        cluster_summaries[f"cluster_{i}"] = {
            "cluster_id": i,
            "count": counts[i],
            "percentage": round((counts[i] / len(queries)) * 100, 2),
            "top_terms": top_words,
            "representative_samples": representative_samples
        }

    return cluster_summaries, queries


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    usable_path = "data/processed/threads_usable.json"
    partially_path = "data/processed/threads_partially_usable.json"

    print("--- Phase 4: Exploring Customer Problems via Clustering ---")
    queries = load_all_customer_queries(usable_path, partially_path)
    print(f"Loaded {len(queries):,} valid customer-initiated messages across both tiers.")

    n_clusters = 7
    summaries, labeled_queries = cluster_customer_problems(queries, n_clusters=n_clusters)

    # Save to disk
    output_path = "data/processed/customer_problem_clusters.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"Saved cluster results to {output_path}")

    # Display results
    print("\n=== CLUSTERING RESULTS (SORTED BY VOLUME) ===")
    sorted_clusters = sorted(summaries.values(), key=lambda x: x["count"], reverse=True)
    for c in sorted_clusters:
        print(f"\nCluster {c['cluster_id']} — Volume: {c['count']:,} queries ({c['percentage']}%)")
        print(f"Top Terms: {', '.join(c['top_terms'][:7])}")
        print("Real Representative Samples:")
        for s in c['representative_samples'][:3]:
            raw_single = s['raw_text'].replace('\n', ' ')
            print(f"  • [Thread: {s['thread_id']} | Tweet: {s['tweet_id']}] {raw_single[:110]}...")
