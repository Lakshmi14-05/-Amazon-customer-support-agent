"""
Executes the upgraded Phase 3 filter on the committed 100MB dataset slice.
Verifies exclusion of edge cases (Thread 101674, 21941),
prints updated category counts, and draws 10 reproducible random samples per tier.
"""

import os
import csv
import json
import random
from collections import defaultdict
import sys

# Ensure src is importable
sys.path.append(os.path.abspath("src"))
from data_processing import classify_thread_quality, is_english_text, check_tightened_confirmation
from run_real_reconstruction_and_sample import load_real_twcs_rows, reconstruct_all_threads


def verify_specific_threads(threads):
    """Explicitly verifies whether Thread 101674 and Thread 21941 are correctly reclassified."""
    target_ids = {"101674", "21941"}
    found_targets = {}

    for t in threads:
        if t["thread_id"] in target_ids:
            cat, reason = classify_thread_quality(t)
            found_targets[t["thread_id"]] = {
                "thread_id": t["thread_id"],
                "quality_label": cat,
                "reason": reason,
                "closing_text": t["messages"][-1]["text"]
            }

    return found_targets


if __name__ == "__main__":
    csv_path = "data/raw/twcs_subsample_100mb.csv"
    print(f"--- Loading Committed 100MB Dataset Slice: {csv_path} ---")
    tweets, children_map = load_real_twcs_rows(csv_path, target_brand="AmazonHelp", max_rows=700000)
    threads = reconstruct_all_threads(tweets, children_map, brand_handle="AmazonHelp")

    # Check Thread 101674 and 21941
    target_results = verify_specific_threads(threads)
    print("\n=== VERIFICATION OF THREAD 101674 & THREAD 21941 ===")
    for tid, info in target_results.items():
        print(f"• Thread {tid}:")
        print(f"    Closing Message: \"{info['closing_text']}\"")
        print(f"    Assigned Category: {info['quality_label'].upper()}")
        print(f"    Reason: {info['reason']}")

    # Classify all threads
    classified = defaultdict(list)
    for t in threads:
        cat, reason = classify_thread_quality(t)
        t["quality_label"] = cat
        t["quality_reason"] = reason
        classified[cat].append(t)

    print("\n=== UPDATED CLASSIFICATION COUNTS (COMMITTED 100MB SLICE) ===")
    total = len(threads)
    for cat in ["usable", "partially_usable", "unusable"]:
        count = len(classified[cat])
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {cat.upper():<18}: {count:>5} ({pct:>5.1f}%)")
    print(f"  TOTAL THREADS     : {total:>5}")

    # Set reproducible seed for fresh random sampling
    random.seed(12345)

    sample_output = {}
    for cat in ["usable", "partially_usable", "unusable"]:
        pool = classified[cat]
        sample_size = min(10, len(pool))
        sample_output[cat] = random.sample(pool, sample_size)

    output_path = "data/processed/committed_random_samples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved fresh random samples to {output_path}")
