"""
Analyzes the distribution of all 231 USABLE threads across rough problem categories.
Saves the full partitions: threads_usable.json and threads_partially_usable.json.
"""

import os
import re
import json
from collections import defaultdict
import sys

sys.path.append(os.path.abspath("src"))
from data_processing import classify_thread_quality
from run_real_reconstruction_and_sample import load_real_twcs_rows, reconstruct_all_threads

# Expanded topic patterns based on Twitter support phrasing
TOPIC_PATTERNS = {
    "delivery_shipping_delay": re.compile(
        r"\b(shipping|delivery|delay|delayed|late|transit|dispatch|dispatched|courier|carrier|track|tracking|arriving|estimate|where is my|still waiting|postman|driver)\b",
        re.IGNORECASE
    ),
    "damaged_defective_item": re.compile(
        r"\b(damage|damaged|broken|smashed|wet|ruined|defective|faulty|cracked|shattered|torn|scratched|dented)\b",
        re.IGNORECASE
    ),
    "missing_item_or_package": re.compile(
        r"\b(missing|stolen|handed to resident|not received|never received|empty box|lost package|didn't receive|not delivered|nowhere to be found)\b",
        re.IGNORECASE
    ),
    "refund_return_exchange": re.compile(
        r"\b(refund|return|returning|exchange|replace|replacement|return label|pass my parcel|send back|drop off|refunded)\b",
        re.IGNORECASE
    ),
    "account_security_login": re.compile(
        r"\b(login|log in|hacked|password|security|email changed|unauthorized|sign in|locked|compromised|account)\b",
        re.IGNORECASE
    ),
    "billing_payment_prime": re.compile(
        r"\b(charge|charged|payment|billed|double charge|subscription|prime|cancel prime|auto-renew|fee|card|bank|invoice)\b",
        re.IGNORECASE
    ),
    "digital_hardware_devices": re.compile(
        r"\b(echo|alexa|kindle|fire tv|firestick|prime video|app|music|dot|ring|device|streaming|tv|audio)\b",
        re.IGNORECASE
    ),
    "general_policy_promotion": re.compile(
        r"\b(affiliate|policy|gift card|pre-order|guarantee|promo|code|cash back|pantry|coupon|voucher|price)\b",
        re.IGNORECASE
    )
}


def assign_rough_category(customer_text):
    matched = []
    for topic, pattern in TOPIC_PATTERNS.items():
        if pattern.search(customer_text):
            matched.append(topic)
    
    if not matched:
        return "other_unclassified"
    return matched[0]


if __name__ == "__main__":
    # Force utf-8 standard output
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    csv_path = "data/raw/twcs_subsample_100mb.csv"
    tweets, children_map = load_real_twcs_rows(csv_path, target_brand="AmazonHelp", max_rows=700000)
    threads = reconstruct_all_threads(tweets, children_map, brand_handle="AmazonHelp")

    usable_threads = []
    partially_usable_threads = []
    unusable_threads = []

    for t in threads:
        cat, reason = classify_thread_quality(t)
        t["quality_label"] = cat
        t["quality_reason"] = reason
        if cat == "usable":
            usable_threads.append(t)
        elif cat == "partially_usable":
            partially_usable_threads.append(t)
        else:
            unusable_threads.append(t)

    # Save full files
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/threads_usable.json", "w", encoding="utf-8") as f:
        json.dump(usable_threads, f, indent=2, ensure_ascii=False)
    with open("data/processed/threads_partially_usable.json", "w", encoding="utf-8") as f:
        json.dump(partially_usable_threads, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(usable_threads):,} usable threads to data/processed/threads_usable.json")

    # Group usable threads by rough topic
    topic_distribution = defaultdict(list)
    for t in usable_threads:
        root_customer_msg = t["messages"][0]["text"]
        cat = assign_rough_category(root_customer_msg)
        topic_distribution[cat].append(t)

    print("\n=== BREAKDOWN OF 231 USABLE THREADS BY PROBLEM CATEGORY ===")
    total_usable = len(usable_threads)
    sorted_topics = sorted(topic_distribution.items(), key=lambda x: len(x[1]), reverse=True)

    for topic, items in sorted_topics:
        pct = (len(items) / total_usable) * 100
        print(f"\n• {topic.upper()} : {len(items)} threads ({pct:.1f}%)")
        print("  Sample Customer Queries:")
        for sample in items[:3]:
            t_text = sample['messages'][0]['text'].replace('\n', ' ')
            print(f"    - [ID: {sample['thread_id']}] {t_text[:110]}...")
