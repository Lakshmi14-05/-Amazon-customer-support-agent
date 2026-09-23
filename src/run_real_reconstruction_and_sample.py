"""
Script to reconstruct threads from real TWCS data, classify their quality,
and draw reproducible random samples from each category (usable, partially_usable, unusable).
"""

import os
import csv
import json
import re
import random
from collections import defaultdict

# Regex patterns matching exact logic in data_processing.py
DM_DEFLECTION_PATTERN = re.compile(
    r"\b(dm|direct message|send us a dm|dm us|message us|reach out via dm|click here to dm|https://t\.co/\w+)\b",
    re.IGNORECASE
)

EXPLICIT_CONFIRMATION_PATTERN = re.compile(
    r"\b(thank you|thanks|worked|sorted|fixed|appreciate it|awesome thanks|got it|perfect thanks|all good now|resolved now|that helped|danke)\b",
    re.IGNORECASE
)

COMPLETED_ACTION_PATTERN = re.compile(
    r"\b(refund processed|replacement sent|ticket #?\w+|issue resolved|credited your account|canceled your order|updated your shipping|fix has been deployed|re-ordered)\b",
    re.IGNORECASE
)

POLICY_INSTRUCTION_PATTERN = re.compile(
    r"\b(go to your account|navigate to|you can return this within|manage your content and devices|turn off auto-renew|change payment method|unmute the microphone|settings >|your orders >|select 'return for refund'|check out our echo help pages)\b",
    re.IGNORECASE
)

SPAM_OR_TRIVIAL_PATTERN = re.compile(
    r"^(@\w+\s*)*(hi|hello|hey|help|\?+|dm|\.|\!+|please help me)$",
    re.IGNORECASE
)

NON_ENGLISH_PATTERN = re.compile(
    r"\b(wir|haben|danke|bitte|schon|abend|gruay|hola|gracias|ayuda|por favor|merci|bonjour)\b",
    re.IGNORECASE
)


def load_real_twcs_rows(csv_path, target_brand="AmazonHelp", max_rows=120000):
    """Loads and filters real tweets for target_brand from TWCS CSV."""
    tweets = {}
    children_map = defaultdict(list)

    print(f"Reading real rows from {csv_path}...")
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                t_id = int(row["tweet_id"])
            except ValueError:
                continue

            author = row["author_id"]
            inbound = row["inbound"].strip().lower() == "true"
            text = row["text"]
            created_at = row["created_at"]
            
            p_id_raw = row.get("in_response_to_tweet_id")
            p_id_str = p_id_raw.strip() if p_id_raw else ""
            parent_id = None
            if p_id_str:
                try:
                    parent_id = int(float(p_id_str))
                except ValueError:
                    parent_id = None

            # Keep tweet if authored by brand or text mentions brand
            if author.lower() == target_brand.lower() or (f"@{target_brand.lower()}" in text.lower()):
                tweets[t_id] = {
                    "tweet_id": t_id,
                    "author_id": author,
                    "inbound": inbound,
                    "created_at": created_at,
                    "text": text,
                    "parent_id": parent_id
                }
                if parent_id is not None:
                    children_map[parent_id].append(t_id)

    print(f"Loaded {len(tweets):,} relevant {target_brand} tweets.")
    return tweets, children_map


def reconstruct_all_threads(tweets, children_map, brand_handle="AmazonHelp"):
    """Reconstructs multi-turn conversation threads."""
    # Root tweets: inbound customer tweets whose parent is None or parent not in dataset
    root_ids = [
        t_id for t_id, t in tweets.items()
        if t["inbound"] and (t["parent_id"] is None or t["parent_id"] not in tweets)
    ]

    threads = []
    for root_id in root_ids:
        curr_id = root_id
        thread_messages = []
        visited = set()

        while curr_id and curr_id in tweets and curr_id not in visited:
            visited.add(curr_id)
            node = tweets[curr_id]
            speaker = "brand" if node["author_id"].lower() == brand_handle.lower() or not node["inbound"] else "customer"
            thread_messages.append({
                "tweet_id": node["tweet_id"],
                "author_id": node["author_id"],
                "speaker": speaker,
                "created_at": node["created_at"],
                "text": node["text"]
            })

            children = children_map.get(curr_id, [])
            curr_id = children[0] if children else None

        speakers = [m["speaker"] for m in thread_messages]
        # Keep threads that have at least 1 customer turn AND at least 1 brand turn
        if "customer" in speakers and "brand" in speakers:
            threads.append({
                "thread_id": str(root_id),
                "brand": brand_handle,
                "turns_count": len(thread_messages),
                "messages": thread_messages
            })

    print(f"Reconstructed {len(threads):,} multi-turn threads.")
    return threads


def classify_thread_quality(thread):
    """
    Exact classification function implemented in data_processing.py
    """
    messages = thread.get("messages", [])

    if len(messages) < 2:
        return "unusable", "Orphaned or single-turn tweet; lacks dialog exchange."

    customer_messages = [m for m in messages if m["speaker"] == "customer"]
    brand_messages = [m for m in messages if m["speaker"] == "brand"]

    if not customer_messages or not brand_messages:
        return "unusable", "Missing either customer problem statement or brand response."

    first_customer_text = customer_messages[0]["text"].strip()
    cleaned_problem = re.sub(r"@\w+", "", first_customer_text).strip()

    # Discard non-English tweets
    if NON_ENGLISH_PATTERN.search(first_customer_text) or NON_ENGLISH_PATTERN.search(brand_messages[0]["text"]):
        return "unusable", "Non-English tweet interaction."

    # Discard trivial spam / hello / empty queries
    if len(cleaned_problem) < 10 or SPAM_OR_TRIVIAL_PATTERN.match(cleaned_problem):
        return "unusable", "Customer query is trivial, spam, or lacks a coherent problem description."

    last_turn = messages[-1]
    last_brand_msg = brand_messages[-1]["text"].strip()
    last_customer_msg = customer_messages[-1]["text"].strip()

    # A. Check for usable criteria:
    # Customer confirms in final message OR brand took action OR complete instruction without deflection
    customer_confirmed = bool(EXPLICIT_CONFIRMATION_PATTERN.search(last_customer_msg)) and (last_turn["speaker"] == "customer")
    brand_action_taken = bool(COMPLETED_ACTION_PATTERN.search(last_brand_msg))
    
    brand_instruction_complete = (
        bool(POLICY_INSTRUCTION_PATTERN.search(last_brand_msg))
        and not bool(DM_DEFLECTION_PATTERN.search(last_brand_msg))
        and len(last_brand_msg) > 50
    )

    is_final_turn_deflection = bool(DM_DEFLECTION_PATTERN.search(last_turn["text"]))

    if (customer_confirmed or brand_action_taken or brand_instruction_complete) and not is_final_turn_deflection:
        return "usable", "Clear customer problem with verified resolution, explicit customer confirmation, or concrete completed action."

    # B. Partially Usable:
    # Substantive issue present, but deflected to DM, phone/chat link, or ends ambiguously
    if len(cleaned_problem) >= 10:
        if is_final_turn_deflection or bool(DM_DEFLECTION_PATTERN.search(last_brand_msg)):
            return "partially_usable", "Substantive customer issue present, but brand redirected to private DM/link without public resolution."
        elif last_turn["speaker"] == "customer":
            return "partially_usable", "Customer follow-up was left unanswered by the brand."
        else:
            return "partially_usable", "Brand engaged with clarification or partial advice, but outcome remains unconfirmed."

    return "unusable", "Fails usability criteria."


if __name__ == "__main__":
    csv_path = "data/raw/twcs_slice_15mb.csv"
    tweets, children_map = load_real_twcs_rows(csv_path, target_brand="AmazonHelp")
    threads = reconstruct_all_threads(tweets, children_map, brand_handle="AmazonHelp")

    classified = defaultdict(list)
    for t in threads:
        cat, reason = classify_thread_quality(t)
        t["quality_label"] = cat
        t["quality_reason"] = reason
        classified[cat].append(t)

    print("\n=== CLASSIFICATION COUNTS ON REAL DATA SLICE ===")
    total = len(threads)
    for cat in ["usable", "partially_usable", "unusable"]:
        count = len(classified[cat])
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {cat.upper():<18}: {count:>4} ({pct:.1f}%)")

    # Set random seed for true reproducible random sampling
    random.seed(42)

    sample_output = {}
    for cat in ["usable", "partially_usable", "unusable"]:
        pool = classified[cat]
        sample_size = min(10, len(pool))
        sample_output[cat] = random.sample(pool, sample_size)

    # Save to JSON for inspection
    with open("data/processed/real_random_samples.json", "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2, ensure_ascii=False)

    print("\nSuccessfully sampled 10 random threads per category and saved to data/processed/real_random_samples.json")
