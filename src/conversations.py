"""
Phase 2: Conversation Thread Reconstruction for AmazonHelp.
Transforms raw TWCS tweet rows into structured, multi-turn conversation threads:
{
    "thread_id": str,
    "brand": "AmazonHelp",
    "messages": [{"speaker": "customer"|"brand", "text": str, "tweet_id": int, "created_at": str}],
    "resolution": str | None,
    "resolved": bool
}
"""

import os
import json
import re
import pandas as pd
from typing import List, Dict, Generator, Any


def extract_brand_tweets(
    raw_csv_path: str,
    output_brand_csv: str,
    brand_handle: str = "AmazonHelp",
    max_tweets: int = 150000,
    chunksize: int = 200000
) -> str:
    """
    Extracts tweets belonging to or addressing the target brand in memory-efficient chunks.
    Saves to data/processed/{brand_handle}_raw.csv.
    """
    os.makedirs(os.path.dirname(output_brand_csv), exist_ok=True)

    if os.path.exists(output_brand_csv):
        print(f"[Notice] Filtered brand CSV already exists at: {output_brand_csv}")
        return output_brand_csv

    print(f"Extracting tweets for {brand_handle} from {raw_csv_path}...")
    brand_lower = brand_handle.lower()
    total_extracted = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunksize, low_memory=False)):
        # Match tweets authored by the brand OR inbound tweets mentioning the brand
        brand_mask = (chunk["author_id"] == brand_handle) | (
            (chunk["inbound"] == True) & (chunk["text"].str.lower().str.contains(f"@{brand_lower}", na=False))
        )
        filtered = chunk[brand_mask]

        if not filtered.empty:
            filtered.to_csv(
                output_brand_csv,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )
            first_chunk = False
            total_extracted += len(filtered)

        print(f"Chunk {i+1}: extracted {total_extracted:,} tweets so far...")
        if total_extracted >= max_tweets:
            print(f"Reached cap of {max_tweets} tweets for fast sub-15 min processing.")
            break

    print(f"Extraction complete. Total tweets saved: {total_extracted:,} -> {output_brand_csv}")
    return output_brand_csv


def reconstruct_threads(df: pd.DataFrame, brand_handle: str = "AmazonHelp") -> List[Dict[str, Any]]:
    """
    Reconstructs conversation threads from tweet rows using in_response_to_tweet_id chaining.
    """
    # Build lookup dictionaries
    # Convert IDs to nullable ints/strings to handle NaNs safely
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df = df.dropna(subset=["tweet_id"])
    df["tweet_id"] = df["tweet_id"].astype(int)

    tweet_map = {}
    children_map = {}

    for _, row in df.iterrows():
        t_id = row["tweet_id"]
        parent_id = row["in_response_to_tweet_id"]
        parent_id = int(parent_id) if pd.notnull(parent_id) else None

        tweet_map[t_id] = {
            "tweet_id": t_id,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": str(row["created_at"]),
            "text": str(row["text"]),
            "parent_id": parent_id
        }

        if parent_id is not None:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(t_id)

    # Identify candidate ROOT tweets:
    # A root is an inbound tweet (customer message) whose parent is None OR whose parent is not in our tweet map.
    root_ids = [
        t_id for t_id, t in tweet_map.items()
        if t["inbound"] and (t["parent_id"] is None or t["parent_id"] not in tweet_map)
    ]

    print(f"Found {len(root_ids):,} candidate root customer queries.")

    threads = []
    dm_regex = re.compile(r"\b(dm|direct message|send us a dm|link below|reach out to us on)\b", re.IGNORECASE)
    resolution_keywords = re.compile(r"\b(resolved|sorted|fixed|refund issued|replacement sent|glad to hear|glad we could help|pleasure|happy to help|welcome|thanks for confirming)\b", re.IGNORECASE)

    for root_id in root_ids:
        # Traverse forward to build chronological thread
        curr_id = root_id
        thread_messages = []
        visited = set()

        while curr_id and curr_id in tweet_map and curr_id not in visited:
            visited.add(curr_id)
            node = tweet_map[curr_id]
            speaker = "brand" if node["author_id"] == brand_handle or not node["inbound"] else "customer"
            thread_messages.append({
                "tweet_id": node["tweet_id"],
                "speaker": speaker,
                "created_at": node["created_at"],
                "text": node["text"]
            })

            # Check for child replies
            children = children_map.get(curr_id, [])
            if children:
                # Follow the first child (primary conversational path)
                curr_id = children[0]
            else:
                curr_id = None

        # Filter: Only keep threads that have at least 1 customer turn AND at least 1 brand reply
        speakers = [m["speaker"] for m in thread_messages]
        if "customer" in speakers and "brand" in speakers:
            # Baseline resolution analysis:
            # Check last brand message and last customer message
            last_msg = thread_messages[-1]
            last_brand_msg = next((m["text"] for m in reversed(thread_messages) if m["speaker"] == "brand"), "")
            
            # Heuristic for resolved in raw data:
            # 1. If it was deflected to DM -> NOT resolved in public thread
            is_dm_deflected = bool(dm_regex.search(last_brand_msg))
            # 2. Did brand confirm resolution or customer express closure?
            has_resolution_signal = bool(resolution_keywords.search(last_brand_msg)) or bool(resolution_keywords.search(last_msg["text"]))
            
            # If thread ends with unanswered customer question -> unresolved
            ends_with_customer = (last_msg["speaker"] == "customer")

            if is_dm_deflected or ends_with_customer:
                resolved = False
                resolution = "Thread ended with private DM redirection or unanswered customer reply."
            elif has_resolution_signal:
                resolved = True
                resolution = last_brand_msg
            else:
                resolved = False
                resolution = "Conversation concluded without explicit public resolution confirmation."

            threads.append({
                "thread_id": str(root_id),
                "brand": brand_handle,
                "turns_count": len(thread_messages),
                "messages": thread_messages,
                "resolution": resolution,
                "resolved": resolved
            })

    print(f"Successfully reconstructed {len(threads):,} multi-turn threads for {brand_handle}.")
    return threads


def save_threads(threads: List[Dict[str, Any]], output_json_path: str):
    """Saves reconstructed threads to JSON."""
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(threads):,} threads to {output_json_path}")


if __name__ == "__main__":
    raw_path = "data/raw/twcs.csv"
    brand_csv = "data/processed/amazon_raw.csv"
    threads_json = "data/processed/amazon_threads.json"

    if os.path.exists(raw_path):
        extract_brand_tweets(raw_path, brand_csv, brand_handle="AmazonHelp", max_tweets=80000)
        df_brand = pd.read_csv(brand_csv, low_memory=False)
        threads = reconstruct_threads(df_brand, brand_handle="AmazonHelp")
        save_threads(threads, threads_json)
    else:
        print(f"[Warning] {raw_path} does not exist yet. Run in Colab or download twcs.csv first.")
