"""
Phase 1: TWCS Dataset Verification, Schema Inspection, and Brand Analysis.
Designed for memory efficiency: streaming file check and chunked pandas processing.
"""

import os
import sys
import csv
from collections import Counter
import pandas as pd


def verify_twcs_file(file_path: str) -> dict:
    """Verify twcs.csv existence, file size, and exact row count without loading into memory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found at '{file_path}'. "
            "In Colab, download via Kaggle API: "
            "`kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw/ --unzip`"
        )

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Streaming count of rows to avoid memory spikes
    total_lines = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            total_lines += 1

    total_rows = max(0, total_lines - 1)  # Subtract header

    info = {
        "file_path": file_path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "total_lines": total_lines,
        "total_rows": total_rows,
    }
    return info


def inspect_schema_sample(file_path: str, sample_size: int = 100) -> pd.DataFrame:
    """Load a safe small head sample to inspect schema, data types, and null patterns."""
    df_sample = pd.read_csv(file_path, nrows=sample_size)
    return df_sample


def analyze_brands_and_volume(file_path: str, chunksize: int = 200000) -> pd.DataFrame:
    """
    Safely streams through twcs.csv in chunks to compute:
    1. Top outbound brand author_ids (inbound == False).
    2. Total tweets answered/sent by each brand.
    Memory footprint stays strictly under ~150MB.
    """
    brand_counts = Counter()

    # We only need author_id and inbound to find brand volume
    usecols = ["author_id", "inbound"]

    for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=chunksize):
        # Brands are author_ids that have inbound == False
        brands_in_chunk = chunk[chunk["inbound"] == False]["author_id"]
        brand_counts.update(brands_in_chunk.value_counts().to_dict())

    df_brands = pd.DataFrame(brand_counts.most_common(30), columns=["brand_author_id", "tweet_volume"])
    return df_brands


def print_column_explanations():
    """Explains role of each column in multi-turn conversation reconstruction."""
    explanations = {
        "tweet_id": "Unique integer identifier for each tweet. Used as node ID in thread graphs.",
        "author_id": "Identifier of the sender. Brand accounts are usually known strings (e.g., 'AppleSupport', 'AmazonHelp'), while customer IDs are anonymized numeric hashes (e.g., '115712').",
        "inbound": "Boolean flag. True = tweet is incoming from customer to brand. False = tweet is outgoing from brand to customer.",
        "created_at": "Timestamp of tweet (format: '%a %b %d %H:%M:%S +0000 %Y'). Vital for ordering chronological turns within a reconstructed thread.",
        "text": "The raw tweet content, containing customer queries, brand responses, @mentions, and masked sensitive links/entities.",
        "response_tweet_id": "Comma-separated list of child tweet_ids that responded to this tweet (forward edge in thread graph). Can be empty or contain multiple IDs (branching).",
        "in_response_to_tweet_id": "Single parent tweet_id that this tweet is replying to (backward edge in thread graph). If NaN/empty, this tweet is the ROOT (originating message) of a conversation thread."
    }
    return explanations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect TWCS dataset")
    parser.add_argument("--path", default="data/raw/twcs.csv", help="Path to twcs.csv")
    args = parser.parse_args()

    print("=== Phase 1: TWCS Dataset Verification ===")
    info = verify_twcs_file(args.path)
    print(f"File verified: {info['file_path']}")
    print(f"File size: {info['file_size_mb']} MB")
    print(f"Total rows: {info['total_rows']:,}")

    print("\n=== Schema and Sample ===")
    sample = inspect_schema_sample(args.path, 5)
    print("Columns:", list(sample.columns))
    print(sample.head())

    print("\n=== Column Reconstruction Roles ===")
    for col, desc in print_column_explanations().items():
        print(f"• {col}: {desc}")

    print("\n=== Scanning Top Brands (Chunked) ===")
    brands = analyze_brands_and_volume(args.path)
    print(brands.to_string(index=False))
