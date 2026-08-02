"""
Semantic Deduplication: Remove near-duplicate questions using embeddings.

Groups samples by db_id, generates sentence embeddings for questions,
and uses FAISS cosine similarity search to identify and remove duplicates
above a threshold.

Usage:
    python pipeline/steps/semantic_dedup.py \
        --input_file results/all_evolved.json \
        --output_file results/all_evolved_dedup.json \
        --threshold 0.90 \
        --batch_size 1024
"""

import argparse
import sys
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import setup_logging


def semantic_deduplication(
    input_path: str,
    output_path: str,
    threshold: float = 0.90,
    batch_size: int = 1024,
    model_name: str = "all-mpnet-base-v2",
    num_examples: int = 5,
):
    """
    Perform semantic deduplication on a JSON dataset of questions.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output deduplicated JSON file
        threshold: Similarity threshold for duplicates (0.0 to 1.0)
        batch_size: Batch size for encoding
        model_name: SentenceTransformer model name or path
        num_examples: Number of duplicate examples to print
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_json(input_path)

    print(f"Initializing SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)

    indices_to_keep = []
    duplicate_examples = []

    # Group by db_id
    grouped = df.groupby("db_id")

    print("Processing groups...")
    for db_id, group in tqdm(grouped, desc="Deduplicating"):
        if len(group) <= 1:
            indices_to_keep.extend(group.index)
            continue

        questions = group["question"].tolist()

        # Generate embeddings
        embeddings = model.encode(
            questions, convert_to_numpy=True,
            show_progress_bar=False, batch_size=batch_size,
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Search top 2 neighbors (first is self)
        distances, neighbors = index.search(embeddings, k=min(len(group), 2))

        # Identify duplicates
        to_discard = set()
        for i in range(len(questions)):
            if neighbors.shape[1] > 1:
                neighbor_idx = neighbors[i][1]
                similarity = distances[i][1]

                if similarity > threshold:
                    original_current_idx = group.index[i]
                    original_neighbor_idx = group.index[neighbor_idx]

                    if original_current_idx in to_discard or original_neighbor_idx in to_discard:
                        continue

                    # Discard the higher-indexed one
                    discard_idx, keep_idx = (
                        (original_current_idx, original_neighbor_idx)
                        if original_current_idx > original_neighbor_idx
                        else (original_neighbor_idx, original_current_idx)
                    )

                    if discard_idx not in to_discard:
                        to_discard.add(discard_idx)
                        duplicate_examples.append({
                            "discarded_question": df.loc[discard_idx]["question"],
                            "kept_question": df.loc[keep_idx]["question"],
                            "similarity": float(similarity),
                            "db_id": db_id,
                        })

        group_indices = set(group.index)
        indices_to_keep.extend(list(group_indices - to_discard))

    # Filter and save
    deduplicated_df = df.loc[sorted(indices_to_keep)].reset_index(drop=True)

    print(f"\nOriginal count:      {len(df)}")
    print(f"After deduplication: {len(deduplicated_df)}")
    print(f"Removed:             {len(df) - len(deduplicated_df)}")

    if num_examples > 0 and duplicate_examples:
        print(f"\n--- {min(num_examples, len(duplicate_examples))} duplicate examples ---")
        random.shuffle(duplicate_examples)
        for ex in duplicate_examples[:num_examples]:
            print(f"\n[DB: {ex['db_id']}] Similarity: {ex['similarity']:.4f}")
            print(f"  Kept:      {ex['kept_question']}")
            print(f"  Discarded: {ex['discarded_question']}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    deduplicated_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"\nSaved to {output_path}")


def run(args):
    """
    Core logic for semantic deduplication. Can be called from pipeline or CLI.

    Args:
        args: argparse.Namespace with all required fields.
    """
    setup_logging()

    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_dedup{ext}"

    semantic_deduplication(
        input_path=args.input_file,
        output_path=args.output_file,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_name=args.model_name,
        num_examples=args.num_examples,
    )


def main():
    parser = argparse.ArgumentParser(description="Semantic Deduplication")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold")
    parser.add_argument("--batch_size", type=int, default=1024, help="Encoding batch size")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2",
                        help="SentenceTransformer model name or path")
    parser.add_argument("--num_examples", type=int, default=5, help="Duplicate examples to show")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
