"""
index_documents.py

Loads the training dataset and indexes all documents into ChromaDB.
Run this after training so the /search endpoint has something to query.

Usage:
    python scripts/index_documents.py
    python scripts/index_documents.py --clear     # wipe existing index first
    python scripts/index_documents.py --batch_size 200
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.utils.helpers import load_config, setup_logging
from src.utils.data_loader import get_dataset
from src.vector_store.chroma_store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(description="Index documents into ChromaDB")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing collection before indexing")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for embedding + indexing")
    parser.add_argument("--subset", type=str, choices=["train", "test", "all"],
                        default="all", help="Which split to index")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["logging"]["level"], config["logging"]["log_file"])

    vs_cfg = config["vector_store"]
    store = ChromaVectorStore(
        collection_name=vs_cfg["collection_name"],
        persist_directory=vs_cfg["persist_directory"],
        embedding_model=vs_cfg["embedding_model"],
    )

    if args.clear:
        logger.warning("Clearing existing collection...")
        store.clear()

    existing = store.get_document_count()
    logger.info(f"Current documents in store: {existing:,}")

    # load dataset
    train_texts, train_labels, test_texts, test_labels = get_dataset(config["data"])

    # decide what to index
    if args.subset == "train":
        texts, labels = train_texts, train_labels
    elif args.subset == "test":
        texts, labels = test_texts, test_labels
    else:  # all
        texts = train_texts + test_texts
        labels = train_labels + test_labels

    logger.info(f"Indexing {len(texts):,} documents ({args.subset} split)...")

    # build metadata — store the label so we can filter by category later
    metadatas = [
        {"label": label, "split": "train" if i < len(train_texts) else "test"}
        for i, label in enumerate(labels)
    ]

    n_indexed = store.add_documents(
        texts=texts,
        metadatas=metadatas,
        batch_size=args.batch_size,
    )

    logger.success(f"Indexed {n_indexed:,} documents. Total in store: {store.get_document_count():,}")

    # quick search test to confirm everything works
    logger.info("\nRunning quick search test...")
    test_queries = [
        "rocket launch space exploration",
        "machine learning neural networks",
        "stock market financial news",
    ]
    for query in test_queries:
        results = store.search(query, top_k=2)
        logger.info(f"  Query: '{query}'")
        for r in results:
            logger.info(f"    [{r['similarity']:.3f}] [{r['metadata'].get('label', '?')}] {r['text'][:80]}...")


if __name__ == "__main__":
    main()
