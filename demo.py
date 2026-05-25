"""
demo.py

Quick demo that shows the full pipeline working in your terminal.
Run this AFTER training to verify everything works before pushing.

Usage:
    python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from loguru import logger
logger.remove()  # silent for demo


def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def demo_tfidf():
    separator("DEMO 1 — TF-IDF Classifier (Baseline)")

    from src.utils.data_loader import load_20newsgroups
    from src.models.tfidf_classifier import TfidfClassifier

    print("  Loading dataset...")
    train_texts, train_labels, test_texts, test_labels = load_20newsgroups()

    print("  Training TF-IDF + SVM...")
    clf = TfidfClassifier(classifier_type="svm")
    clf.train(train_texts, train_labels)

    sample_texts = [
        "NASA launches new mission to explore Jupiter's moons",
        "Stock market hits record high after Fed rate decision",
        "Scientists develop new mRNA vaccine for influenza",
    ]

    print("\n  Sample Predictions:")
    for text in sample_texts:
        result = clf.predict_single(text)
        print(f"  [{result['confidence']:.2f}] {result['label']:<30} → {text[:50]}...")


def demo_extraction():
    separator("DEMO 2 — Information Extraction (NER + Key Phrases)")

    from src.extraction.information_extractor import InformationExtractor

    extractor = InformationExtractor()

    text = """
    Elon Musk's SpaceX launched a Falcon 9 rocket from Cape Canaveral
    on March 15, 2024. The mission cost $67 million. Contact info@spacex.com
    or visit www.spacex.com for updates.
    """

    print(f"  Input: {text.strip()[:80]}...")
    result = extractor.extract(text)

    print("\n  Extracted Entities:")
    for ent in result["entities"]:
        print(f"    [{ent['type']:<10}] {ent['text']}")

    print(f"\n  Key Phrases: {', '.join(result['key_phrases'][:5])}")

    if result["patterns"]:
        print("\n  Patterns Found:")
        for pat_type, matches in result["patterns"].items():
            print(f"    {pat_type}: {matches}")


def demo_vector_search():
    separator("DEMO 3 — Semantic Vector Search (ChromaDB)")

    from src.vector_store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        persist_directory="/tmp/demo_chroma",
        collection_name="demo_collection",
    )
    store.clear()

    docs = [
        "SpaceX Falcon 9 rocket successfully launches into orbit",
        "NASA announces new Mars rover mission for 2026",
        "Federal Reserve raises interest rates by 25 basis points",
        "Apple reports record quarterly revenue from iPhone sales",
        "Python programming language releases version 3.13 with JIT",
        "Machine learning model beats human performance on NLP task",
        "Scientists discover new treatment for Alzheimer's disease",
        "England wins cricket test series against Australia",
    ]
    metas = [{"topic": t} for t in [
        "space", "space", "finance", "finance",
        "technology", "technology", "health", "sports"
    ]]

    store.add_documents(docs, metas)

    queries = [
        "rocket launch space exploration",
        "stock market money economy",
        "artificial intelligence deep learning",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = store.search(query, top_k=2)
        for r in results:
            print(f"    [{r['similarity']:.3f}] {r['text']}")


def main():
    print("\n" + "="*55)
    print("  Text Classification & Extraction — Live Demo")
    print("="*55)
    print("\n  Running 3 demos:")
    print("  1. TF-IDF text classification")
    print("  2. Named entity extraction")
    print("  3. Semantic vector search")

    try:
        demo_tfidf()
    except Exception as e:
        print(f"  [SKIP] TF-IDF demo failed: {e}")
        print("  Run: python scripts/train.py first")

    try:
        demo_extraction()
    except Exception as e:
        print(f"  [SKIP] Extraction demo failed: {e}")
        print("  Run: python setup.py first")

    try:
        demo_vector_search()
    except Exception as e:
        print(f"  [SKIP] Vector search demo failed: {e}")

    print("\n" + "="*55)
    print("  Demo complete. Start the full API with:")
    print("  uvicorn api.main:app --reload")
    print("  Then open: http://localhost:8000/docs")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
