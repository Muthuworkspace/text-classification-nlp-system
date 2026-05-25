"""
train.py

Main training script. Trains all three classifiers on the configured dataset
and saves them to disk. Run this before starting the API.

Usage:
    python scripts/train.py                          # train all models
    python scripts/train.py --model tfidf            # train only TF-IDF
    python scripts/train.py --model transformer      # train only transformer
    python scripts/train.py --dataset custom         # use custom CSV
"""

import argparse
import sys
import os

# make sure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from src.utils.helpers import load_config, setup_logging, format_metrics, ensure_dirs
from src.utils.data_loader import get_dataset


def train_tfidf(train_texts, train_labels, test_texts, test_labels, config):
    from src.models.tfidf_classifier import TfidfClassifier

    logger.info("=" * 50)
    logger.info("Training TF-IDF Classifier")
    logger.info("=" * 50)

    clf = TfidfClassifier(
        classifier_type=config["tfidf"]["classifier"],
        tfidf_max_features=config["tfidf"]["max_features"],
    )
    train_result = clf.train(train_texts, train_labels)
    eval_result = clf.evaluate(test_texts, test_labels)

    clf.save("models/tfidf_classifier")
    logger.success(f"TF-IDF done | Test Acc: {eval_result['accuracy']:.4f}")
    return eval_result


def train_embedding(train_texts, train_labels, test_texts, test_labels, config):
    from src.models.embedding_classifier import EmbeddingClassifier

    logger.info("=" * 50)
    logger.info("Training Embedding Classifier (KNN)")
    logger.info("=" * 50)

    clf = EmbeddingClassifier(
        embedding_model=config["embedding"]["model_name"],
        classifier_type="knn",
    )
    clf.train(train_texts, train_labels)
    eval_result = clf.evaluate(test_texts, test_labels)

    clf.save("models/embedding_classifier")
    logger.success(f"Embedding done | Test Acc: {eval_result['accuracy']:.4f}")
    return eval_result


def train_transformer(train_texts, train_labels, test_texts, test_labels, config):
    from src.models.transformer_classifier import TransformerClassifier
    from sklearn.model_selection import train_test_split

    logger.info("=" * 50)
    logger.info(f"Fine-tuning {config['transformer']['model_name']}")
    logger.info("=" * 50)

    # carve out a small validation split from training data
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    clf = TransformerClassifier(
        model_name=config["transformer"]["model_name"],
        num_epochs=config["transformer"]["num_epochs"],
        batch_size=config["transformer"]["batch_size"],
        learning_rate=config["transformer"]["learning_rate"],
        warmup_steps=config["transformer"]["warmup_steps"],
        weight_decay=config["transformer"]["weight_decay"],
        max_seq_length=config["transformer"]["max_seq_length"],
        save_path=config["transformer"]["save_path"],
    )
    clf.train(tr_texts, tr_labels, val_texts=val_texts, val_labels=val_labels)
    eval_result = clf.evaluate(test_texts, test_labels)

    logger.success(f"Transformer done | Test Acc: {eval_result['accuracy']:.4f}")
    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Train text classification models")
    parser.add_argument("--model", choices=["tfidf", "embedding", "transformer", "all"],
                        default="all", help="Which model(s) to train")
    parser.add_argument("--dataset", choices=["20newsgroups", "custom"],
                        default=None, help="Dataset override (default: from config)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["logging"]["level"], config["logging"]["log_file"])
    ensure_dirs("models/tfidf_classifier", "models/embedding_classifier",
                config["transformer"]["save_path"], "logs")

    if args.dataset:
        config["data"]["dataset"] = args.dataset

    # load dataset
    train_texts, train_labels, test_texts, test_labels = get_dataset(config["data"])
    logger.info(f"Dataset loaded | Train: {len(train_texts)} | Test: {len(test_texts)}")

    all_results = {}

    if args.model in ("tfidf", "all"):
        all_results["tfidf"] = train_tfidf(train_texts, train_labels, test_texts, test_labels, config)

    if args.model in ("embedding", "all"):
        all_results["embedding"] = train_embedding(train_texts, train_labels, test_texts, test_labels, config)

    if args.model in ("transformer", "all"):
        all_results["transformer"] = train_transformer(train_texts, train_labels, test_texts, test_labels, config)

    # final comparison
    logger.info("\n" + "=" * 50)
    logger.info("  FINAL RESULTS COMPARISON")
    logger.info("=" * 50)
    for model_name, result in all_results.items():
        logger.info(f"  {model_name:<20} → Accuracy: {result['accuracy']:.4f}")
    logger.info("=" * 50)
    logger.success("All models trained. Run scripts/index_documents.py next.")


if __name__ == "__main__":
    main()
