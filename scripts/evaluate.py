"""
evaluate.py

Load trained models and run a full evaluation on the test set.
Generates a comparison table across all three models.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model transformer
    python scripts/evaluate.py --save_report
"""

import argparse
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils.helpers import load_config, setup_logging
from src.utils.data_loader import get_dataset


def evaluate_model(model_name: str, model, test_texts, test_labels) -> dict:
    import time

    logger.info(f"Evaluating {model_name}...")

    start = time.perf_counter()
    pred_labels = model.predict(test_texts)
    latency = (time.perf_counter() - start) / len(test_texts) * 1000  # ms per sample

    acc = accuracy_score(test_labels, pred_labels)
    f1_macro = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(test_labels, pred_labels, average="weighted", zero_division=0)
    report = classification_report(test_labels, pred_labels, output_dict=True, zero_division=0)

    result = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "avg_latency_ms": round(latency, 3),
        "classification_report": report,
    }

    logger.info(f"  Accuracy   : {acc:.4f}")
    logger.info(f"  F1 (macro) : {f1_macro:.4f}")
    logger.info(f"  Latency    : {latency:.2f} ms/sample")

    return result


def print_comparison_table(results: list):
    print("\n" + "=" * 70)
    print(f"  {'MODEL':<22} {'ACCURACY':>10} {'F1 MACRO':>10} {'LATENCY':>12}")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['model']:<22} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} "
            f"{r['avg_latency_ms']:>10.1f} ms"
        )
    print("=" * 70)

    # improvement summary
    if len(results) >= 2:
        baseline_acc = results[0]["accuracy"]
        best_acc = max(r["accuracy"] for r in results)
        improvement = (1 - (1 - best_acc) / (1 - baseline_acc)) * 100 if baseline_acc < 1 else 0
        print(f"\n  Misclassification reduction (best vs baseline): {improvement:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifiers")
    parser.add_argument("--model", choices=["tfidf", "embedding", "transformer", "all"],
                        default="all")
    parser.add_argument("--save_report", action="store_true",
                        help="Save JSON report to logs/eval_report.json")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["logging"]["level"], config["logging"]["log_file"])

    _, _, test_texts, test_labels = get_dataset(config["data"])
    logger.info(f"Test set: {len(test_texts)} samples")

    all_results = []

    if args.model in ("tfidf", "all"):
        tfidf_path = "models/tfidf_classifier"
        if os.path.exists(tfidf_path):
            from src.models.tfidf_classifier import TfidfClassifier
            clf = TfidfClassifier()
            clf.load(tfidf_path)
            all_results.append(evaluate_model("TF-IDF + SVM", clf, test_texts, test_labels))
        else:
            logger.warning("TF-IDF model not found. Skipping.")

    if args.model in ("embedding", "all"):
        emb_path = "models/embedding_classifier"
        if os.path.exists(emb_path):
            from src.models.embedding_classifier import EmbeddingClassifier
            clf = EmbeddingClassifier()
            clf.load(emb_path)
            all_results.append(evaluate_model("Embedding + KNN", clf, test_texts, test_labels))
        else:
            logger.warning("Embedding model not found. Skipping.")

    if args.model in ("transformer", "all"):
        tf_path = config["transformer"]["save_path"]
        if os.path.exists(tf_path):
            from src.models.transformer_classifier import TransformerClassifier
            clf = TransformerClassifier(model_name=config["transformer"]["model_name"])
            clf.load(tf_path)
            all_results.append(evaluate_model("RoBERTa (fine-tuned)", clf, test_texts, test_labels))
        else:
            logger.warning("Transformer model not found. Skipping.")

    if not all_results:
        logger.error("No models found. Run scripts/train.py first.")
        return

    print_comparison_table(all_results)

    if args.save_report:
        os.makedirs("logs", exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": config["data"]["dataset"],
            "test_samples": len(test_texts),
            "results": all_results,
        }
        report_path = "logs/eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
