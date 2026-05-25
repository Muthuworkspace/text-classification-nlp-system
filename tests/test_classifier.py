"""
test_classifier.py

End-to-end tests for TF-IDF classifier and embedding classifier.
Transformer tests are skipped by default (too slow for unit testing).

Run with: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tfidf_classifier import TfidfClassifier


# small synthetic dataset — enough to check the pipeline works
TRAIN_TEXTS = [
    "NASA plans a new mission to Jupiter next year",
    "SpaceX Falcon 9 rocket successfully enters orbit",
    "Astronomers discover a new exoplanet with water",
    "The space telescope captured stunning galaxy images",
    "Mars rover discovers ancient river bed on red planet",
    "The stock market crashed after the interest rate hike",
    "Apple reports record revenue in quarterly earnings",
    "Inflation hits highest level in four decades this month",
    "Federal Reserve raises rates to combat rising inflation",
    "Oil prices surge due to geopolitical tensions in Middle East",
    "Scientists develop new mRNA vaccine for influenza virus",
    "Hospital reports breakthrough in cancer treatment therapy",
    "New study links sleep deprivation to heart disease risk",
    "FDA approves new drug for treatment resistant depression",
    "Researchers find gene mutation linked to Alzheimer disease",
]

TRAIN_LABELS = [
    "space", "space", "space", "space", "space",
    "finance", "finance", "finance", "finance", "finance",
    "health", "health", "health", "health", "health",
]

TEST_TEXTS = [
    "New satellite launched into geostationary orbit around Earth",
    "Stock market rebounds after positive earnings reports",
    "Doctors find new treatment for autoimmune disease patients",
]

TEST_LABELS = ["space", "finance", "health"]


class TestTfidfClassifier:

    def setup_method(self):
        self.clf = TfidfClassifier(classifier_type="logreg", tfidf_max_features=5000)

    def test_train_and_predict(self):
        self.clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        preds = self.clf.predict(TEST_TEXTS)
        assert len(preds) == len(TEST_TEXTS)
        assert all(p in ["space", "finance", "health"] for p in preds)

    def test_predict_single(self):
        self.clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        result = self.clf.predict_single("NASA launched another rocket into space today")
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in ["space", "finance", "health"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_evaluate_returns_accuracy(self):
        self.clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        metrics = self.clf.evaluate(TEST_TEXTS, TEST_LABELS)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "classification_report" in metrics

    def test_predict_without_train_raises(self):
        fresh = TfidfClassifier()
        with pytest.raises(RuntimeError):
            fresh.predict(["some text"])

    def test_save_and_load(self, tmp_path):
        self.clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        original_preds = self.clf.predict(TEST_TEXTS)

        save_dir = str(tmp_path / "tfidf_model")
        self.clf.save(save_dir)

        loaded_clf = TfidfClassifier()
        loaded_clf.load(save_dir)
        loaded_preds = loaded_clf.predict(TEST_TEXTS)

        assert original_preds == loaded_preds

    def test_predict_proba_shape(self):
        self.clf.train(TRAIN_TEXTS, TRAIN_LABELS)
        proba = self.clf.predict_proba(TEST_TEXTS)
        assert proba.shape[0] == len(TEST_TEXTS)
        assert proba.shape[1] == 3  # 3 classes
        # rows should sum to ~1
        import numpy as np
        row_sums = proba.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in row_sums)

    def test_different_classifier_types(self):
        for clf_type in ["svm", "logreg"]:
            clf = TfidfClassifier(classifier_type=clf_type)
            clf.train(TRAIN_TEXTS, TRAIN_LABELS)
            preds = clf.predict(TEST_TEXTS)
            assert len(preds) == len(TEST_TEXTS)

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError):
            TfidfClassifier(classifier_type="unknown_model")
