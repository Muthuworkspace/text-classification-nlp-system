"""
test_preprocessing.py

Unit tests for the text cleaning and feature extraction modules.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.feature_extractor import TfidfFeatureExtractor


class TestTextCleaner:

    def setup_method(self):
        self.cleaner = TextCleaner(remove_stopwords=True, lemmatize=True)

    def test_basic_cleaning(self):
        result = self.cleaner.clean("Hello World! This is a TEST.")
        assert isinstance(result, str)
        assert result == result.lower()

    def test_url_removal(self):
        text = "Visit https://example.com for more info"
        result = self.cleaner.clean(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_email_removal(self):
        text = "Contact us at support@company.com"
        result = self.cleaner.clean(text)
        assert "@" not in result

    def test_html_removal(self):
        text = "Hello <b>World</b> <br/>"
        result = self.cleaner.clean(text)
        assert "<b>" not in result
        assert "<br/>" not in result

    def test_empty_string(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   ") == ""

    def test_none_like_handling(self):
        # should not crash on non-string
        result = self.cleaner.clean(123)
        assert result == ""

    def test_stopword_removal(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = self.cleaner.clean(text)
        tokens = result.split()
        # "the", "over" etc. should be removed
        assert "the" not in tokens

    def test_batch_cleaning(self):
        texts = ["Hello world!", "NASA launched a rocket.", ""]
        results = self.cleaner.clean_batch(texts)
        assert len(results) == 3
        assert isinstance(results[0], str)
        assert results[2] == ""

    def test_min_token_length(self):
        cleaner = TextCleaner(min_token_length=4)
        result = cleaner.clean("a big cat sat on a mat")
        tokens = result.split()
        assert all(len(t) >= 4 for t in tokens)

    def test_lemmatization(self):
        text = "running dogs are chasing cats"
        result = self.cleaner.clean(text)
        # lemmatizer should convert "running" → "run", "dogs" → "dog"
        assert "running" not in result or "run" in result


class TestTfidfFeatureExtractor:

    def setup_method(self):
        self.texts = [
            "machine learning algorithms for classification",
            "deep neural networks image recognition",
            "natural language processing text mining",
            "space exploration mars missions nasa",
            "stock market financial analysis trading",
        ]
        self.extractor = TfidfFeatureExtractor(max_features=500)

    def test_fit_transform_shape(self):
        X = self.extractor.fit_transform(self.texts)
        assert X.shape[0] == len(self.texts)
        assert X.shape[1] <= 500

    def test_transform_without_fit_raises(self):
        fresh = TfidfFeatureExtractor()
        with pytest.raises(RuntimeError):
            fresh.transform(self.texts)

    def test_transform_consistent_shape(self):
        self.extractor.fit(self.texts)
        X_train = self.extractor.transform(self.texts)
        X_new = self.extractor.transform(["a completely new sentence about nothing"])
        assert X_train.shape[1] == X_new.shape[1]

    def test_save_and_load(self, tmp_path):
        self.extractor.fit(self.texts)
        save_path = str(tmp_path / "tfidf.pkl")
        self.extractor.save(save_path)

        loaded = TfidfFeatureExtractor()
        loaded.load(save_path)

        X_original = self.extractor.transform(self.texts)
        X_loaded = loaded.transform(self.texts)

        # sparse matrices comparison
        diff = (X_original - X_loaded)
        assert diff.nnz == 0 or abs(diff).max() < 1e-9
