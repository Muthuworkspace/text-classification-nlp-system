"""
feature_extractor.py

Two strategies here:
1. TF-IDF sparse vectors (fast, interpretable, decent baseline)
2. Dense sentence embeddings via sentence-transformers (captures semantics)

The idea is to run both and compare — spoiler: embeddings win on most tasks.
"""

import os
import pickle
from typing import List, Union

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class TfidfFeatureExtractor:
    """
    Wrapper around sklearn TfidfVectorizer.
    Fits on training data, transforms any text to sparse vectors.
    """

    def __init__(self, max_features: int = 50000, ngram_range=(1, 2), sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,  # log(tf) instead of raw tf — helps with long docs
            strip_accents="unicode",
            analyzer="word",
            min_df=2,    # ignore terms that appear in less than 2 docs
            max_df=0.95, # ignore terms in more than 95% of docs (too common = useless)
        )
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "TfidfFeatureExtractor":
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vocabulary size: {vocab_size:,}")
        return self

    def transform(self, texts: List[str]):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"TF-IDF vectorizer saved to {path}")

    def load(self, path: str) -> "TfidfFeatureExtractor":
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._is_fitted = True
        logger.info(f"TF-IDF vectorizer loaded from {path}")
        return self


class DenseEmbeddingExtractor:
    """
    Dense sentence embeddings using sentence-transformers.
    Default model: all-MiniLM-L6-v2 (good balance of speed vs quality).
    
    These embeddings capture meaning much better than TF-IDF — 
    "car" and "automobile" will be close in vector space.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Encode text(s) to dense vectors.
        Returns numpy array of shape (n_samples, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.debug(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # quick test
    sample_texts = [
        "The stock market crashed today due to inflation concerns.",
        "NASA successfully launched a Mars rover.",
        "The football team won the championship.",
    ]

    print("--- TF-IDF ---")
    tfidf = TfidfFeatureExtractor(max_features=1000)
    X = tfidf.fit_transform(sample_texts)
    print(f"TF-IDF shape: {X.shape}")

    print("\n--- Dense Embeddings ---")
    embedder = DenseEmbeddingExtractor()
    vecs = embedder.encode(sample_texts, show_progress=False)
    print(f"Embedding shape: {vecs.shape}")
    # cosine sim between first two (should be low — different topics)
    cos_sim = np.dot(vecs[0], vecs[1])
    print(f"Cosine sim (finance vs space): {cos_sim:.4f}")
