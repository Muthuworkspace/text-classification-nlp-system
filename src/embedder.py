import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-distilroberta-v1"):
        self.roberta_model = SentenceTransformer(model_name)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.is_tfidf_fitted = False

    def fit_tfidf(self, corpus: list[str]):
        self.tfidf_vectorizer.fit(corpus)
        self.is_tfidf_fitted = True

    def get_tfidf_embedding(self, texts: list[str]) -> np.ndarray:
        if not self.is_tfidf_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted first!")
        return self.tfidf_vectorizer.transform(texts).toarray()

    def get_roberta_embedding(self, texts: list[str]) -> np.ndarray:
        return self.roberta_model.encode(texts, convert_to_numpy=True)