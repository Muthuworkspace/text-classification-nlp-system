"""
api/main.py

FastAPI app — loads trained models directly from HuggingFace Hub.
Works on Render, Railway, or any cloud platform.
No local model files needed.
"""

import os
import re
import string
import unicodedata
import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

HF_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "Muthuworkspace/text-classification-nlp"
)

app_state = {}

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading TF-IDF model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download

        tfidf_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="tfidf_classifier/tfidf_vectorizer.pkl",
            repo_type="model"
        )
        svm_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="tfidf_classifier/svm_classifier.pkl",
            repo_type="model"
        )
        le_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="tfidf_classifier/label_encoder.pkl",
            repo_type="model"
        )

        with open(tfidf_path, "rb") as f:
            app_state["tfidf"] = pickle.load(f)
        with open(svm_path, "rb") as f:
            app_state["svm"] = pickle.load(f)
        with open(le_path, "rb") as f:
            app_state["le"] = pickle.load(f)

        logger.info("TF-IDF model loaded!")

    except Exception as e:
        logger.error(f"Model load failed: {e}")
        app_state["tfidf"] = None

    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Text Classification API",
    description="Classifies text into 10 categories using TF-IDF + SVM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str
    model_config = {"json_schema_extra": {
        "example": {
            "text": "NASA launches new satellite into orbit around Mars"
        }
    }}

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model_used: str

@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": app_state.get("tfidf") is not None,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    loaded = app_state.get("tfidf") is not None
    return {
        "status": "ok" if loaded else "model not loaded",
        "model_loaded": loaded,
        "classes": list(app_state["le"].classes_) if loaded else []
    }

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    if app_state.get("tfidf") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    import numpy as np
    cleaned = clean_text(text)
    vec     = app_state["tfidf"].transform([cleaned])
    pred    = app_state["svm"].predict(vec)[0]
    label   = app_state["le"].inverse_transform([pred])[0]
    scores  = app_state["svm"].decision_function(vec)[0]
    exp_s   = np.exp(scores - scores.max())
    conf    = round(float(exp_s[pred] / exp_s.sum()), 4)

    return ClassifyResponse(
        label=label,
        confidence=conf,
        model_used="tfidf+svm"
    )

@app.post("/classify/batch")
def classify_batch(texts: list[str]):
    if app_state.get("tfidf") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for text in texts[:20]:
        cleaned = clean_text(text)
        vec     = app_state["tfidf"].transform([cleaned])
        pred    = app_state["svm"].predict(vec)[0]
        label   = app_state["le"].inverse_transform([pred])[0]
        results.append({
            "text": text[:80],
            "label": label
        })
    return {"results": results, "total": len(results)}
