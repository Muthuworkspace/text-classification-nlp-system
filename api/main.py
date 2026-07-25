"""
api/main.py

FastAPI app — loads trained models directly from HuggingFace Hub.
Works on Render, Railway, or any cloud platform.
No local model files needed.
"""

import os
import json
import pickle
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ─────────────────────────────────────────────
# HuggingFace repo where models are stored
# ─────────────────────────────────────────────
HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "Muthuworkspace/text-classification-nlp"
)

# global model store
app_state = {}


def download_models_from_hf():
    """
    Download all model files from HuggingFace Hub into a temp directory.
    Called once on startup.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    logger.info(f"Downloading models from HuggingFace: {HF_MODEL_REPO}")

    # temp directory to store downloaded models
    tmp_dir = tempfile.mkdtemp()
    logger.info(f"Using temp directory: {tmp_dir}")

    # list all files in the repo
    all_files = list(list_repo_files(HF_MODEL_REPO))
    logger.info(f"Files in repo: {all_files}")

    downloaded = {}
    for filename in all_files:
        try:
            local_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=filename,
                local_dir=tmp_dir
            )
            downloaded[filename] = local_path
            logger.info(f"Downloaded: {filename}")
        except Exception as e:
            logger.warning(f"Could not download {filename}: {e}")

    return tmp_dir, downloaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""

    logger.info("Starting up — loading models from HuggingFace...")

    try:
        tmp_dir, downloaded = download_models_from_hf()

        # ── Load TF-IDF + SVM ──
        try:
            import pickle
            tfidf_path = os.path.join(tmp_dir, "tfidf_classifier", "tfidf_vectorizer.pkl")
            svm_path   = os.path.join(tmp_dir, "tfidf_classifier", "svm_classifier.pkl")
            le_path    = os.path.join(tmp_dir, "tfidf_classifier", "label_encoder.pkl")

            with open(tfidf_path, "rb") as f:
                app_state["tfidf"] = pickle.load(f)
            with open(svm_path, "rb") as f:
                app_state["svm"] = pickle.load(f)
            with open(le_path, "rb") as f:
                app_state["label_encoder"] = pickle.load(f)

            logger.info("TF-IDF + SVM loaded")
        except Exception as e:
            logger.error(f"TF-IDF load failed: {e}")
            app_state["tfidf"] = None

        # ── Load RoBERTa ──
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification
            )
            import torch

            tf_dir = os.path.join(tmp_dir, "transformer_classifier")

            app_state["tokenizer"] = AutoTokenizer.from_pretrained(tf_dir)
            app_state["rob_model"] = AutoModelForSequenceClassification.from_pretrained(tf_dir)
            app_state["rob_model"].eval()

            mappings_path = os.path.join(tf_dir, "label_mappings.json")
            with open(mappings_path) as f:
                mappings = json.load(f)
            app_state["id2label"] = {
                int(k): v for k, v in mappings["id2label"].items()
            }

            logger.info("RoBERTa loaded")
        except Exception as e:
            logger.error(f"RoBERTa load failed: {e}")
            app_state["rob_model"] = None

        logger.info("All models ready!")

    except Exception as e:
        logger.error(f"Startup failed: {e}")

    yield

    logger.info("Shutting down...")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Text Classification API",
    description="NLP pipeline — TF-IDF + SVM and fine-tuned RoBERTa classifier",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────
import re
import string
import unicodedata

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
from pydantic import BaseModel
from typing import Optional, Literal
from fastapi import HTTPException


class ClassifyRequest(BaseModel):
    text: str
    model: Literal["tfidf", "transformer"] = "transformer"

    model_config = {"json_schema_extra": {
        "example": {
            "text": "NASA launches new satellite into orbit around Mars",
            "model": "transformer"
        }
    }}


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model_used: str


@app.get("/", tags=["Health"])
def root():
    return {
        "status":       "running",
        "docs":         "/docs",
        "models_loaded": {
            "tfidf":       app_state.get("tfidf") is not None,
            "transformer": app_state.get("rob_model") is not None,
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "tfidf_loaded":       app_state.get("tfidf") is not None,
        "transformer_loaded": app_state.get("rob_model") is not None,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
def classify(request: ClassifyRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.model == "tfidf":
        if app_state.get("tfidf") is None:
            raise HTTPException(status_code=503, detail="TF-IDF model not loaded")

        cleaned = clean_text(text)
        vec     = app_state["tfidf"].transform([cleaned])
        pred    = app_state["svm"].predict(vec)[0]
        label   = app_state["label_encoder"].inverse_transform([pred])[0]
        scores  = app_state["svm"].decision_function(vec)[0]
        conf    = round(float(abs(max(scores)) / sum(abs(scores))), 4)

        return ClassifyResponse(label=label, confidence=conf, model_used="tfidf+svm")

    else:
        if app_state.get("rob_model") is None:
            raise HTTPException(status_code=503, detail="Transformer model not loaded")

        import torch
        device    = "cuda" if torch.cuda.is_available() else "cpu"
        rob_model = app_state["rob_model"].to(device)
        tokenizer = app_state["tokenizer"]

        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=256, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = rob_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1).squeeze()
            pred_id = probs.argmax().item()
            conf    = round(probs[pred_id].item(), 4)

        label = app_state["id2label"][pred_id]
        return ClassifyResponse(label=label, confidence=conf, model_used="roberta")


@app.post("/classify/batch", tags=["Classification"])
def classify_batch(texts: list[str], model: str = "tfidf"):
    if not texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    results = []
    for text in texts[:20]:  # max 20 at once
        try:
            req    = ClassifyRequest(text=text, model=model)
            result = classify(req)
            results.append({"text": text[:80], "label": result.label, "confidence": result.confidence})
        except Exception as e:
            results.append({"text": text[:80], "label": "error", "confidence": 0.0})

    return {"results": results, "total": len(results)}
