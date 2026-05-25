"""
classify.py — /classify endpoints

POST /classify          → classify text, return label + confidence + extracted info
POST /classify/batch    → classify multiple texts at once
GET  /classify/models   → list available models and their status
"""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from loguru import logger

router = APIRouter()


# --- Request / Response Models ---

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to classify")
    model: Literal["tfidf", "embedding", "transformer"] = Field(
        default="transformer",
        description="Which model to use"
    )
    include_extraction: bool = Field(
        default=True,
        description="Whether to run information extraction (NER, key phrases)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "text": "NASA announced a new mission to explore the outer planets of the solar system.",
            "model": "transformer",
            "include_extraction": True
        }
    }}


class EntityResult(BaseModel):
    text: str
    type: str
    description: Optional[str] = None


class ExtractionResult(BaseModel):
    entities: List[EntityResult]
    key_phrases: List[str]
    patterns: dict


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model_used: str
    extraction: Optional[ExtractionResult] = None


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)
    model: Literal["tfidf", "embedding", "transformer"] = "transformer"

    model_config = {"json_schema_extra": {
        "example": {
            "texts": [
                "The stock market dropped 3% after the Fed rate announcement.",
                "New research links coffee consumption to reduced risk of Parkinson's disease."
            ],
            "model": "transformer"
        }
    }}


class BatchClassifyResponse(BaseModel):
    results: List[dict]
    model_used: str
    total: int


# --- Endpoints ---

@router.post("", response_model=ClassifyResponse)
async def classify_text(request: Request, body: ClassifyRequest):
    """
    Classify a single text document.
    Supports three models: tfidf (fastest), embedding (balanced), transformer (best accuracy).
    """
    state = request.app.state.models

    # pick the right model
    model_key = body.model
    classifier = state.get(model_key)

    if classifier is None:
        # fallback to whatever is available
        for fallback in ["transformer", "embedding", "tfidf"]:
            if state.get(fallback) is not None:
                classifier = state[fallback]
                model_key = fallback
                logger.warning(f"Requested model '{body.model}' not loaded. Falling back to '{fallback}'")
                break

    if classifier is None:
        raise HTTPException(status_code=503, detail="No classifier models are loaded. Run scripts/train.py first.")

    try:
        result = classifier.predict_single(body.text)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

    response = ClassifyResponse(
        label=result["label"],
        confidence=result["confidence"],
        model_used=model_key,
    )

    # information extraction (optional)
    if body.include_extraction:
        extractor = state.get("extractor")
        if extractor:
            try:
                extracted = extractor.extract(body.text)
                response.extraction = ExtractionResult(
                    entities=[
                        EntityResult(
                            text=e["text"],
                            type=e["type"],
                            description=e.get("description"),
                        )
                        for e in extracted["entities"]
                    ],
                    key_phrases=extracted["key_phrases"],
                    patterns=extracted["patterns"],
                )
            except Exception as e:
                logger.warning(f"Extraction failed (non-fatal): {e}")

    return response


@router.post("/batch", response_model=BatchClassifyResponse)
async def classify_batch(request: Request, body: BatchClassifyRequest):
    """
    Classify multiple texts in one request.
    More efficient than calling /classify in a loop.
    """
    state = request.app.state.models
    classifier = state.get(body.model)

    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Model '{body.model}' is not loaded.")

    try:
        labels = classifier.predict(body.texts)
    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        {"text": text[:100] + "..." if len(text) > 100 else text, "label": label}
        for text, label in zip(body.texts, labels)
    ]

    return BatchClassifyResponse(
        results=results,
        model_used=body.model,
        total=len(results),
    )


@router.get("/models")
async def list_models(request: Request):
    """List available models and their load status."""
    state = request.app.state.models
    return {
        "models": {
            "tfidf": {
                "loaded": state.get("tfidf") is not None,
                "description": "TF-IDF + SVM — fast baseline",
            },
            "embedding": {
                "loaded": state.get("embedding") is not None,
                "description": "Dense embeddings + KNN — semantic understanding",
            },
            "transformer": {
                "loaded": state.get("transformer") is not None,
                "description": "Fine-tuned RoBERTa — highest accuracy",
            },
        }
    }
