"""
main.py

FastAPI application entry point.
Loads all models on startup and exposes them via REST endpoints.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import classify, search
from src.utils.helpers import load_config, setup_logging

config = load_config()
setup_logging(
    log_level=config["logging"]["level"],
    log_file=config["logging"]["log_file"],
)

# global model store — populated on startup
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models once at startup, clean up on shutdown.
    Using lifespan instead of @app.on_event (deprecated in newer FastAPI).
    """
    logger.info("Starting up — loading models...")

    # Load transformer model (or fall back to embedding classifier)
    model_save_path = config["transformer"]["save_path"]
    if os.path.exists(model_save_path):
        from src.models.transformer_classifier import TransformerClassifier
        transformer = TransformerClassifier(
            model_name=config["transformer"]["model_name"],
            max_seq_length=config["transformer"]["max_seq_length"],
            save_path=model_save_path,
        )
        transformer.load(model_save_path)
        app_state["transformer"] = transformer
        logger.info("Transformer model loaded")
    else:
        logger.warning(f"No transformer model found at {model_save_path}. Run scripts/train.py first.")
        app_state["transformer"] = None

    # Load TF-IDF classifier
    tfidf_path = "models/tfidf_classifier"
    if os.path.exists(tfidf_path):
        from src.models.tfidf_classifier import TfidfClassifier
        tfidf_clf = TfidfClassifier(classifier_type=config["tfidf"]["classifier"])
        tfidf_clf.load(tfidf_path)
        app_state["tfidf"] = tfidf_clf
        logger.info("TF-IDF classifier loaded")
    else:
        app_state["tfidf"] = None

    # Load embedding classifier
    emb_path = "models/embedding_classifier"
    if os.path.exists(emb_path):
        from src.models.embedding_classifier import EmbeddingClassifier
        emb_clf = EmbeddingClassifier(embedding_model=config["embedding"]["model_name"])
        emb_clf.load(emb_path)
        app_state["embedding"] = emb_clf
        logger.info("Embedding classifier loaded")
    else:
        app_state["embedding"] = None

    # Load vector store
    from src.vector_store.chroma_store import ChromaVectorStore
    vector_store = ChromaVectorStore(
        collection_name=config["vector_store"]["collection_name"],
        persist_directory=config["vector_store"]["persist_directory"],
        embedding_model=config["vector_store"]["embedding_model"],
    )
    app_state["vector_store"] = vector_store
    logger.info(f"Vector store loaded — {vector_store.get_document_count()} docs indexed")

    # Load information extractor
    from src.extraction.information_extractor import InformationExtractor
    extractor = InformationExtractor(spacy_model=config["extraction"]["spacy_model"])
    app_state["extractor"] = extractor
    logger.info("Information extractor loaded")

    logger.info("All components ready. API is live.")
    yield

    # cleanup (nothing much needed here)
    logger.info("Shutting down...")


app = FastAPI(
    title="Text Classification & Information Extraction API",
    description="NLP pipeline: classify text + extract entities + semantic search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# attach state to routers
app.state.models = app_state

app.include_router(classify.router, prefix="/classify", tags=["Classification"])
app.include_router(search.router, prefix="/search", tags=["Search"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    loaded = {k: (v is not None) for k, v in app_state.items()}
    return {"status": "ok", "models_loaded": loaded}
