"""
search.py — /search endpoints

POST /search          → semantic search over indexed documents
POST /search/index    → index new documents into vector store
GET  /search/stats    → collection stats
"""

from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from loguru import logger

router = APIRouter()


# --- Request / Response Models ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filter on document metadata (e.g. {'category': 'sci.space'})"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "query": "space exploration missions to Mars",
            "top_k": 5,
            "metadata_filter": None
        }
    }}


class SearchResult(BaseModel):
    text: str
    similarity: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_indexed: int


class IndexRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)
    metadatas: Optional[List[Dict[str, Any]]] = None

    model_config = {"json_schema_extra": {
        "example": {
            "texts": [
                "Astronomers discovered a new black hole at the center of the Milky Way.",
                "Python 3.13 introduces a new JIT compiler for faster execution."
            ],
            "metadatas": [
                {"category": "science", "source": "arxiv"},
                {"category": "technology", "source": "blog"}
            ]
        }
    }}


class IndexResponse(BaseModel):
    indexed: int
    total_in_store: int
    message: str


# --- Endpoints ---

@router.post("", response_model=SearchResponse)
async def semantic_search(request: Request, body: SearchRequest):
    """
    Semantic search over the indexed document store.
    Returns top-k most similar documents to the query using cosine similarity.
    """
    vector_store = request.app.state.models.get("vector_store")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")

    doc_count = vector_store.get_document_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents indexed yet. Use POST /search/index to add documents."
        )

    actual_top_k = min(body.top_k, doc_count)

    try:
        raw_results = vector_store.search(
            query=body.query,
            top_k=actual_top_k,
            metadata_filter=body.metadata_filter,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    results = [
        SearchResult(
            text=r["text"],
            similarity=r["similarity"],
            metadata=r.get("metadata", {}),
        )
        for r in raw_results
    ]

    return SearchResponse(
        query=body.query,
        results=results,
        total_indexed=doc_count,
    )


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: Request, body: IndexRequest):
    """
    Index new documents into the vector store.
    Documents are embedded and stored for future semantic search.
    """
    vector_store = request.app.state.models.get("vector_store")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")

    # validate metadata length
    if body.metadatas and len(body.metadatas) != len(body.texts):
        raise HTTPException(
            status_code=400,
            detail=f"metadatas length ({len(body.metadatas)}) must match texts length ({len(body.texts)})"
        )

    try:
        n_indexed = vector_store.add_documents(
            texts=body.texts,
            metadatas=body.metadatas,
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

    total = vector_store.get_document_count()
    return IndexResponse(
        indexed=n_indexed,
        total_in_store=total,
        message=f"Successfully indexed {n_indexed} documents. Total in store: {total}",
    )


@router.get("/stats")
async def store_stats(request: Request):
    """Get statistics about the vector store."""
    vector_store = request.app.state.models.get("vector_store")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available")

    return {
        "collection_name": vector_store.collection_name,
        "document_count": vector_store.get_document_count(),
        "embedding_model": vector_store.embedding_model_name,
        "persist_directory": vector_store.persist_directory,
    }
