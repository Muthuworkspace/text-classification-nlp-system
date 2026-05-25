"""
chroma_store.py

Vector database layer using ChromaDB.
Stores document embeddings and enables semantic search via cosine similarity + KNN.

This is the part that makes "sub-second retrieval from millions of records" work.
ChromaDB handles the indexing and ANN (approximate nearest neighbor) search under the hood.
"""

import os
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer


class ChromaVectorStore:
    """
    Wrapper around ChromaDB for document storage + semantic search.
    
    Supports:
    - Adding documents with optional metadata
    - Querying by natural language (cosine similarity search)
    - Querying by vector (if you already have the embedding)
    - Filtering by metadata
    - Persistent storage (survives restarts)
    """

    def __init__(
        self,
        collection_name: str = "document_store",
        persist_directory: str = "data/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        os.makedirs(persist_directory, exist_ok=True)

        # persistent client — data saved to disk
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # load embedding model (same as the classifier uses — consistency matters)
        logger.info(f"Loading embedding model: {embedding_model}")
        self._embedder = SentenceTransformer(embedding_model)

        # get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine distance for similarity
        )

        existing = self.collection.count()
        logger.info(f"Collection '{collection_name}' loaded — {existing:,} documents indexed")

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Add a list of documents to the vector store.
        Texts are embedded and stored with optional metadata.
        Returns number of documents added.
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return 0

        # generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # default empty metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]

        total_added = 0

        # process in batches to avoid memory issues with large datasets
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_meta = metadatas[start:end]
            batch_ids = ids[start:end]

            # embed the batch
            embeddings = self._embedder.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_list=True,
            )

            self.collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            total_added += len(batch_texts)
            logger.debug(f"Indexed batch {start}–{end} ({total_added} total so far)")

        logger.info(f"Successfully indexed {total_added} documents. Total in store: {self.collection.count():,}")
        return total_added

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search: find top-k most similar documents to the query.
        Returns list of dicts with text, metadata, distance, and similarity score.
        """
        query_embedding = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_list=True
        )[0]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        results = self.collection.query(**kwargs)

        # reformat into something cleaner
        output = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # chromadb cosine distance: 0 = identical, 2 = opposite
            # convert to similarity: 1 - dist/2 keeps it in [0,1]
            similarity = round(1 - dist / 2, 4)
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": round(dist, 4),
                "similarity": similarity,
            })

        return output

    def search_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search using a pre-computed embedding vector."""
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({
                "text": doc,
                "metadata": meta,
                "similarity": round(1 - dist / 2, 4),
            })
        return output

    def delete_documents(self, ids: List[str]):
        """Remove documents by their IDs."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from collection")

    def get_document_count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Nuke everything in the collection. Use carefully."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' cleared")


if __name__ == "__main__":
    # quick demo
    store = ChromaVectorStore(persist_directory="/tmp/test_chroma")

    sample_docs = [
        "SpaceX successfully launches Falcon 9 rocket into orbit",
        "NASA announces new Mars mission for 2026",
        "Stock markets rally after Fed rate decision",
        "Apple reports record quarterly earnings",
        "New deep learning model beats human performance on NLP benchmark",
        "Python 3.13 released with major performance improvements",
    ]

    metas = [
        {"category": "space", "source": "news"},
        {"category": "space", "source": "news"},
        {"category": "finance", "source": "news"},
        {"category": "finance", "source": "news"},
        {"category": "tech", "source": "news"},
        {"category": "tech", "source": "news"},
    ]

    store.add_documents(sample_docs, metas)

    print("\n--- Search: 'rocket launch' ---")
    results = store.search("rocket launch", top_k=3)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['text']}")

    print("\n--- Search: 'stock market finance' ---")
    results = store.search("stock market finance", top_k=3)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['text']}")
