import chromadb
from src.embedder import RobertaEmbedder

embedder = RobertaEmbedder()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")

def add_document_to_db(doc_id: str, text: str, metadata: dict):
    vector = embedder.get_embedding(text)
    collection.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata]
    )

def search_similar_documents(query: str, n_results: int = 3):
    query_vector = embedder.get_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    return results