import os
import joblib
import chromadb
from pathlib import Path
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader
from src.embedder import TextEmbedder

app = FastAPI(title="Research Paper Classification & RAG System")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "models" / "knn_classifier.pkl"
CHROMA_PATH = str(BASE_DIR / "chroma_db")

INDEX_FILE = TEMPLATES_DIR / "index.html"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

print("-> Loading trained KNN classifier...")
knn_classifier = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

print("-> Initializing RoBERTa Embedder...")
# LAZY LOADING INSIDE THE ENDPOINT
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    return _embedder

@app.post("/predict")
def predict(text: str):
    embedder = get_embedder()
    vector = embedder.get_embedding(text)
    # ... rest of your prediction logic

print("-> Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="research_papers")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not INDEX_FILE.exists():
        return PlainTextResponse("Error: index.html not found.", status_code=500)
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(None), file: UploadFile = File(None)):
    if not INDEX_FILE.exists():
        return PlainTextResponse("Error: index.html not found.", status_code=500)

    extracted_text = ""

    # 1. Handle uploaded PDF file if provided
    if file and file.filename and file.filename.endswith(".pdf"):
        try:
            reader = PdfReader(file.file)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted_text += t + "\n"
        except Exception as e:
            return templates.TemplateResponse(request, "index.html", {
                "request": request,
                "error": f"Error reading uploaded PDF: {e}"
            })
    
    # 2. Otherwise fall back to pasted text box
    elif text:
        extracted_text = text

    cleaned_text = extracted_text.strip()
    if not cleaned_text:
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "error": "Please upload a valid PDF file or paste research text.",
            "input_text": text or ""
        })

    # Truncate to first 500 words for safe token limits
    truncated_text = " ".join(cleaned_text.split()[:500])

    # 3. Generate RoBERTa embedding
    query_embedding = embedder.get_roberta_embedding([truncated_text])
    if isinstance(query_embedding, tuple):
        query_embedding = query_embedding[0]

    # 4. Predict category using KNN
    predicted_label = knn_classifier.predict(query_embedding)[0] if knn_classifier else "Unknown"

    # 5. Calculate confidence percentage
    confidence = 0.0
    if knn_classifier and hasattr(knn_classifier, "predict_proba"):
        probs = knn_classifier.predict_proba(query_embedding)[0]
        classes = list(knn_classifier.classes_)
        if predicted_label in classes:
            confidence = float(probs[classes.index(predicted_label)]) * 100

    # 6. Retrieve RAG chunks from ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )
    
    rag_matches = []
    if results and 'documents' in results and results['documents']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            rag_matches.append({
                "source": meta.get("source", "Unknown"),
                "category": meta.get("category", "Unknown"),
                "snippet": doc[:300] + "..."
            })

    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "input_text": text or "",
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "rag_matches": rag_matches
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
