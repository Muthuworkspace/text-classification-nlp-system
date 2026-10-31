import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from PyPDF2 import PdfReader
from src.embedder import TextEmbedder

# Define a robust text chunking helper
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = "research_dataset/train_dataset.csv"
TRAIN_DIR = "research_dataset/train"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset metadata not found at {DATA_PATH}.")
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found at {TRAIN_DIR}.")

# Load CSV to map filenames to their labels
df = pd.read_csv(DATA_PATH)

X_train_texts = []
y_train = []
all_chunks = []
all_chunk_metadatas = []
all_chunk_ids = []

print(f"-> Reading PDF files from flat folder {TRAIN_DIR} using CSV metadata...")

for idx, row in df.iterrows():
    filename = row['filename']
    label = row['label']
    
    # Path to the PDF directly inside the flat train folder
    pdf_path = os.path.join(TRAIN_DIR, filename)
    
    if not os.path.exists(pdf_path):
        print(f"Warning: {filename} not found in {TRAIN_DIR}, skipping...")
        continue
        
    try:
        reader = PdfReader(pdf_path)
        full_pdf_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_pdf_text += t + "\n"
                
        if not full_pdf_text.strip():
            continue
            
        # Store for KNN classification
        X_train_texts.append(full_pdf_text)
        y_train.append(label)
        
        # Chunk text for ChromaDB RAG search
        chunks = chunk_text(full_pdf_text)
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_chunk_metadatas.append({"source": filename, "category": label})
            all_chunk_ids.append(f"{filename}_chunk_{c_idx}")
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

print(f"-> Successfully processed {len(X_train_texts)} PDF files.")

# Generate RoBERTa dense embeddings for KNN classification
print("-> Generating RoBERTa dense embeddings for training samples...")
embedder = TextEmbedder()
X_train_embeddings = embedder.get_roberta_embedding(X_train_texts)

# Train KNN Classifier with distance weights and K=3 for higher precision
print("-> Training optimized KNN classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine')
knn_classifier.fit(X_train_embeddings, y_train)

# Save the trained model
knn_model_path = os.path.join(MODEL_DIR, "knn_classifier.pkl")
joblib.dump(knn_classifier, knn_model_path)
print(f"-> RoBERTa-backed KNN Model saved successfully to {knn_model_path}")

# Ingest chunks into ChromaDB for RAG search
print("-> Ingesting PDF chunks into ChromaDB vector store...")
import chromadb
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")

if all_chunks:
    chunk_embeddings = embedder.get_roberta_embedding(all_chunks).tolist()
    collection.add(
        documents=all_chunks,
        embeddings=chunk_embeddings,
        metadatas=all_chunk_metadatas,
        ids=all_chunk_ids
    )
    print(f"-> Indexed {len(all_chunks)} PDF chunks into ChromaDB using RoBERTa!")

print("=== Flat Folder Training & Ingestion Complete ===")