import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from PyPDF2 import PdfReader
from src.embedder import TextEmbedder

# Paths
MODEL_DIR = "models"
DATA_PATH = "research_dataset/test_dataset.csv"
TEST_DIR = "research_dataset/test"

if not os.path.exists(DATA_PATH):
    print(f"Test dataset metadata not found at {DATA_PATH}.")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"Test folder not found at {TEST_DIR}.")
    exit()

# Load test dataset metadata to map filenames to true labels
test_df = pd.read_csv(DATA_PATH)

X_test_texts = []
y_test_true = []

print(f"-> Reading test PDF files from flat folder {TEST_DIR} using CSV metadata...")

for idx, row in test_df.iterrows():
    filename = row['filename']
    label = row['label']
    
    pdf_path = os.path.join(TEST_DIR, filename)
    
    if not os.path.exists(pdf_path):
        print(f"Warning: Test file {filename} not found in {TEST_DIR}, skipping...")
        continue
        
    try:
        reader = PdfReader(pdf_path)
        full_pdf_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_pdf_text += t + "\n"
                
        cleaned_text = full_pdf_text.strip()
        if cleaned_text:
            # Safely truncate to the first 500 words to fit model token limits
            words = cleaned_text.split()[:500]
            truncated_text = " ".join(words)
            
            X_test_texts.append(str(truncated_text))
            y_test_true.append(str(label))
            
    except Exception as e:
        print(f"Error reading test file {filename}: {e}")

print(f"-> Successfully processed and validated {len(X_test_texts)} test PDF files.")

# Load the RoBERTa-trained KNN classifier
knn_path = os.path.join(MODEL_DIR, "knn_classifier.pkl")
if not os.path.exists(knn_path):
    print("KNN model not found. Please run train.py first.")
    exit()

knn_classifier = joblib.load(knn_path)

# Generate RoBERTa dense embeddings for the test set
print("-> Generating RoBERTa dense embeddings for test evaluation...")
embedder = TextEmbedder()
X_test_embeddings = embedder.get_roberta_embedding(X_test_texts)
if isinstance(X_test_embeddings, tuple):
    X_test_embeddings = X_test_embeddings[0]

# Predict labels using KNN on RoBERTa embeddings
y_pred = knn_classifier.predict(X_test_embeddings)
if isinstance(y_pred, tuple):
    y_pred = y_pred[0]

# Calculate Accuracy & Metrics
accuracy = accuracy_score(y_test_true, y_pred)
print(f"\n========================================")
print(f" RoBERTa + KNN Test Accuracy: {accuracy * 100:.2f}%")
print(f"========================================\n")

print("Detailed Classification Report:")
print(classification_report(y_test_true, y_pred))