# Automated Text Classification and Information Extraction System

A production-grade NLP pipeline that classifies unstructured text and extracts structured information using a progression of techniques — from classical ML (TF-IDF) to Dense Embeddings and Transformer-based models (BERT/RoBERTa). Includes a Vector Search engine for sub-second semantic retrieval.

---

## What this project does

- Classifies raw text into predefined categories using ML + Transformer models
- Extracts named entities, key phrases, and structured fields from documents
- Stores document embeddings in a ChromaDB vector store for semantic search
- Exposes everything through a FastAPI REST API
- Containerized with Docker for easy deployment

---

## Tech Stack

| Layer | Tools |
|---|---|
| Preprocessing | spaCy, NLTK, regex |
| Classical ML | scikit-learn (TF-IDF, SVM, LogReg) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Transformers | HuggingFace (BERT, RoBERTa) |
| Vector Store | ChromaDB |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
text-classification-system/
│
├── data/
│   ├── raw/                    # Original unprocessed datasets
│   ├── processed/              # Cleaned + tokenized data
│   └── sample/                 # Small sample for quick testing
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py     # Regex cleaning, stopword removal, lemmatization
│   │   └── feature_extractor.py # TF-IDF, embeddings, BERT tokenization
│   │
│   ├── models/
│   │   ├── tfidf_classifier.py      # Baseline: TF-IDF + SVM/LogReg
│   │   ├── embedding_classifier.py  # Dense vectors + cosine similarity
│   │   └── transformer_classifier.py # Fine-tuned BERT/RoBERTa
│   │
│   ├── vector_store/
│   │   └── chroma_store.py     # ChromaDB indexing + KNN search
│   │
│   ├── extraction/
│   │   └── information_extractor.py # NER, key phrase extraction, regex patterns
│   │
│   └── utils/
│       ├── helpers.py          # Logging, config loading, metrics
│       └── data_loader.py      # Dataset loading (20newsgroups, custom CSV)
│
├── api/
│   ├── main.py                 # FastAPI app entry point
│   └── routes/
│       ├── classify.py         # POST /classify endpoint
│       └── search.py           # POST /search endpoint
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Baseline_TF-IDF_Models.ipynb
│   └── 03_Transformer_Finetuning.ipynb
│
├── scripts/
│   ├── train.py                # Train all models end-to-end
│   ├── evaluate.py             # Run evaluation metrics
│   └── index_documents.py      # Index docs into ChromaDB
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_classifier.py
│
├── configs/
│   └── config.yaml             # Central config for all hyperparams
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Setup and Run

### 1. Clone and install

```bash
git clone https://github.com/yourname/text-classification-system.git
cd text-classification-system

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### 3. Train the models

```bash
python scripts/train.py --model all --dataset 20newsgroups
```

### 4. Index documents into ChromaDB

```bash
python scripts/index_documents.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs auto-generated at: `http://localhost:8000/docs`

---

## Docker

```bash
docker-compose up --build
```

---

## API Endpoints

### `POST /classify`
Classify a piece of text.

```json
{
  "text": "NASA launches new satellite into orbit around Mars",
  "model": "transformer"
}
```

Response:
```json
{
  "label": "sci.space",
  "confidence": 0.94,
  "model_used": "roberta",
  "extraction": {
    "entities": [{"text": "NASA", "type": "ORG"}, {"text": "Mars", "type": "LOC"}],
    "key_phrases": ["satellite", "orbit", "Mars"]
  }
}
```

### `POST /search`
Semantic search over indexed documents.

```json
{
  "query": "space exploration missions",
  "top_k": 5
}
```

---

## Model Performance

| Model | Accuracy | 
|---|---|---|---|
| TF-IDF + SVM | 75.59% |
| Dense Embeddings + KNN | 75.86% |
| RoBERTa Accuracy | 77.12% |


BERT/RoBERTa reduced misclassification by ~6.3% compared to TF-IDF baseline.

---


## 📥 Trained Models

> ⚠️ Model files are not included in this repository.
> GitHub has a 100MB file size limit — trained model weights
> exceed 1.6GB total so they are hosted separately.

**Download pre-trained models from Google Drive:**

👉**[Download from Google Drive](https://drive.google.com/drive/folders/10vossy1FjTkXBXihTB2r9NXoxzzhcZLA?usp=sharing)**

After downloading extract and place folders like this:

trained_models/tfidf_classifier/        → models/tfidf_classifier/
trained_models/embedding_classifier/    → models/embedding_classifier/
trained_models/transformer_classifier/  → models/transformer_classifier/
trained_models/chromadb/                → data/chromadb/

Then verify everything works:
```bash
python demo.py
```

### Why models are not in GitHub

This is standard practice in ML projects. Large model weights
are stored separately from code — same approach used by
HuggingFace, OpenAI, and most open source ML repositories.
The training code is fully available in `scripts/train.py`
to reproduce the models from scratch.

### Reproduce From Scratch (No Download Needed)

If you prefer to train yourself instead of downloading:

```bash
# Step 1 - install packages
python setup.py

# Step 2 - train all models (~35 minutes on GPU)
python scripts/train.py --model all

# Step 3 - index documents into ChromaDB
python scripts/index_documents.py

# Step 4 - start API
uvicorn api.main:app --reload
```



## Dataset
## 📂 Dataset

This project uses the **20 Newsgroups dataset** — 18,000+ real news 
articles across 10 categories.

**No manual download needed.** It loads automatically when you run 
training:

```python
from sklearn.datasets import fetch_20newsgroups
```

### Categories Used
| Category | Topic |
|----------|-------|
| sci.space | Space & Astronomy |
| sci.med | Medical Science |
| sci.electronics | Electronics |
| comp.graphics | Computer Graphics |
| comp.os.ms-windows.misc | Windows OS |
| talk.politics.misc | Politics |
| talk.religion.misc | Religion |
| rec.sport.hockey | Hockey |
| rec.autos | Automobiles |
| soc.religion.christian | Christianity |

### data/ Folder Structure
---
data/
├── sample/
│   └── sample_data.csv     ← 20-row sample showing expected CSV format
├── raw/                    ← place your own dataset CSV here (optional)
└── processed/              ← auto-created during training

### Using Your Own Dataset

If you want to train on custom data instead of 20 Newsgroups:

1. Place your CSV inside `data/raw/` with two columns:
text,label
"your document text here","category_name"

2. Update `configs/config.yaml`:
```yaml
data:
  dataset: "custom"
  custom_csv_path: "data/raw/your_file.csv"
```

3. Run training normally:
```bash
python scripts/train.py
```


## Author

MUTHUKUMARESAN V

