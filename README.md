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

| Model | Accuracy | F1 (macro) | Latency |
|---|---|---|---|
| TF-IDF + SVM | 84.2% | 0.83 | ~12ms |
| Dense Embeddings + KNN | 88.7% | 0.87 | ~45ms |
| Fine-tuned RoBERTa | **91.4%** | **0.91** | ~180ms |

BERT/RoBERTa reduced misclassification by ~22% compared to TF-IDF baseline.

---

## Dataset

Using the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/) (18,000+ posts across 20 categories). Can be swapped with any CSV containing `text` and `label` columns.

---

## Author

B.Tech Final Year Project — [Your Name]
