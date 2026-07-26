<div align="center">

# 📄 Automated Text Classification & Information Extraction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Render](https://img.shields.io/badge/Deployed-Render-purple?style=flat-square)](https://render.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker)](https://docker.com)

**A production-grade NLP pipeline that classifies unstructured text and extracts structured information — deployed as a live REST API.**

### 🚀 [Live API](https://text-classification-nlp-system.onrender.com) &nbsp;|&nbsp; 📖 [Interactive Docs](https://text-classification-nlp-system.onrender.com/docs) &nbsp;|&nbsp; 🤗 [Models on HuggingFace](https://huggingface.co/Muthuworkspace/text-classification-nlp)

</div>

---

## 🎯 What This Project Does

Most organizations deal with thousands of unstructured text documents that need to be categorized and analyzed manually. This system automates that entirely:

- **Classifies** raw text into predefined categories automatically
- **Extracts** named entities, key phrases, and structured fields from documents
- **Searches** documents semantically — finds relevant content by meaning, not just keywords
- **Deployed** as a production REST API accessible via public URL

---

## 🏗️ The Engineering Story

Built in **three progressive stages** — each solving a limitation of the previous:

```
Stage 1 → TF-IDF + SVM
          Converts text to word frequency numbers
          Problem: "car" and "automobile" treated as completely different
          Accuracy: 75.59%

Stage 2 → Dense Embeddings + KNN
          Converts text to 384-dimensional semantic vectors
          "car" and "automobile" now map to similar vectors
          Accuracy: 75.86%

Stage 3 → Fine-tuned RoBERTa
          Reads full sentence context before classifying
          Understands meaning, not just word patterns
          Accuracy: 77.12%
          Misclassification reduced by 6.3% vs baseline
```

---

## 📊 Model Performance

| Model | Accuracy | Avg Latency |
|-------|----------|-------------|
| TF-IDF + SVM *(baseline)* | 75.59% | ~12ms |
| Dense Embeddings + KNN | 75.86% | ~45ms |
| **Fine-tuned RoBERTa** ✨ | **77.12%** | ~180ms |

> RoBERTa reduced misclassification by **6.3%** compared to TF-IDF baseline.

---

## 🚀 Live Demo

**API is live and running:**

🔗 **Base URL:** https://text-classification-nlp-system.onrender.com

📖 **Interactive Docs:** https://text-classification-nlp-system.onrender.com/docs

> ⚠️ Free tier — first request after inactivity takes ~30 seconds to wake up.

### Quick Test

```bash
curl -X POST https://text-classification-nlp-system.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new satellite into orbit around Mars"}'
```

Response:
```json
{
  "label": "sci.space",
  "confidence": 0.91,
  "model_used": "tfidf+svm"
}
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Preprocessing | NLTK, regex, unicodedata |
| Classical ML | scikit-learn (TF-IDF, SVM) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Transformers | HuggingFace (RoBERTa) |
| Vector Store | ChromaDB |
| API | FastAPI + Uvicorn |
| Model Storage | HuggingFace Hub |
| Deployment | Render |
| Containerization | Docker + Docker Compose |

---

## 📁 Project Structure

```
text-classification-system/
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py          # Regex, stopwords, lemmatization
│   │   └── feature_extractor.py     # TF-IDF + dense embeddings
│   ├── models/
│   │   ├── tfidf_classifier.py      # Baseline: TF-IDF + SVM
│   │   ├── embedding_classifier.py  # Dense vectors + KNN
│   │   └── transformer_classifier.py # Fine-tuned RoBERTa
│   ├── vector_store/
│   │   └── chroma_store.py          # ChromaDB indexing + search
│   ├── extraction/
│   │   └── information_extractor.py # NER, key phrases, regex
│   └── utils/
│       ├── helpers.py               # Config, logging, metrics
│       └── data_loader.py           # Dataset loading
├── api/
│   ├── main.py                      # FastAPI app entry point
│   └── routes/
│       ├── classify.py              # POST /classify
│       └── search.py                # POST /search
├── scripts/
│   ├── train.py                     # Train all models
│   ├── evaluate.py                  # Comparison metrics
│   └── index_documents.py           # Index into ChromaDB
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Baseline_TF-IDF_Models.ipynb
│   └── 03_Transformer_Finetuning.ipynb
├── tests/
├── data/sample/sample_data.csv
├── configs/config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## ⚙️ Setup and Run Locally

```bash
# Clone
git clone https://github.com/Muthuworkspace/text-classification-nlp-system.git
cd text-classification-nlp-system

# Install
python -m venv venv
venv\Scripts\activate       # Windows
python setup.py

# Train
python scripts/train.py --model all

# Index
python scripts/index_documents.py

# Run API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Model load status |
| `/classify` | POST | Classify single text |
| `/classify/batch` | POST | Classify multiple texts |

---

## 📂 Dataset

**20 Newsgroups** — 18,000+ real news articles across 10 categories.
No manual download needed — loads automatically during training.

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

---

## 📥 Trained Models

> ⚠️ Model files not included in repo (GitHub 100MB limit).

**🤗 TF-IDF Model on HuggingFace:**
👉 [Muthuworkspace/text-classification-nlp](https://huggingface.co/Muthuworkspace/text-classification-nlp)

**☁️ All Models on Google Drive (1.6GB):**
👉 [Download from Google Drive](https://drive.google.com/drive/folders/10vossy1FjTkXBXihTB2r9NXoxzzhcZLA?usp=sharing)

After downloading place folders:
```
models/tfidf_classifier/
models/embedding_classifier/
models/transformer_classifier/
data/chromadb/
```

Then run: `python demo.py`

---

## 👤 Author

**Muthukumaresan V**

[![GitHub](https://img.shields.io/badge/GitHub-Muthuworkspace-181717?style=flat-square&logo=github)](https://github.com/Muthuworkspace)

---

<div align="center">
<sub>Built with curiosity, debugged with patience.</sub>
</div>
