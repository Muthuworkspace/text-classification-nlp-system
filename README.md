# 📄 Automated Text Classification & Information Extraction System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow?style=flat-square&logo=huggingface&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple?style=flat-square&logo=chromadb&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=flat-square&logo=docker&logoColor=white)

A production-grade NLP pipeline that classifies unstructured text and extracts structured information using dense transformer embeddings and RAG-powered vector search — deployed as a live REST API and fully containerized with Docker.

🚀 **FastAPI REST API** | 📖 **Interactive Docs** | 🐳 **Docker Ready**

---

## 🎯 What This Project Does
Organizations frequently deal with thousands of unstructured research papers, reports, and documents that require manual categorization. This system automates that entire lifecycle:
* **Classifies raw text** into predefined academic and technical domains automatically.
* **Leverages dense transformer embeddings** (RoBERTa) to capture deep contextual semantics rather than relying on sparse word counts.
* **Powers RAG-style semantic retrieval** backed by ChromaDB vector storage for lightning-fast similarity search across records.

---

## 📈 Performance Evolution & Results
* **Baseline Approach:** Initial implementation utilized TF-IDF sparse vectorization paired with a K-Nearest Neighbors (k-NN) classifier, establishing a solid baseline accuracy of **63%**.
* **Advanced Architecture:** Upgraded to **RoBERTa dense embeddings**, which captured contextual nuance effectively and reduced overall misclassification rates significantly.
* **Vector Database Integration:** Integrated **ChromaDB** for efficient chunking, embedding storage, and rapid semantic vector search.

---

## 🛠️ Tech Stack
* **Core Language:** Python 3.11+
* **NLP & Embeddings:** Hugging Face Transformers, Sentence-Transformers (RoBERTa)
* **Vector Store & Retrieval:** ChromaDB, Cosine Similarity
* **Machine Learning:** scikit-learn (k-NN classifier)
* **Backend & API:** FastAPI, Uvicorn
* **Deployment & Containerization:** Docker & Docker Compose

---

## 📂 Project Architecture
```text
text-classification-nlp-system/
│
├── src/                    # Core NLP modules, embedders, vector store logic
├── Dockerfile              # Production container configuration
├── docker-compose.yml      # Multi-container orchestration setup
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
