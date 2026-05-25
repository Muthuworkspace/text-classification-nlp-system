"""
setup.py

Run this ONCE after cloning the repo.
Downloads NLTK data and spaCy model needed for preprocessing.

Usage:
    python setup.py
"""

import subprocess
import sys


def run(cmd):
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"  [WARN] Command failed: {cmd}")


def main():
    print("\n" + "=" * 50)
    print("  Text Classification System — Setup")
    print("=" * 50 + "\n")

    # 1. Install pip packages
    print("[1/3] Installing Python packages...")
    run(f"{sys.executable} -m pip install -r requirements.txt -q")

    # 2. Download NLTK data
    print("\n[2/3] Downloading NLTK data...")
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    print("  NLTK data downloaded")

    # 3. Download spaCy model
    print("\n[3/3] Downloading spaCy model...")
    run(f"{sys.executable} -m spacy download en_core_web_sm")

    print("\n" + "=" * 50)
    print("  Setup complete! Next steps:")
    print("")
    print("  1. Train models:")
    print("     python scripts/train.py")
    print("")
    print("  2. Index documents:")
    print("     python scripts/index_documents.py")
    print("")
    print("  3. Start API:")
    print("     uvicorn api.main:app --reload")
    print("")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
