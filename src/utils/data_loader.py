"""
data_loader.py

Loads datasets for training and evaluation.
Supports:
  - 20 Newsgroups (sklearn built-in, great for testing)
  - Custom CSV (drop your own dataset in data/raw/)

Both return the same format: (texts, labels) tuples.
"""

from typing import Tuple, List, Optional

import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


# subset of 20newsgroups categories — using all 20 makes training slower
# pick a focused subset to show cleaner results in demos
DEFAULT_CATEGORIES = [
    "sci.space",
    "sci.med",
    "sci.electronics",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "talk.politics.misc",
    "talk.religion.misc",
    "rec.sport.hockey",
    "rec.autos",
    "soc.religion.christian",
]


def load_20newsgroups(
    categories: Optional[List[str]] = None,
    remove_headers: bool = True,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load 20 Newsgroups dataset.
    Returns: (train_texts, train_labels, test_texts, test_labels)
    """
    cats = categories or DEFAULT_CATEGORIES
    remove = ("headers", "footers", "quotes") if remove_headers else ()

    logger.info(f"Loading 20 Newsgroups ({len(cats)} categories)...")

    train_data = fetch_20newsgroups(
        subset="train",
        categories=cats,
        remove=remove,
        random_state=random_seed,
    )
    test_data = fetch_20newsgroups(
        subset="test",
        categories=cats,
        remove=remove,
        random_state=random_seed,
    )

    train_texts = train_data.data
    train_labels = [train_data.target_names[t] for t in train_data.target]
    test_texts = test_data.data
    test_labels = [test_data.target_names[t] for t in test_data.target]

    logger.info(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    logger.info(f"Classes: {train_data.target_names}")

    return train_texts, train_labels, test_texts, test_labels


def load_custom_csv(
    csv_path: str,
    text_column: str = "text",
    label_column: str = "label",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load a custom CSV dataset.
    CSV must have at least two columns: text and label.
    Returns: (train_texts, train_labels, test_texts, test_labels)
    """
    logger.info(f"Loading custom CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"CSV must have '{text_column}' and '{label_column}' columns. Found: {list(df.columns)}")

    # drop rows with missing values
    initial_len = len(df)
    df = df.dropna(subset=[text_column, label_column])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing values")

    texts = df[text_column].tolist()
    labels = df[label_column].astype(str).tolist()

    logger.info(f"Total samples: {len(texts)}")
    logger.info(f"Classes: {sorted(set(labels))}")

    # split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_seed, stratify=labels
    )

    logger.info(f"Train: {len(train_texts)} | Test: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels


def get_dataset(config: dict) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Unified loader — reads from config dict and dispatches to right loader.
    """
    dataset_name = config.get("dataset", "20newsgroups")

    if dataset_name == "20newsgroups":
        return load_20newsgroups(
            test_size=config.get("test_size", 0.2),
            random_seed=config.get("random_seed", 42),
        )
    elif dataset_name == "custom":
        return load_custom_csv(
            csv_path=config["custom_csv_path"],
            text_column=config.get("text_column", "text"),
            label_column=config.get("label_column", "label"),
            test_size=config.get("test_size", 0.2),
            random_seed=config.get("random_seed", 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use '20newsgroups' or 'custom'")


def create_sample_csv(output_path: str = "data/sample/sample_data.csv", n_per_class: int = 20):
    """
    Generate a tiny sample CSV for quick testing without downloading anything.
    Not meant for real training — just sanity checks.
    """
    import random

    categories = {
        "technology": [
            "Python 3.13 brings significant performance improvements and new syntax features",
            "Machine learning models are getting better at understanding human language",
            "Open source software powers most of the modern internet infrastructure",
            "GPU computing accelerates training of large neural networks",
            "Docker containers simplify deployment of microservices at scale",
        ],
        "sports": [
            "The team won the championship after a thrilling overtime victory",
            "The athlete broke the world record in the 100 meter sprint",
            "Cricket fans celebrated as India won the test series",
            "Football season kicks off with record-breaking attendance",
            "The marathon runner finished in under two hours for the first time",
        ],
        "science": [
            "Researchers discovered a new exoplanet in the habitable zone",
            "Scientists developed a new vaccine candidate for malaria",
            "The James Webb Space Telescope captured stunning images of distant galaxies",
            "CRISPR gene editing shows promise for treating inherited diseases",
            "New battery technology could double the range of electric vehicles",
        ],
    }

    rows = []
    for label, templates in categories.items():
        for _ in range(n_per_class):
            text = random.choice(templates) + f" (sample {random.randint(1000,9999)})"
            rows.append({"text": text, "label": label})

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Sample CSV written to {output_path} ({len(df)} rows)")
    return output_path
