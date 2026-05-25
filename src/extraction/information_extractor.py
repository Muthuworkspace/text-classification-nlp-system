"""
information_extractor.py

Extracts structured information from unstructured text:
- Named entities (people, orgs, locations, dates, money, etc.) via spaCy
- Key phrases (noun chunks + TF-IDF ranked terms)
- Structured patterns via regex (emails, phone numbers, URLs, dates)

This is the "information extraction" part of the pipeline title.
"""

import re
from typing import List, Dict, Any, Optional

import spacy
from loguru import logger


# patterns for common structured fields
PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "url": re.compile(r"https?://[^\s]+|www\.[^\s]+"),
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"),
    "money": re.compile(r"\$[\d,]+(?:\.\d{2})?|\b[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|INR)\b"),
    "percentage": re.compile(r"\b\d+(?:\.\d+)?%"),
}


class InformationExtractor:
    """
    Extracts structured information from raw text using spaCy + regex patterns.
    
    Keeps it simple: spaCy handles the heavy NLP (NER, dependency parsing),
    regex handles structured patterns that NLP gets wrong anyway (emails, phones).
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", max_key_phrases: int = 10):
        self.max_key_phrases = max_key_phrases
        logger.info(f"Loading spaCy model: {spacy_model}")

        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
            raise

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Full extraction pipeline on a single text.
        Returns a dict with entities, key_phrases, and pattern matches.
        """
        if not text or not text.strip():
            return self._empty_result()

        doc = self.nlp(text)

        return {
            "entities": self._extract_entities(doc),
            "key_phrases": self._extract_key_phrases(doc),
            "patterns": self._extract_patterns(text),
            "sentences": [sent.text.strip() for sent in doc.sents],
            "noun_chunks": self._extract_noun_chunks(doc),
        }

    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract from multiple texts using spaCy's pipe (faster than looping)."""
        results = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            try:
                result = {
                    "entities": self._extract_entities(doc),
                    "key_phrases": self._extract_key_phrases(doc),
                    "noun_chunks": self._extract_noun_chunks(doc),
                }
                results.append(result)
            except Exception as e:
                logger.error(f"Extraction failed for doc: {e}")
                results.append(self._empty_result())
        return results

    def extract_entities_only(self, text: str) -> List[Dict[str, str]]:
        """Lightweight version — just NER, no other processing."""
        doc = self.nlp(text)
        return self._extract_entities(doc)

    # --- private methods ---

    def _extract_entities(self, doc) -> List[Dict[str, str]]:
        """
        Extract named entities. spaCy entity types:
        PERSON, ORG, GPE (geopolitical), LOC, DATE, TIME, MONEY, PRODUCT, EVENT, etc.
        """
        entities = []
        seen = set()  # deduplicate

        for ent in doc.ents:
            key = (ent.text.strip(), ent.label_)
            if key in seen:
                continue
            if len(ent.text.strip()) < 2:
                continue

            seen.add(key)
            entities.append({
                "text": ent.text.strip(),
                "type": ent.label_,
                "description": spacy.explain(ent.label_) or ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        return entities

    def _extract_key_phrases(self, doc) -> List[str]:
        """
        Key phrase extraction based on:
        1. Noun chunks (spaCy's noun phrase detector)
        2. Filtered by POS tags (nouns, proper nouns, adjectives)
        3. Ranked by position (earlier = more important, roughly)
        """
        phrases = []
        seen = set()

        for chunk in doc.noun_chunks:
            # filter out stopword-only chunks and very short ones
            root = chunk.root
            if root.is_stop or root.is_punct:
                continue
            text = chunk.text.strip().lower()
            if len(text) < 3 or text in seen:
                continue
            seen.add(text)
            phrases.append(chunk.text.strip())

        # also add standalone important tokens (nouns + proper nouns not in chunks)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                text = token.text.strip().lower()
                if len(text) > 2 and text not in seen:
                    seen.add(text)
                    phrases.append(token.text.strip())

        return phrases[: self.max_key_phrases]

    def _extract_noun_chunks(self, doc) -> List[str]:
        """Raw noun chunks from spaCy (less filtered than key phrases)."""
        return [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]

    def _extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Run all regex patterns and return matches per type."""
        result = {}
        for pattern_name, pattern in PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                result[pattern_name] = list(set(matches))  # deduplicate
        return result

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "entities": [],
            "key_phrases": [],
            "patterns": {},
            "sentences": [],
            "noun_chunks": [],
        }


if __name__ == "__main__":
    extractor = InformationExtractor()

    sample = """
    Elon Musk's SpaceX successfully launched a Falcon 9 rocket from Cape Canaveral on March 15, 2024.
    The mission cost approximately $67 million. Contact mission control at info@spacex.com or visit www.spacex.com.
    NASA Administrator Bill Nelson praised the achievement, calling it a milestone for commercial spaceflight.
    """

    result = extractor.extract(sample)

    print("=== Entities ===")
    for ent in result["entities"]:
        print(f"  [{ent['type']}] {ent['text']}")

    print("\n=== Key Phrases ===")
    print(", ".join(result["key_phrases"]))

    print("\n=== Patterns ===")
    for pat_type, matches in result["patterns"].items():
        print(f"  {pat_type}: {matches}")
