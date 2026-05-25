"""
text_cleaner.py

Handles all the dirty work before any model sees the text.
Regex cleaning, stopword removal, lemmatization — the stuff
nobody likes writing but everyone needs.
"""

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# make sure these are downloaded before running
# python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


class TextCleaner:
    """
    Cleans raw text through a configurable pipeline.
    Each step can be toggled via constructor args.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

    def clean(self, text: str) -> str:
        """
        Full cleaning pipeline on a single string.
        Returns cleaned, space-joined token string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self._normalize_unicode(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_html_tags(text)
        text = self._lowercase(text)
        text = self._remove_punctuation(text)
        text = self._remove_extra_whitespace(text)

        tokens = text.split()
        tokens = self._filter_tokens(tokens)

        return " ".join(tokens)

    def clean_batch(self, texts: list) -> list:
        """Clean a list of texts. Returns list of cleaned strings."""
        return [self.clean(t) for t in texts]

    # --- private helpers ---

    def _normalize_unicode(self, text: str) -> str:
        # converts weird unicode chars to closest ascii equivalent
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")

    def _remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", " ", text)

    def _remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def _remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def _lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_punctuation(self, text: str) -> str:
        # keep apostrophes? nah, strip everything
        return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    def _remove_extra_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _filter_tokens(self, tokens: list) -> list:
        filtered = []
        for tok in tokens:
            if len(tok) < self.min_token_length:
                continue
            if tok.isdigit():
                continue
            if self.remove_stopwords and tok in STOP_WORDS:
                continue
            if self.lemmatize:
                tok = LEMMATIZER.lemmatize(tok)
            filtered.append(tok)
        return filtered


# quick sanity check when running this file directly
if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Check out https://example.com! NASA launched <b>3 rockets</b> today. It's incredible!!!"
    print("Original:", sample)
    print("Cleaned: ", cleaner.clean(sample))
