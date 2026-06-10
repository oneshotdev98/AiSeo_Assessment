"""Shared spaCy pipeline, loaded once per process.

We use `en_core_web_sm`: the checks only need the tagger + dependency parser
(for subject/root-verb detection) and the sentencizer. The large model's word
vectors add nothing here — semantic similarity is handled by
sentence-transformers in the gap analyzer, which is a far stronger embedding
space than spaCy's static vectors.
"""

from functools import lru_cache

import spacy
from spacy.language import Language

SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' is not installed. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc
