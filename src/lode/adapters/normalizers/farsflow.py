from farsflow import JoinerFixer, Pipeline, SpaceCleaner
from farsflow import Normalizer as FFNormalizerStep

from lode.domain.interfaces import Normalizer


class FarsflowNormalizer(Normalizer):
    """
    Persian text normalizer backed by farsflow.

    Two pipelines:
    - index: full normalization for stored documents
    - query: light normalization for user queries
      (JoinerFixer intentionally omitted — preserves tsvector matching)
    """
    def __init__(self) -> None:
        self._index_pipeline = Pipeline([
            FFNormalizerStep(),
            JoinerFixer(),
            SpaceCleaner(),
        ])

        self._query_pipeline = Pipeline([
            FFNormalizerStep(),
            JoinerFixer(),
            SpaceCleaner(),
        ])


    async def normalize_index(self, text: str) -> str:
        """Process text for storage and indexing."""
        if not text:
            return ""
        return self._index_pipeline(text)


    async def normalize_query(self, text: str) -> str:
        """Light processing for user search queries."""
        if not text:
            return ""
        return self._query_pipeline(text)

