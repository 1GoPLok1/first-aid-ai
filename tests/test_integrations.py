import pytest
from app.search_engine import SearchEngine
from app.qdrant_client import get_qdrant_client
from app.embeddings import EmbeddingModel


class TestIntegration:
    def setup_method(self):
        self.client = get_qdrant_client()
        self.embedding_model = EmbeddingModel("intfloat/multilingual-e5-small")
        self.search_engine = SearchEngine(self.client, self.embedding_model)

    def test_search_returns_results(self):
        results = self.search_engine.search("первая помощь при кровотечении")

        assert len(results) > 0

        first_result = results[0]
        assert "text" in first_result
        assert "score" in first_result
        assert "source" in first_result
        assert isinstance(first_result["score"], float)
        assert 0 <= first_result["score"] <= 1

    def test_filter_by_category(self):
        results = self.search_engine.search(
            "питание",
            category="nutrition",
            limit=3
        )

        for result in results:
            assert result.get("category") == "nutrition"

    def test_connection_to_qdrant(self):
        assert self.client is not None
        collections = self.client.get_collections()
        assert collections is not None