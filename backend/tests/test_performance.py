import pytest
from app.search_engine import SearchEngine


@pytest.mark.benchmark
def test_search_performance(benchmark):
    engine = SearchEngine()

    result = benchmark(
        engine.search,
        "первая помощь при сердечном приступе",
        limit=5
    )

    assert len(result) > 0