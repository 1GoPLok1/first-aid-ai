from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict

client = QdrantClient(host="localhost", port=6333)

def hybrid_search(query_vector: List[float], category: str, top_k: int = 5):
    search_filter = None
    if category and category != "all":
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category)
                )
            ]
        )

    try:
        results = client.search(
            collection_name="medical_assistant",
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True, 
            with_vectors=False, 
            score_threshold=0.6 
        )
        
        formatted_results = []
        for hit in results:
            formatted_results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "score": hit.score
            })
            
        return formatted_results

    except Exception as e:
        print(f"Error during search: {e}")
        return []