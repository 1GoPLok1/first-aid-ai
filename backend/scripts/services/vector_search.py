from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams
)
import numpy as np
import grpc
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class QdrantService:

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.collection_name = "medical_assistant"
        
        self.client = QdrantClient(
            host=host,
            port=port,
            timeout=30,
            grpc_port=6334,  
            prefer_grpc=True  
        )
        
    def create_collection_if_not_exists(self, vector_size: int = 384):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    hnsw_config={
                        "m": 16,          
                        "ef_construct": 100  
                    }
                ),
                optimizers_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000
                }
            )
            logger.info(f"Collection '{self.collection_name}' created")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Collection '{self.collection_name}' already exists")
            else:
                raise
    
    def search_vectors(self, query_vector: List[float], 
                       category: str = None, 
                       top_k: int = 5) -> List[Dict]:

        search_filter = None
        if category:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )
        
        search_params = SearchParams(
            hnsw_ef=128,  
            exact=False   
        )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            search_params=search_params,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]