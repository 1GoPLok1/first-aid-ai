from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=1000)
    category: Optional[str] = Field(None, regex="^(pmp|zolh)$")
    session_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Как остановить кровотечение из раны?",
                "category": "pmp",
                "session_id": "session_123456",
                "top_k": 5
            }
        }

@router.post("/query")
async def process_query(request: QueryRequest, req: Request):

    start_time = datetime.now()
    
    logger.info(
        f"Received query: {json.dumps({
            'session_id': request.session_id,
            'query_length': len(request.query),
            'category': request.category
        })}"
    )
    
    try:
        from services.query_processor import QueryProcessor
        processor = QueryProcessor()
        result = await processor.process(request)
        
        # Логирование успешного ответа
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Query processed successfully in {processing_time:.3f}s, "
            f"found {len(result.context_fragments)} results"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))