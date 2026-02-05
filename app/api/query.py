from fastapi import APIRouter, HTTPException
from app.schemas.query import QueryRequest, QueryResponse
from rag.langchain.rag_pipeline import rag_pipeline

router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # Invoke LangChain pipeline
        result = rag_pipeline.invoke(request.user_question)

        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            status=result["status"],
            sources=result["sources"]
        )

    except Exception as e:
        # Safe error handling
        raise HTTPException(
            status_code=500,
            detail="Internal error while processing the query."
        )
