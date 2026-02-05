from typing import List, Dict
from pydantic import BaseModel

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser

from rag.langchain.retriever_runnable import retriever_runnable
from rag.langchain.context_builder import context_runnable
from genai.prompts.rag_prompt import RAG_PROMPT
from genai.llm.llm_runnable import llm
from genai.guardrails.guardrail_runnable import guardrail_runnable


# ------------------------------------------------------------------
# 1. Output schema (STRUCTURED RAG RESPONSE)
# ------------------------------------------------------------------

class RAGResponse(BaseModel):
    answer: str
    chunks: List[Dict]


parser = PydanticOutputParser(pydantic_object=RAGResponse)


# ------------------------------------------------------------------
# 2. Helper: normalize LLM output
# ------------------------------------------------------------------

def extract_answer(output):
    """
    Normalize LLM output to plain string.
    """
    if isinstance(output, AIMessage):
        return output.content
    return str(output)


# ------------------------------------------------------------------
# 3. RAG PIPELINE
# ------------------------------------------------------------------

rag_pipeline = (
    # Step 1: Accept user query
    RunnableParallel(
        query=RunnablePassthrough()
    )

    # Step 2: Retrieve relevant chunks
    | retriever_runnable

    # Step 3: Build context from retrieved chunks
    | context_runnable

    # Step 4: Prepare prompt inputs
    | RunnableLambda(
        lambda x: {
            "question": x["query"],
            "context": x["context"],
            "chunks": x["chunks"],
        }
    )

    # Step 5: Generate answer + pass chunks through
    | RunnableParallel(
        answer=RAG_PROMPT | llm | RunnableLambda(extract_answer),
        chunks=lambda x: x["chunks"],
    )

    # Step 6: Enforce structured output
    # | RunnableLambda(lambda x: RAGResponse(**x))

    # Optional (enable later)
    | guardrail_runnable
)
