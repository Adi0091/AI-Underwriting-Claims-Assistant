import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Read from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insurance-rag-index")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index(dimension: int = 384):
    """
    Create Pinecone index if it does not already exist.

    Args:
        dimension: Embedding vector dimension (must match embedding model)
    """
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",      # Pinecone managed infra
                region="us-east-1"
            )
        )
        print(f"Index '{INDEX_NAME}' created")
    else:
        print(f"Index '{INDEX_NAME}' already exists")


def get_index():
    """
    Get Pinecone index handle.
    """
    return pc.Index(INDEX_NAME)
