import time
from typing import List
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_PORT, QDRANT_COLLECTIONS, GEMINI_API_KEY, EMBED_MODEL

# Sample queries for benchmarking
SAMPLE_QUERIES = [
    "What does ClearLoop do?",
    "Explain KoraPay’s product offering.",
    "How does ComplyX ensure regulatory compliance?",
    "Who is the target market for FlowFi?",
    "What is the vision of LendEasy?"
]

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)

def benchmark_retriever(collection_name: str, queries: List[str], k: int = 3):
    print(f"\n🚀 Benchmarking: {collection_name}")
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    total_time = 0
    for query in queries:
        start = time.time()
        docs = vectorstore.similarity_search(query, k=k)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"\n🔎 Query: {query}")
        print(f"⏱️ Time: {elapsed:.3f} sec")
        if docs:
            print(f"📄 Top Result Preview: {docs[0].page_content[:150]}...")
        else:
            print("❌ No result found")

    avg_time = total_time / len(queries)
    print(f"\n✅ Average time for {collection_name}: {avg_time:.3f} sec/query")

if __name__ == "__main__":
    for collection in QDRANT_COLLECTIONS:
        benchmark_retriever(collection, SAMPLE_QUERIES)
