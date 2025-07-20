import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.retriever import get_retriever

query = "What does ClearLoop do?"
retriever = get_retriever(index_type="hnsw", k=5)
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content[:200]}...\n")
