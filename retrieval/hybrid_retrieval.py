# hybrid_retriever_benchmark.py
import time
import json
import sys
import os
from glob import glob
from typing import List, Dict

import numpy as np
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim
from langchain_core.documents import Document

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL

# --- Load all benchmark JSONs ---
def load_all_benchmarks(folder_path: str) -> List[Dict[str, str]]:
    all_data = []
    for file in glob(os.path.join(folder_path, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

# --- Hybrid Retriever ---
class HybridRetriever:
    def __init__(
        self,
        dense_index_type: str = "hnsw",
        sparse_docs_folder: str = None,
        k: int = 20,
        dense_weight: float = 0.8,
        sparse_weight: float = 0.2,
    ):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Dense setup
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
        )
        qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

        collection_name = None
        for name in QDRANT_COLLECTIONS:
            if dense_index_type in name:
                collection_name = name
                break
        if not collection_name:
            raise ValueError(f"No Qdrant collection found for {dense_index_type}")

        dense_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding_model
        )
        self.dense_retriever = dense_store.as_retriever(search_kwargs={"k": k})

        # Sparse setup
        from bm25_retriever import BM25Retriever, load_pdfs_from_folder
        if sparse_docs_folder:
            all_docs = load_pdfs_from_folder(sparse_docs_folder)
            self.sparse_retriever = BM25Retriever(all_docs)
        else:
            self.sparse_retriever = None

    def get_relevant_documents(self, query: str, k: int = None) -> List[Document]:
        k = k or self.k

        # Dense retrieval
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        for doc in dense_docs:
            doc.metadata["dense_score"] = max(doc.metadata.get("score", 0.0), 0.01)

        # Sparse retrieval
        sparse_docs = []
        if self.sparse_retriever:
            sparse_docs = self.sparse_retriever.get_relevant_documents(query, k=k)
            for doc in sparse_docs:
                doc.metadata["bm25_score"] = max(doc.metadata.get("bm25_score", 0.0), 0.01)

        # Combine results aggressively
        combined = {}
        for doc in dense_docs:
            combined[doc.page_content] = {"doc": doc, "dense": doc.metadata.get("dense_score", 0.01), "sparse": 0.01}
        for doc in sparse_docs:
            if doc.page_content in combined:
                combined[doc.page_content]["sparse"] = doc.metadata.get("bm25_score", 0.01)
            else:
                combined[doc.page_content] = {"doc": doc, "dense": 0.01, "sparse": doc.metadata.get("bm25_score", 0.01)}

        hybrid_docs = []
        for entry in combined.values():
            score = self.dense_weight * entry["dense"] + self.sparse_weight * entry["sparse"]
            entry["doc"].metadata["hybrid_score"] = score
            hybrid_docs.append(entry["doc"])

        hybrid_docs = sorted(hybrid_docs, key=lambda d: d.metadata.get("hybrid_score", 0), reverse=True)
        return hybrid_docs[:k]

# --- Benchmark ---
def benchmark_hybrid(hybrid: HybridRetriever, benchmark_data: List[Dict[str, str]], k: int = 20):
    total_time = 0.0
    total_similarity = 0.0
    valid_queries = 0

    for item in benchmark_data:
        query = item.get("query") or item.get("question") or item.get("prompt")
        answer = item.get("answer") or item.get("solution")
        if not query or not answer:
            continue

        valid_queries += 1
        start = time.time()
        docs = hybrid.get_relevant_documents(query, k=k)
        total_time += time.time() - start

        # Average cosine similarity across top-k docs
        ground_emb = hybrid.embedding_model.embed_documents([answer])[0]
        similarities = []
        for doc in docs:
            doc_emb = hybrid.embedding_model.embed_documents([doc.page_content])[0]
            sim = cos_sim(doc_emb, ground_emb).item()
            similarities.append(sim)
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        total_similarity += avg_sim

    avg_time = total_time / valid_queries if valid_queries else 0.0
    avg_cos_sim = total_similarity / valid_queries if valid_queries else 0.0

    print(f"\n========================= HYBRID BENCHMARK SUMMARY =========================")
    print(f"Total Queries:          {valid_queries}")
    print(f"Average Retrieval Time: {avg_time:.3f} sec/query")
    print(f"Average Cosine Similarity (top {k} docs): {avg_cos_sim:.4f}")
    print(f"{'='*70}\n")

# --- Main ---
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    benchmark_data = load_all_benchmarks(benchmark_folder)

    folder_path = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\data"
    hybrid = HybridRetriever(dense_index_type="hnsw", sparse_docs_folder=folder_path, k=20)

    benchmark_hybrid(hybrid, benchmark_data, k=20)
