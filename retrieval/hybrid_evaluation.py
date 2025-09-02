import time
import json
from glob import glob
from typing import List, Dict
import sys
import os
from sentence_transformers.util import cos_sim
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import sys, os
# Add parent directory to import config and hybrid_retriever
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL
from retrieval.hybrid_retrieval import HybridRetriever

# -----------------------------
# Load all benchmark JSONs from folder
# -----------------------------
def load_all_benchmarks(folder_path: str) -> List[Dict[str, str]]:
    all_data = []
    json_files = glob(os.path.join(folder_path, "*.json"))
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

# -----------------------------
# Initialize embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Benchmark hybrid retriever
# -----------------------------
def benchmark_hybrid(
    hybrid: HybridRetriever,
    benchmark_data: List[Dict[str, str]],
    k: int = 10
):
    total_time = 0.0
    total_similarity = 0.0
    valid_queries = 0

    for i, item in enumerate(benchmark_data, 1):
        query = item.get("query") or item.get("question") or item.get("prompt")
        ground_truth = item.get("answer") or item.get("solution")

        if not query or not ground_truth:
            continue

        valid_queries += 1
        start_time = time.time()
        top_docs = hybrid.get_relevant_documents(query, k=k)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # Compute average cosine similarity over top k docs
        ground_emb = embedding_model.embed_documents([ground_truth])[0]
        similarities = []
        for doc in top_docs:
            doc_emb = embedding_model.embed_documents([doc.page_content])[0]
            similarities.append(cos_sim(doc_emb, ground_emb).item())

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        total_similarity += avg_sim

        print(f"[{i}/{len(benchmark_data)}] Query: {query}")
        print(f"⏱️  Time: {elapsed_time:.3f}s | Avg Cosine Similarity: {avg_sim:.4f}\n")

    if valid_queries == 0:
        print("No valid queries found in benchmark.")
        return

    print("="*25 + " HYBRID BENCHMARK SUMMARY " + "="*25)
    print(f"Total Queries:          {valid_queries}")
    print(f"Average Retrieval Time: {total_time / valid_queries:.3f} sec/query")
    print(f"Average Cosine Similarity (top {k} docs): {total_similarity / valid_queries:.4f}")
    print("="*70)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    sparse_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\data"  # PDFs for BM25

    benchmark_data = load_all_benchmarks(benchmark_folder)
    print(f"Loaded {len(benchmark_data)} benchmark queries.\n")

    # Initialize hybrid retriever
    hybrid = HybridRetriever(dense_index_type="hnsw", sparse_docs_folder=sparse_folder, k=10)

    # Run benchmark
    benchmark_hybrid(hybrid, benchmark_data, k=10)
