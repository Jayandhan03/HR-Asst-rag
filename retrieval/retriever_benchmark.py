import time
import json
from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim
import sys
import os
from glob import glob

# Add the parent directory to the system path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL

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
# Initialize Qdrant client
# -----------------------------
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)

# -----------------------------
# Benchmark function
# -----------------------------
def benchmark_retriever(collection_name: str, benchmark_data: List[Dict[str, str]], k: int = 8):
    """
    Benchmarks the retriever for a given collection on response time and accuracy.

    Args:
        collection_name (str): The name of the Qdrant collection to test.
        benchmark_data (List[Dict[str, str]]): List of queries with corresponding ground-truth answers.
        k (int): The number of documents to retrieve for each query.
    """
    print(f"\n🚀 Benchmarking Collection: {collection_name}")
    print("-" * 80)
    
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    total_time = 0.0
    total_accuracy = 0.0
    valid_queries = 0

    for i, item in enumerate(benchmark_data, 1):
        query = item.get("query") or item.get("question") or item.get("prompt")
        ground_truth_answer = item.get("answer") or item.get("solution")
        
        if not query or not ground_truth_answer:
            continue

        valid_queries += 1
        print(f"\n({valid_queries}/{len(benchmark_data)}) Query: {query}")

        # --- 1. Measure Retrieval Time ---
        start_time = time.time()
        docs = vectorstore.similarity_search(query, k=k)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # --- 2. Measure Accuracy using Cosine Similarity ---
        retrieved_content = "\n".join([doc.page_content for doc in docs])
        similarity_score = 0.0
        if retrieved_content:
            embeddings = embedding_model.embed_documents([retrieved_content, ground_truth_answer])
            similarity_score = cos_sim(embeddings[0], embeddings[1]).item()
        total_accuracy += similarity_score

        # --- 3. Print Detailed Results ---
        print(f"⏱️  Time: {elapsed_time:.3f} sec")
        print(f"🎯 Accuracy (Cosine Similarity): {similarity_score:.4f}")
        
        if docs:
            print("\n📄 Retrieved Context:")
            for doc_idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"   Chunk {doc_idx} — Source: {source} | Page: {page}")
        else:
            print("❌ No results found")
        print("-" * 40)

    # --- 4. Final Summary ---
    if valid_queries == 0:
        print("No valid queries found in benchmark data.")
        return

    avg_time = total_time / valid_queries
    avg_accuracy = total_accuracy / valid_queries

    print("\n" + "="*25 + " BENCHMARK SUMMARY " + "="*25)
    print(f"Collection Tested:      {collection_name}")
    print(f"Total Queries:          {valid_queries}")
    print(f"Average Retrieval Time: {avg_time:.3f} sec/query")
    print(f"Average Accuracy Score: {avg_accuracy:.4f}")
    print("="*69 + "\n")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    benchmark_data = load_all_benchmarks(benchmark_folder)
    print(f"Loaded {len(benchmark_data)} total benchmark queries from '{benchmark_folder}'\n")

    if not QDRANT_COLLECTIONS:
        print("Error: QDRANT_COLLECTIONS not defined in config.py.")
    else:
        for collection in QDRANT_COLLECTIONS:
            benchmark_retriever(collection, benchmark_data)


# import time
# from typing import List, Dict
# from qdrant_client import QdrantClient
# from langchain_community.vectorstores import Qdrant
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sentence_transformers.util import cos_sim
# import numpy as np
# import sys
# import os

# # Add the parent directory to the system path to import config
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # Note: Ensure your config.py file contains the necessary QDRANT and EMBED_MODEL constants.
# from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_PORT, QDRANT_COLLECTIONS, GEMINI_API_KEY, EMBED_MODEL

# # Sample queries and their corresponding ground-truth answers for benchmarking
# BENCHMARK_DATA: Dict[str, str] = {
#     "When is salary credited to my bank account?": "Salary is credited to your bank account on the last working day of each month.",
#     "What is the purpose of the Employee Handbook?": "The Employee Handbook serves as a guide to the company's policies, procedures, and culture. It outlines the rights and responsibilities of both the employee and the employer.",
#     "What is the purpose of the Exit and Resignation Procedure?": "The Exit and Resignation Procedure ensures a smooth and professional transition for departing employees, covering aspects like notice periods, handover of responsibilities, and final settlements.",
#     "What is a grievance?": "A grievance is a formal complaint raised by an employee regarding a workplace issue, such as a violation of their rights, unfair treatment, or a breach of company policy.",
#     "Is my spouse covered under the company health insurance?": "Yes, your spouse can be covered under the company's group health insurance policy. You may need to enroll them during the designated enrollment period and may be required to contribute to the premium."
# }

# # Initialize embedding model
# # This model is used for both retrieval and similarity calculation for consistency.
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# # Initialize Qdrant client
# qdrant_client = QdrantClient(
#     url=QDRANT_HOST,
#     api_key=QDRANT_API_KEY,
# )

# def benchmark_retriever(collection_name: str, benchmark_data: Dict[str, str], k: int = 8):
#     """
#     Benchmarks the retriever for a given collection on response time and accuracy.

#     Args:
#         collection_name (str): The name of the Qdrant collection to test.
#         benchmark_data (Dict[str, str]): A dictionary with queries as keys and ground-truth answers as values.
#         k (int): The number of documents to retrieve for each query.
#     """
#     print(f"\n🚀 Benchmarking Collection: {collection_name}")
#     print("-" * 80)
    
#     vectorstore = Qdrant(
#         client=qdrant_client,
#         collection_name=collection_name,
#         embeddings=embedding_model
#     )

#     total_time = 0.0
#     total_accuracy = 0.0
#     queries = list(benchmark_data.keys())

#     for i, query in enumerate(queries, 1):
#         ground_truth_answer = benchmark_data[query]
#         print(f"\n({i}/{len(queries)}) Query: {query}")

#         # --- 1. Measure Retrieval Time ---
#         start_time = time.time()
#         docs = vectorstore.similarity_search(query, k=k)
#         elapsed_time = time.time() - start_time
#         total_time += elapsed_time

#         # --- 2. Measure Accuracy using Cosine Similarity ---
#         retrieved_content = "\n".join([doc.page_content for doc in docs])
        
#         similarity_score = 0.0
#         if retrieved_content:
#             # Generate embeddings for both the retrieved content and the ground-truth answer
#             embeddings = embedding_model.embed_documents([retrieved_content, ground_truth_answer])
#             # Calculate cosine similarity
#             similarity_score = cos_sim(embeddings[0], embeddings[1]).item()
        
#         total_accuracy += similarity_score

#         # --- 3. Print Detailed Per-Query Results ---
#         print(f"⏱️  Time: {elapsed_time:.3f} sec")
#         print(f"🎯 Accuracy (Cosine Similarity): {similarity_score:.4f}")
        
#         if docs:
#             print("\n📄 Retrieved Context:")
#             for doc_idx, doc in enumerate(docs, 1):
#                 source = doc.metadata.get("source", "Unknown")
#                 page = doc.metadata.get("page", "N/A")
#                 print(f"   Chunk {doc_idx} — Source: {source} | Page: {page}")
#         else:
#             print("❌ No results found")
#         print("-" * 40)

#     # --- 4. Calculate and Print Final Summary ---
#     avg_time = total_time / len(queries)
#     avg_accuracy = total_accuracy / len(queries)

#     print("\n" + "="*25 + " BENCHMARK SUMMARY " + "="*25)
#     print(f"Collection Tested:      {collection_name}")
#     print(f"Total Queries:          {len(queries)}")
#     print(f"Average Retrieval Time: {avg_time:.3f} sec/query")
#     print(f"Average Accuracy Score: {avg_accuracy:.4f}")
#     print("="*69 + "\n")

# if __name__ == "__main__":
#     # Ensure QDRANT_COLLECTIONS is defined in your config file
#     if not QDRANT_COLLECTIONS:
#         print("Error: QDRANT_COLLECTIONS not defined in config.py. Please add the collection names you want to test.")
#     else:
#         for collection in QDRANT_COLLECTIONS:
#             benchmark_retriever(collection, BENCHMARK_DATA)
