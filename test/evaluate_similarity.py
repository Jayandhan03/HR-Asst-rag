import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
import argparse
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = 443  # default for cloud


def load_eval_set(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_accuracy(eval_set, retriever, k=5):
    correct = 0
    total = len(eval_set)

    for idx, item in enumerate(eval_set):
        question = item["question"]
        true_context = item["context"].strip()

        docs = retriever.get_relevant_documents(question)
        retrieved_texts = [doc.page_content.strip() for doc in docs]

        match_found = any(true_context in retrieved for retrieved in retrieved_texts)

        if match_found:
            correct += 1
        else:
            print(f"❌ Missed [Q{idx+1}]: {question[:60]}...")

    accuracy = correct / total
    return accuracy

def main(eval_json_path, collection_name):
    # 1. Load eval set
    eval_set = load_eval_set(eval_json_path)

    # 2. Load Gemini Embeddings
    
    embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Connect to Qdrant
    client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_fn,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Compute accuracy
    accuracy = compute_accuracy(eval_set, retriever, k=5)
    print(f"\n✅ Top-5 Similarity Search Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True, help="Path to eval_set.json")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    args = parser.parse_args()

    main(args.eval_json, args.collection)
