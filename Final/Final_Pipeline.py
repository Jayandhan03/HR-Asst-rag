import os
import sys
from datetime import datetime
import warnings
import logging

# Silence warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("langchain").setLevel(logging.CRITICAL)

# Extend system path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === 🧩 Imports ===
from retrieval.retriever import get_retriever
from Reranker.reranker import bm25_rerank
from render_to_docx import render_to_docx
from llm.response import get_llm_response
from llm.selector import should_use_rag  
from llm.direct_answer import get_direct_answer  

# === 🚀 HR Assistant Main Loop ===
def run_hr_assistant():
    print("📚 HR Policy Assistant Ready!")
    print("Type your query related to HR policy (type 'exit' to quit)\n")

    retriever = get_retriever(index_type="hnsw", k=10)
    conversation_log = []

    while True:
        query = input("👤 You: ").strip()
        if query.lower() in ["exit", "quit", "end"]:
            print("👋 Session ended. Take care!\n")

            if conversation_log:
                full_convo = "\n\n".join(
                    f"👤 You: {q}\n🤖 HR Assistant: {a}" for q, a in conversation_log
                )
                filename = f"HR_Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.docx"
                render_to_docx(query="HR Conversation", response=full_convo, filename=filename)
                print(f"📄 Full conversation saved to: {filename}\n")
            break

        try:
            # Step 0: Decide if RAG is needed
            if should_use_rag(query):
                # Step 1: Retrieve relevant documents
                retrieved_docs = retriever.get_relevant_documents(query)

                # Step 2: Rerank using BM25
                reranked_docs = bm25_rerank(query=query, documents=retrieved_docs, top_n=5)

                # Step 3: Generate response using retrieved context
                response, memory = get_llm_response(query=query, reranked_docs=reranked_docs)
            else:
                # Direct LLM response without context
                response = get_direct_answer(query)

            # Step 4: Show answer
            print(f"\n🤖 HR Assistant: {response}\n")

            # Step 5: Add to conversation log
            conversation_log.append((query, response))

        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

# Entry point
if __name__ == "__main__":
    run_hr_assistant()
