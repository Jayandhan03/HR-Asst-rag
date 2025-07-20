import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Reranker.reranker import bm25_rerank
from response import generate_llm_output
from render_to_docx import render_to_docx
from langchain_core.documents import Document

def run_pipeline(query: str, retrieved_docs: list[Document]):
    """
    Full pipeline: rerank → LLM generation → docx output.

    Args:
        query (str): User question.
        retrieved_docs (list[Document]): Raw documents from vector search.
    """
    print(f"[Pipeline] Received query: {query}")
    print(f"[Pipeline] Retrieved {len(retrieved_docs)} raw documents.")

    # Step 1: Rerank with BM25
    reranked_docs = bm25_rerank(query=query, documents=retrieved_docs, top_n=5)

    # Step 2: Generate output using LLM
    print(f"[Pipeline] Generating LLM output...")
    answer = generate_llm_output(query=query, reranked_docs=reranked_docs)

    # Step 3: Render to .docx file
    print(f"[Pipeline] Rendering to DOCX...")
    render_to_docx(query=query, response=answer, filename="llm_output.docx")

    print("[Pipeline] ✅ Done! Answer written to llm_output.docx")

# You can now import this run_pipeline() anywhere or run it directly
if __name__ == "__main__":
    # Example usage
    query = "climate insurance for businesses"
    from Reranker.sample_docs import docs # Import dummy docs or your retriever
    run_pipeline(query, docs)
