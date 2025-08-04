from retrieval.retriever import get_retriever
from Reranker.reranker import bm25_rerank
from llm.response import get_llm_response
from llm.selector import should_use_rag
from llm.direct_answer import get_direct_answer

retriever = get_retriever(index_type="hnsw", k=10)

def handle_query(query):
    if should_use_rag(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        reranked_docs = bm25_rerank(query=query, documents=retrieved_docs, top_n=5)
        response, _ = get_llm_response(query=query, reranked_docs=reranked_docs)
    else:
        response = get_direct_answer(query)
    return response
