import os
import sys
from datetime import datetime
import io
import streamlit as st

# LangChain components needed
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain

# Your other project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrieval.retriever import get_retriever
from Reranker.reranker import bm25_rerank
from llm.selector import should_use_rag
from llm.direct_answer import get_direct_answer

# Simple function to format chat history for the prompt
def format_chat_history(chat_log: list) -> str:
    if not chat_log:
        return "No prior conversation history."
    return "\n".join(f"Human: {q}\nAssistant: {a}" for q, a in chat_log)

def get_chat_log_text(chat_log):
    if not chat_log:
        return "No conversation yet."
    return "\n".join(f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(chat_log))

# === 🚀 HR Assistant Main Loop ===
def streamlit_hr_assistant():
    st.set_page_config(page_title="HR Policy Assistant", layout="centered")
    st.title("🤖 HR Policy Assistant")
    st.markdown("Ask any question related to the HR policy document.")

    # === 🧠 INITIALIZE STATEFUL OBJECTS ONCE ===

    # 1. Initialize Conversation Log
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []

    # 2. Initialize Retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever(index_type="hnsw", k=10)

    # 3. Initialize LLM with Groq API key from environment
    if "llm" not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is missing. Please set it in your HF Space Secrets.")
        st.session_state.llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            api_key=groq_api_key
        )

    # 4. Initialize Prompt Template
    if "prompt" not in st.session_state:
        st.session_state.prompt = ChatPromptTemplate.from_template(
            """
            System: You are an HR policy assistant for Resilience X.
            Your primary task is to answer user queries based on the provided context and conversation history.
            1. Review the 'Conversation History' to understand context.
            2. Read the 'Context from Documents' to find relevant facts.
            3. Synthesize the information from history and documents.
            4. If documents do not contain the answer, suggest clarification from authorities. Do not make up information.
            5. Keep the answer compact and to the point.

            ---
            Conversation History:
            {chat_history}
            ---
            Context from Documents:
            {context}
            ---
            User's Current Query: {input}
            ---
            Final Answer:
            """
        )

    # Display conversation history
    for q, a in st.session_state.conversation_log:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # Handle User Input
    if user_input := st.chat_input("Type your HR-related question here..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                if should_use_rag(user_input):
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(user_input)
                    reranked_docs = bm25_rerank(query=user_input, documents=retrieved_docs, top_n=5)
                    context = "\n\n".join(doc.page_content.strip() for doc in reranked_docs)
                else:
                    context = "No documents were retrieved as the query was determined to be a general question."

                chat_chain = LLMChain(
                    llm=st.session_state.llm,
                    prompt=st.session_state.prompt,
                    verbose=True
                )

                chat_history_str = format_chat_history(st.session_state.conversation_log)
                result = chat_chain.invoke({
                    "chat_history": chat_history_str,
                    "context": context,
                    "input": user_input
                })
                response = result["text"]

            except Exception as e:
                response = f"An error occurred: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.conversation_log.append((user_input, response))

    # Download conversation log
    if st.session_state.conversation_log:
        chat_log_text = get_chat_log_text(st.session_state.conversation_log)
        chat_log_bytes = io.BytesIO(chat_log_text.encode("utf-8"))

        st.download_button(
            label="📥 Download Conversation Log",
            data=chat_log_bytes,
            file_name=f"hr_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_chat_log"
        )

if __name__ == "__main__":
    streamlit_hr_assistant()
