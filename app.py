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
# from render_to_docx import render_to_docx  # Assuming this exists
from llm.selector import should_use_rag
from llm.direct_answer import get_direct_answer

# A simple function to format the chat history for the prompt
def format_chat_history(chat_log: list) -> str:
    """Formats the chat log into a string for the LLM prompt."""
    if not chat_log:
        return "No prior conversation history."
    
    history = ""
    for user_query, assistant_response in chat_log:
        history += f"Human: {user_query}\nAssistant: {assistant_response}\n"
    return history

def get_chat_log_text(chat_log):
    """Formats the entire conversation log into a plain text string for download."""
    if not chat_log:
        return "No conversation yet."
    
    lines = []
    for i, (q, a) in enumerate(chat_log, 1):
        lines.append(f"Q{i}: {q}\nA{i}: {a}\n")
    return "\n".join(lines)

# === 🚀 HR Assistant Main Loop ===
def streamlit_hr_assistant():
    st.set_page_config(page_title="HR Policy Assistant", layout="centered")
    st.title("🤖 HR Policy Assistant")
    st.markdown("Ask any question related to the HR policy document.")

    # === 🧠 INITIALIZE STATEFUL OBJECTS ONCE ===

    # 1. Initialize Conversation Log (This is now our ONLY memory)
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []
    
    # 2. Initialize Retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever(index_type="hnsw", k=10)

    # 3. Initialize LLM (no changes here)
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(
            temperature=0.2,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # 4. Initialize Prompt Template (no longer needs memory keys)
    if "prompt" not in st.session_state:
        st.session_state.prompt = ChatPromptTemplate.from_template(
            """
            System: You are an HR policy assistant for Resilience X.
            Your primary task is to answer user queries based on the provided context and conversation history.
            1.  First, ALWAYS review the 'Conversation History' to understand the context of the user's question.
            2.  Next, carefully read the 'Context from Documents' to find relevant facts.
            3.  Synthesize the information from the history and the documents to provide a complete, formal, and helpful response.
            4.  If the documents do not contain the answer, suggest to get clarified from responsible authorities. Do not make up information.
            5.  Make sure to mold the answer compact and to the point.

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
    
    # We no longer need to initialize ConversationBufferMemory

    # --- Display conversation history ---
    for q, a in st.session_state.conversation_log:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # --- Handle User Input ---
    if user_input := st.chat_input("Type your HR-related question here..."):
        # Display the user's message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Thinking..."):
            try:
                # Decide if we need to retrieve documents
                if should_use_rag(user_input):
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(user_input)
                    reranked_docs = bm25_rerank(query=user_input, documents=retrieved_docs, top_n=5)
                    context = "\n\n".join(doc.page_content.strip() for doc in reranked_docs)
                else:
                    context = "No documents were retrieved as the query was determined to be a general question."

                # Create the chain for this turn. Notice no memory object is passed.
                chat_chain = LLMChain(
                    llm=st.session_state.llm,
                    prompt=st.session_state.prompt,
                    verbose=True # Keep this on to see the final prompt in your terminal!
                )

                # Manually format the history from our own log
                # We pass all but the current query, as that is handled by the 'input' key
                chat_history_str = format_chat_history(st.session_state.conversation_log)

                # Invoke the chain with manually prepared history
                result = chat_chain.invoke({
                    "chat_history": chat_history_str,
                    "context": context,
                    "input": user_input
                })
                response = result["text"]

            except Exception as e:
                response = f"An error occurred: {str(e)}"

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Manually save the successful turn to our log
        st.session_state.conversation_log.append((user_input, response))

    # --- Download conversation log button ---
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
