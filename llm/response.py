# In your llm/response.py file

import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# This function remains the same: it receives a pre-built chain
def get_llm_response(
    chat_chain: LLMChain,
    query: str,
    reranked_docs: List[Document]
) -> str:
    """
    Given a user query, reranked docs, and a persistent LLMChain instance,
    generate an LLM response. The chain manages its own memory.
    """
    context = "\n\n".join(doc.page_content.strip() for doc in reranked_docs)
    try:
        result = chat_chain.invoke({"context": context, "input": query})
        return result["output"].strip()
    except Exception as e:
        import traceback
        print("\n❌ Exception caught!")
        traceback.print_exc()
        return f"❌ LLM Error: {str(e)}"

# This factory function will be updated with a better prompt
def create_chat_chain() -> LLMChain:
    """Creates and returns a new instance of the LLMChain with memory."""
    load_dotenv()

    llm = ChatGroq(
        temperature=0.2,
        model_name="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output"
    )

    # === ⭐️ IMPORTANT PROMPT MODIFICATION ===
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an HR policy assistant for the company Resilience X.\n"
         "Your responsibilities:\n"
         "1. First, ALWAYS review the 'Chat history' to understand the context of the conversation.\n" # <-- ADDED INSTRUCTION
         "2. Carefully read and analyse ALL provided 'Retrieved document chunks' to find the answer.\n"
         "3. Merge scattered details from the documents and the chat history into one complete understanding.\n"
         "4. Transform your findings into a formal, end-user-ready response.\n"
         "5. If the information is not in the documents, state that clearly.\n"
         "6. Do not invent details. Only use the information provided.\n"
         "7. Speak as the HR department, not as an AI model."),
        # This part remains the same
        ("human", "Retrieved document chunks:\n{context}\n\n"
         "Chat history:\n{chat_history}\n\n"
         "User query: {input}\n\n"
         "Final Answer:")
    ])

    return LLMChain(
        llm=llm,
        prompt=custom_prompt,
        memory=memory,
        # Set verbose=True to see the final prompt in your console for debugging
        verbose=True,
        output_key="output"
    )