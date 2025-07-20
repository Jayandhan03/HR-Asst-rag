from langchain.chains import LLMChain
from Prompt.prompt_template import prompt_template
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def generate_llm_output(query: str, reranked_docs: list) -> str:
    """
    Generate a response using reranked context and Groq-hosted LLM.

    Args:
        query (str): User's query.
        reranked_docs (list): List of LangChain Document objects.

    Returns:
        str: LLM-generated answer.
    """
    # Combine top-N documents into one context string
    context = "\n\n".join([doc.page_content for doc in reranked_docs])

    # Load Groq model
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Build chain using prompt template
    chain = LLMChain(prompt=prompt_template, llm=llm)
    response = chain.run({"context": context, "question": query})

    return response
