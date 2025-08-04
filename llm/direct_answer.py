from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

direct_prompt = PromptTemplate.from_template("""
You are an experienced HR expert. Answer the following employee question based on general HR knowledge.

Question: {question}

Answer in a helpful, professional, and concise manner.
""")

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.5)

def get_direct_answer(question: str) -> str:
    return llm.predict(direct_prompt.format(question=question)).strip()
