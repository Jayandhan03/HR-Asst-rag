from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

classifier_prompt = PromptTemplate.from_template("""
You are an intelligent classifier that determines whether a user's HR-related question requires looking up official documents or can be answered from general knowledge.

Question: {question}

Respond with only "yes" if the question requires referring to HR documents, policies, or handbook. Otherwise, respond with "no".
""")

llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

def should_use_rag(question: str) -> bool:
    response = llm.predict(classifier_prompt.format(question=question)).strip().lower()
    return "yes" in response
