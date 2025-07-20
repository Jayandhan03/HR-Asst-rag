from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant helping startup investors make better decisions.
Answer the question below using only the provided context.

Context:
{context}

Question:
{question}

Answer in a clear, concise, and professional tone. Avoid speculation.
""",
)
