from docx import Document
from datetime import datetime

def render_to_docx(query: str, response: str, filename: str = "output.docx") -> None:
    """
    Renders the query and response into a formatted .docx file.

    Args:
        query (str): The original user query.
        response (str): The LLM-generated answer.
        filename (str): Output filename.
    """
    doc = Document()
    doc.add_heading("Startup Investment Research Report", level=1)
    doc.add_paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Query: {query}")
    doc.add_heading("AI-Generated Answer", level=2)
    doc.add_paragraph(response)
    doc.save(filename)



# | Index Type         | Accuracy        | Speed                | Memory   | Use Case                          |
# | ------------------ | --------------- | -------------------- | -------- | --------------------------------- |
# | **FLAT**           | Highest (exact) | Slowest              | Highest  | Small datasets, highest precision |
# | **HNSW**           | Very good       | Fast                 | Moderate | General-purpose, large datasets   |
# | **Quantized (PQ)** | Slightly lower  | Fastest (low memory) | Lowest   | Huge datasets, memory constrained |
