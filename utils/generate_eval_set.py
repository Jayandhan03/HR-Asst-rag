import argparse
import json
import random
import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from chunking.semantic_chunker import DocumentChunker
from ingest.loader import PDFStreamingLoader

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


import re

def extract_json_block(text: str) -> str:
    """Extracts the first JSON block from a string using regex."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON object found in response.")

def generate_qa(chunk: str) -> Dict[str, str]:
    client = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")
    prompt = f"""Based on the following passage, generate a clear and concise question and its answer:

Passage:
\"\"\"
{chunk}
\"\"\"

Return the response in JSON format like this:
{{
  "question": "...",
  "answer": "..."
}}
Only return the JSON. No explanation, no markdown.
"""

    response = client.invoke([{"role": "user", "content": prompt}])

    try:
        raw_text = response.content
        print("🧪 Raw response:", repr(raw_text))

        json_block = extract_json_block(raw_text)
        output = json.loads(json_block)

        return {
            "question": output["question"],
            "answer": output["answer"],
            "context": chunk,
        }

    except Exception as e:
        print("⚠️ Failed to parse response:")
        print(response.content)
        raise e




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True, help="Directory with PDF files to chunk")
    parser.add_argument("--output", default="eval_set.json", help="Path to save the eval JSON")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of chunks to sample for QA")
    args = parser.parse_args()

    # ✅ Instantiate components
    loader = PDFStreamingLoader(folder_path=args.pdf_dir)
    chunker = DocumentChunker()

    # 🔹 Load and chunk docs
    docs = loader.load_all()
    chunks = chunker.chunk_documents(docs)

    print(f"✅ Loaded {len(docs)} docs, chunked into {len(chunks)}. Sampling {args.num_samples}...")

    sampled_chunks = random.sample(chunks, args.num_samples)
    qa_pairs = []

    for idx, chunk in enumerate(sampled_chunks):
        print(f"🔎 Generating Q&A for chunk {idx + 1}/{args.num_samples}...")
        qa = generate_qa(chunk.page_content)
        qa_pairs.append(qa)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ Eval set saved to: {args.output}")


if __name__ == "__main__":
    main()
