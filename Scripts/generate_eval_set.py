import json
import argparse
from pathlib import Path
from random import sample
from typing import List, Dict
from langchain_core.documents import Document

DEFAULT_INPUT_PATH = "outputs/chunks.json"
DEFAULT_OUTPUT_PATH = "data/evaluation_set.json"

def load_chunks(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(**doc) for doc in data]

def generate_eval_pairs(docs: List[Document], k: int = 10) -> List[Dict[str, str]]:
    selected = sample(docs, k) if k < len(docs) else docs
    eval_pairs = []

    for i, doc in enumerate(selected):
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > 300:
            content = content[:300] + "..."
        eval_pairs.append({
            "query": f"What is chunk {i+1} about?",
            "expected_answer": content
        })

    return eval_pairs

def save_eval_set(pairs: List[Dict[str, str]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"✅ Evaluation set saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation set from document chunks.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to JSON with chunked documents.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Where to save evaluation JSON.")
    parser.add_argument("--num", type=int, default=10, help="How many evaluation samples to generate.")

    args = parser.parse_args()

    docs = load_chunks(args.input)
    eval_set = generate_eval_pairs(docs, args.num)
    save_eval_set(eval_set, args.output)

if __name__ == "__main__":
    main()
