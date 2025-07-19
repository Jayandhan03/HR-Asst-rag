import json

def load_evaluation_set(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # returns list of {"query": ..., "expected_answer": ...}


def compute_retrieval_accuracy(retriever, queries_with_answers, k=3):
    correct = 0
    for query, expected_answer in queries_with_answers:
        results = retriever.similarity_search(query, k=k)
        retrieved_texts = [doc.page_content for doc in results]
        if any(expected_answer.lower() in text.lower() for text in retrieved_texts):
            correct += 1
    accuracy = correct / len(queries_with_answers)
    print(f"🎯 Retrieval Accuracy: {accuracy * 100:.2f}%")
    return accuracy
