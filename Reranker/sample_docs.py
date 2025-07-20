import nltk
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
nltk.download("punkt")
from langchain_core.documents import Document
from Reranker.reranker import bm25_rerank
from dotenv import load_dotenv
load_dotenv()

query = "climate insurance for businesses"
docs = [
    Document(page_content="Climate insurance helps companies recover from climate-related damages."),
    Document(page_content="Machine learning is transforming the financial sector."),
    Document(page_content="Startups often seek venture capital funding for early-stage growth."),
    Document(page_content="Insurance policies are critical for disaster recovery."),
    Document(page_content="The impact of climate change is growing across industries.")
]

reranked = bm25_rerank(query, docs, top_n=3)
for i, doc in enumerate(reranked):
    print(f"\nRank {i+1}: {doc.page_content}")