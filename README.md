---
title: HR Assistant Bot
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.26.0"
app_file: app.py
pinned: false
---

# 🤖 HR Assistant Bot

**HR Policy Q&A Assistant (RAG-powered)**

A Retrieval-Augmented Generation (RAG) system that intelligently answers employee HR policy questions using PDF documents. Built with LangChain and Qdrant, the assistant retrieves, reranks, and generates accurate answers from company policy files.

---

## 2. Introduction

**Brief Description**  
The HR Policy Q&A Assistant is an AI-powered chatbot designed to instantly answer any queries related to an organization’s HR policies. It ensures quick, accurate, and context-aware responses without requiring manual intervention from HR personnel.

**Problem Statement**  
In most organizations, employees often face delays when seeking clarification on HR policies, benefits, or procedures. Traditional support channels like emails or HR tickets are time-consuming, leading to inefficiencies and frustration.

**What the Bot Solves for Employees and HR**  
This assistant eliminates the need to wait for HR responses by providing instant, reliable answers to employee queries. It empowers employees with self-service access to HR information while significantly reducing the repetitive workload on HR teams.

---

## 🧠 3. Architecture Overview

**🔍 System Workflow**  

The HR Policy Q&A Assistant is built on a robust Retrieval-Augmented Generation (RAG) architecture that transforms static HR documents into an intelligent, conversational knowledge system.
code
Code
[📄 PDF Upload & Streaming]
            |
            v
  [🧩 Semantic Chunking]
            |
            v
  [🔢 Embedding Generation (all-MiniLM-L6-v2)]
            |
            v
  [🗃️ Qdrant Cloud Vector DB (FLAT / HNSW / QUANTIZED)]
            |
  (User Query Enters Here)
            |
            v
  [🎯 Dense Retrieval]
            |
            v
  [📚 BM25 Reranking]
            |
            v
  [🤖 LLM Response Generation]
            |
            v
  [💬 Conversational Memory]
            |
            v
  [💻 User Interface / 📄 DOCX Output]
code
Code
**⚙️ Pipeline Breakdown**  

1.  **PDF Streaming & Ingestion**  
    HR policy PDFs are dynamically streamed into the system, enabling incremental ingestion and continuous updates without downtime.

2.  **Semantic Chunking**  
    Documents are broken into meaningful, context-aware chunks, preserving relationships between ideas instead of arbitrary splits.

3.  **Embedding Generation**  
    Each chunk is embedded using Hugging Face’s `all-MiniLM-L6-v2` — a lightweight, high-performing model optimized for semantic similarity.

4.  **Vector Storage**  
    The embeddings are stored in Qdrant Cloud, indexed under three configurations:
    -   ⚡ **FLAT** – For precision and baseline accuracy
    -   🧭 **HNSW** – For high-speed approximate nearest neighbor search
    -   💾 **Quantized** – For efficient memory usage

5.  **Dense Retrieval**  
    User queries are embedded and compared against stored vectors to fetch the most relevant information — enabling contextually deep understanding rather than shallow keyword matches.

6.  **BM25 Reranking**  
    The top retrieved chunks are reranked with BM25, combining semantic and lexical relevance for balanced, high-precision results.

7.  **LLM Response Generation**  
    The refined chunks are passed to an LLM, which generates concise, accurate, and human-like answers tailored to HR-related queries.

8.  **Conversational Memory**  
    A memory layer maintains context across multiple turns — allowing employees and HR to have a natural, flowing chat experience.

9.  **Output Rendering**  
    The final answer is displayed in the chat and can be exported as a formatted DOCX report for record-keeping or official use.

**🧪 Retriever Benchmarking Results**  
Multiple retrieval methods — Dense, Sparse, and Hybrid — were tested extensively. Dense Retrieval emerged as the best-performing approach, offering both speed and contextual depth in HR-specific Q&A tasks.

---

## 📂 4. Folder Structure
.
├── chunking/ # Semantic chunking logic
├── data/ # HR policy PDFs
├── embedding/ # Embedding models
├── Final/ # Final runnable scripts
├── ingest/ # Incremental ingestion pipeline
├── interface/ # CLI / frontend setup (in progress)
├── llm/ # LLM interaction & prompt templates
├── Prompt/ # Prompt customization
├── render/ # DOCX response renderer
├── Reranker/ # BM25/MMR reranking
├── retrieval/ # Retriever logic (Qdrant)
├── Tracing/ # LangSmith/OpenTelemetry (observability)
├── utils/ # Common utilities (logging, config, etc.)
└── vectorstore/ # Qdrant index handling
code
Code
---

## ✅ Features

-   📥 **Incremental PDF ingestion**
-   ✂️ **Semantic chunking + embedding**
-   🧠 **Multi-index vector store (Flat, HNSW, IVF) using Qdrant**
-   ⚖️ **BM25/MMR-based reranking for relevance**
-   💬 **LLM-based direct answer generation**
-   🧾 **DOCX rendering of answers**
-   🧠 **Prompt templating support**
-   📡 **LangSmith integration**
-   🧠 **Multi-turn memory (WIP)**
-   🌐 **Streamlit interface (planned)**
-   🐳 **Deployed in Huggingface spaces**

---

## 🚀 How It Works

1.  Ingest HR PDFs and split them into semantically meaningful chunks.
2.  Embed the chunks using OpenAI or HuggingFace models.
3.  Store them in Qdrant with efficient vector indexing.
4.  Retrieve top-k documents using similarity search.
5.  Rerank results using BM25 or MMR.
6.  Use LLM with a templated prompt to generate the final response.
7.  Export the response to DOCX.

---

## 💻 Usage

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
Step 2: Run the application (example)
code
Bash
python app.py --query "Is my spouse covered under the company health insurance?"
Step 3: Run with Docker (Optional)
code
Bash
docker run --env-file .env -p 8501:8501 hr-assistant-bot:latest
🔍 Sample Output
Q: "How many casual leaves do employees get per year?"
A: "Employees are entitled to 12 days of casual leave per calendar year. These are credited at the beginning of the year and are meant to cover unforeseen personal needs."
🧰 Tech Stack
Orchestration: LangChain
Vector DB: Qdrant
LLMs: OpenAI / Ollama / HuggingFace
Reranking: BM25 / MMR
Observability: LangSmith
Core Language: Python
🛠️ Planned Improvements

Streamlit / Gradio UI

Redis/SQLite-based chat memory

Docker + cloud deployment

Slack/MS Teams integration
👤 Author
Jayandhan S — Passionate about building agentic GenAI systems and real-world AI assistants.
📜 License
MIT License```