---
title: HR Assistant Bot
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.26.0"
app_file: app.py
pinned: false
# to run locally: python -m streamlit run app.py
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

ASCII Flowchart:

📄 PDF Upload & Streaming
│
▼
🧩 Semantic Chunking
│
▼
🔢 Embedding Generation (all-MiniLM-L6-v2)
│
▼
🗃️ Qdrant Cloud Vector DB (FLAT / HNSW / QUANTIZED)
│
▼
🎯 Dense Retrieval
│
▼
📚 BM25 Reranking
│
▼
🤖 LLM Response Generation
│
▼
💬 Conversational Memory
│
▼
📄 DOCX / Text Output

markdown
Copy code

**⚙️ Pipeline Breakdown**  

1. **PDF Streaming & Ingestion** – HR policy PDFs are dynamically streamed into the system, enabling incremental ingestion and continuous updates without downtime.  
2. **Semantic Chunking** – Documents are broken into meaningful, context-aware chunks.  
3. **Embedding Generation** – Each chunk is embedded using Hugging Face’s `all-MiniLM-L6-v2`.  
4. **Vector Storage** – Stored in Qdrant Cloud with three index types:  
   - ⚡ FLAT – high precision  
   - 🧭 HNSW – fast approximate nearest neighbor search  
   - 💾 Quantized – memory-efficient  
5. **Dense Retrieval** – Retrieves most relevant chunks based on semantic similarity.  
6. **BM25 Reranking** – Combines semantic + lexical relevance.  
7. **LLM Response Generation** – Generates accurate, human-like answers.  
8. **Conversational Memory** – Maintains context across chat turns.  
9. **Output Rendering** – Export answers to DOCX or text.

**🧪 Retriever Benchmarking Results**  
Dense Retriever provided the best trade-off between speed and semantic relevance.

---

## 📂 4. Folder Structure

chunking/ # Semantic chunking logic
data/ # HR policy PDFs
embedding/ # Embedding models
Final/ # Final runnable scripts
ingest/ # Incremental ingestion pipeline
interface/ # CLI / frontend setup (in progress)
llm/ # LLM interaction & prompt templates
Prompt/ # Prompt customization
render/ # DOCX response renderer
Reranker/ # BM25/MMR reranking
retrieval/ # Retriever logic (Qdrant)
Tracing/ # LangSmith/OpenTelemetry (observability)
utils/ # Common utilities (logging, config, etc.)
vectorstore/ # Qdrant index handling

yaml
Copy code

---

## ✅ Features

- 📥 Incremental PDF ingestion  
- ✂️ Semantic chunking + embedding  
- 🧠 Multi-index vector store (Flat, HNSW, IVF) using Qdrant  
- ⚖️ BM25/MMR-based reranking  
- 💬 LLM-based answer generation  
- 🧾 DOCX rendering of answers  
- 🧠 Prompt templating support  
- 📡 LangSmith integration  
- 🧠 Multi-turn memory (WIP)  
- 🌐 Streamlit interface (planned)  
- 🐳 Deployed in Huggingface spaces

---

## 🚀 How It Works

1. Ingest HR PDFs and split into semantic chunks  
2. Embed chunks using OpenAI or HuggingFace models  
3. Store them in Qdrant with vector indexing  
4. Retrieve top-k documents using similarity search  
5. Rerank results using BM25/MMR  
6. Generate final response using LLM + prompt template  
7. Export response to DOCX

---

## 💻 Usage

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run CLI
python app.py --query "Is my spouse covered under the company health insurance?"
🔍 Sample Output

Q: "How many casual leaves do employees get per year?"
A: "Yes, your legal spouse is eligible for coverage under our medical, dental, and vision plans. You will need to provide documentation to verify their eligibility."

🧰 Tech Stack
LangChain

Qdrant

OpenAI / Ollama / HuggingFace

BM25 / MMR

LangSmith

Python

🛠️ Planned Improvements
✅ Streamlit / Gradio UI

✅ Redis/SQLite-based chat memory

✅ Docker + cloud deployment

bash
Copy code
docker run --env-file .env -p 8501:8501 hr-assistant-bot:latest
✅ Slack/MS Teams integration

👤 Author
Jayandhan S — Passionate about building agentic GenAI systems and real-world AI assistants.

📜 License
MIT License