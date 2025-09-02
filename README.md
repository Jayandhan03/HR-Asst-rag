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


# HR Assistant Bot

This Space allows employees to query HR policies using an AI-powered assistant.
🧠 HR Policy Q&A Assistant (RAG-powered)

A Retrieval-Augmented Generation (RAG) system that intelligently answers employee HR policy questions using PDF documents. Built with LangChain and Qdrant, the assistant retrieves, reranks, and generates accurate answers from company policy files.

---

## 📂 Folder Structure

```

├── chunking/         # Semantic chunking logic
├── data/             # HR policy PDFs
├── embedding/        # Embedding models
├── Final/            # Final runnable scripts
├── ingest/           # Incremental ingestion pipeline
├── interface/        # CLI / frontend setup (in progress)
├── llm/              # LLM interaction & prompt templates
├── Prompt/           # Prompt customization
├── render/           # DOCX response renderer
├── Reranker/         # BM25/MMR reranking
├── retrieval/        # Retriever logic (Qdrant)
├── Tracing/          # LangSmith/OpenTelemetry (observability)
├── utils/            # Common utilities (logging, config, etc.)
├── vectorstore/      # Qdrant index handling

````

---

## ✅ Features

- 📥 **Incremental PDF ingestion**
- ✂️ **Semantic chunking + embedding**
- 🧠 **Multi-index vector store (Flat, HNSW, IVF) using Qdrant**
- ⚖️ **BM25/MMR-based reranking for relevance**
- 💬 **LLM-based direct answer generation**
- 🧾 **DOCX rendering of answers**
- 🧠 **Prompt templating support**
- 📡 **LangSmith integration**
- 🧠 **Multi-turn memory (WIP)**
- 🌐 **Streamlit interface (planned)**
- 🐳 **Dockerized deployment (in progress)**

---

## 🚀 How It Works

1. Ingest HR PDFs and split them into semantically meaningful chunks
2. Embed the chunks using OpenAI or HuggingFace models
3. Store them in Qdrant with efficient vector indexing
4. Retrieve top-k documents using similarity search
5. Rerank results using BM25 or MMR
6. Use LLM with templated prompt to generate final response
7. Export response to DOCX

---

## 💻 Usage

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run CLI
python Final/Final_Pipeline.py --query "Is my spouse covered under the company health insurance?"
````

---

## 🔍 Sample Output

**Q:** "How many casual leaves do employees get per year?"
**A:** "Yes, your legal spouse is eligible for coverage under our medical, dental, and vision plans. You will need to provide documentation to verify their eligibility."

---

## 🧰 Tech Stack

* **LangChain**
* **Qdrant**
* **OpenAI / Ollama / HuggingFace**
* **BM25 / MMR**
* **LangSmith**
* **Python**, **Docker**

---

## 🛠️ Planned Improvements

* ✅ Streamlit / Gradio UI
* ✅ Redis/SQLite-based chat memory
* ✅ Docker + cloud deployment DOCKER COMMAND - {"http://localhost:8501/",docker run --env-file .env -p 8501:8501 hr-assistant-bot:latest}
* ✅ Slack/MS Teams integration

---

## 👤 Author

**Jayandhan S** — Passionate about building agentic GenAI systems and real-world AI assistants.

---

## 📜 License

MIT License

```


