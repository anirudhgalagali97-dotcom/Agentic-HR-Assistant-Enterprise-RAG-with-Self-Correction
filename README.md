# 🤖 Agentic HR Assistant: Enterprise RAG with Self-Correction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-orange)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-black)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)

## 📌 Overview

Moving beyond "Naive RAG," this project implements a highly advanced **Agentic Retrieval-Augmented Generation (RAG)** pipeline. Designed specifically for complex Enterprise HR policies, this assistant doesn't just blindly retrieve and generate—it analyzes intent, evaluates its own retrieved context, corrects hallucinations, and autonomously routes to live web searches when local knowledge falls short.

## ✨ Key Features

* **Agentic Orchestration:** Built with **LangGraph** to manage complex, cyclical conversational states and memory checkpointers.
* **Intelligent Routing & Guardrails:** The system evaluates document relevance before generating an answer. If local documents score poorly, the agent autonomously falls back to a **DuckDuckGo Web Search**.
* **Hybrid Search Retrieval:** Combines semantic vector search (**ChromaDB**) with keyword-based search (**BM25**) to ensure high recall for specialized HR terminology.
* **Self-Correction (Hallucination Checker):** A final LLM evaluation node ensures the generated response is strictly grounded in the retrieved context.
* **100% Local Execution:** Powered entirely by local LLMs via **Ollama (Llama 3.2)**, ensuring absolute data privacy for sensitive HR documents.

## 🏗️ System Architecture

The agent follows a stateful directed graph:

1. **Analyze Query:** Extracts intent, metadata filters, and identifies knowledge gaps.
2. **Retrieve:** Executes Hybrid Search (Vector + BM25) across chunked HR PDFs.
3. **Grade Relevance:** Evaluates chunks. If relevant -> Generate. If irrelevant -> Web Search.
4. **Generate Answer:** Synthesizes the final response with citations.
5. **Check Hallucination:** Final guardrail to ensure factual grounding.

## 💻 Tech Stack

* **Framework:** LangChain & LangGraph (Latest modular architecture)
* **LLM:** Ollama (Llama 3.2)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace
* **Backend:** FastAPI
* **Frontend:** Gradio
* **Evaluation:** DeepEval

## 🚀 Installation & Setup

### 1. Prerequisites

* Python 3.11 or 3.12
* [Ollama](https://ollama.ai/) installed and running locally.
* Pull the required model:
  ```bash
  ollama pull llama3.2
  ```

### 2. Environment Setup

Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/anirudhgalagali97-dotcom/Agentic-HR-Assistant-Enterprise-RAG-with-Self-Correction.git
cd Agentic-HR-Assistant-Enterprise-RAG-with-Self-Correction
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

This project uses the modern, modularized LangChain ecosystem:

```bash
pip install -r requirements.txt
# Or install manually:
pip install langchain-core langchain-community langchain-text-splitters langgraph langchain-chroma langchain-ollama duckduckgo-search lark fastapi uvicorn gradio
```

## 🎯 Usage

### Start the Backend API (FastAPI)

```bash
python -m api.main
```

### Start the User Interface (Gradio)

Open a second terminal, activate the environment, and run:

```bash
python -m frontend.app
```

Navigate to http://127.0.0.1:7860 in your browser to interact with the agent.

## 📊 Evaluation

The system includes an automated evaluation suite (eval_suite.py) utilizing DeepEval. It measures metrics like Contextual Precision, Faithfulness, and Answer Relevancy across a benchmark of complex HR scenarios.

**Note:** Full evaluation metric generation requires OpenAI API keys. However, the core RAG pipeline and agentic routing execute entirely locally.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📄 License

This project is licensed under the MIT License.
