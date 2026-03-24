# 🔍 Agentic RAG System

A production-ready Agentic RAG (Retrieval-Augmented Generation) system built with LangChain, LangGraph, and OpenAI. This system implements intelligent document retrieval, hybrid search, and agentic workflows with hallucination guardrails and comprehensive observability.

## 🌟 Features

### Core Features
- **📄 Data Ingestion Pipeline**: Load PDFs, split using RecursiveCharacterTextSplitter, store in persistent ChromaDB
- **🔄 Hybrid Retrieval**: Ensemble retriever combining BM25 + Vector Search for optimal results
- **🎯 Self-Querying Retriever**: LLM-powered metadata filtering from natural language queries
- **🤖 Agentic Logic**: Stateful LangGraph workflow with intelligent routing
- **🛡️ Hallucination Guardrails**: Document grading node that evaluates relevance before generation
- **📊 Observability**: Comprehensive logging of token usage, latency, and context precision

### Agent Workflow
1. **Analyze Query**: Understand user intent and determine search strategy
2. **Retrieve**: Fetch context from vector database using hybrid retrieval
3. **Grade Documents**: Evaluate document relevance (Hallucination Guardrail)
4. **Web Search** (optional): Fallback to DuckDuckGo if local docs insufficient
5. **Generate Answer**: Synthesize response using relevant context
6. **Check Hallucination**: Final quality verification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │   Upload    │  │   Query     │  │   Document Stats        ││
│  │   Endpoint   │  │   Endpoint   │  │   Endpoint              ││
│  └─────────────┘  └─────────────┘  └─────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Workflow                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ ANALYZE  │───▶│ RETRIEVE │───▶│  GRADE   │───▶│ GENERATE │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │                                     │                    │
│       │         ┌──────────┐                │                    │
│       └────────▶│WEBSearch │◀───────────────┘                    │
│                 └──────────┘         (if no relevant docs)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Observability Layer                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Token Usage    │  │  Latency Track   │  │ Context Prec.  │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- OpenAI API Key

### Local Development Setup

1. **Clone and setup environment:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

2. **Add PDF documents:**
```bash
# Place your PDFs in the data/documents directory
mkdir -p data/documents
cp your-documents/*.pdf data/documents/
```

3. **Run the system:**
```bash
# Option A: Run FastAPI server
python -m api.main

# Option B: Run Gradio frontend
python -m frontend.app

# Option C: Run both (development)
./start.sh
```

4. **Access the interfaces:**
- FastAPI: http://localhost:8000
- Gradio UI: http://localhost:7860
- API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

## 📁 Project Structure

```
agentic-rag/
├── config/
│   └── settings.py          # Configuration management
├── data_ingestion/
│   └── ingest.py            # PDF loading and ChromaDB storage
├── retrieval/
│   ├── retriever.py         # Hybrid retrieval (BM25 + Vector)
│   └── self_query.py        # Self-querying retriever
├── agents/
│   ├── state.py             # LangGraph state definition
│   ├── nodes.py             # Graph nodes (Retrieve, WebSearch, Grade, Answer)
│   └── graph.py              # LangGraph workflow
├── observability/
│   └── logging.py           # Token usage, latency, precision tracking
├── api/
│   └── main.py              # FastAPI backend
├── frontend/
│   └── app.py               # Gradio interface
├── data/
│   ├── documents/           # Place PDFs here
│   └── chroma_db/           # Persistent vector store
├── logs/                    # Application logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔌 API Endpoints

### Query Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query the RAG system |
| GET | `/health` | Health check |

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Ingest documents from data directory |
| POST | `/upload` | Upload and ingest a single PDF |
| DELETE | `/documents` | Clear all documents |
| GET | `/stats/documents` | Get document statistics |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats/system` | Get system statistics |
| POST | `/reset` | Reset the agent and cache |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'

# Get document stats
curl http://localhost:8000/stats/documents
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | **Required.** Your OpenAI API key |
| `TAVILY_API_KEY` | - | Optional. Tavily API key for enhanced web search |
| `LLM_MODEL` | `gpt-4-turbo-preview` | OpenAI model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `VECTOR_STORE_PATH` | `./data/chroma_db` | ChromaDB persistence path |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `GRADIO_PORT` | `7860` | Gradio server port |

## 📊 Observability

The system tracks the following metrics for every query:

### Metrics Tracked
- **Token Usage**: Prompt, completion, and total tokens
- **Latency**: Total, retrieval, grading, and generation latency
- **Context Precision**: Ratio of relevant documents to total retrieved
- **Hallucination Score**: Probability of hallucination in generated answer
- **Web Search Usage**: Whether web search was triggered

### Accessing Metrics

```bash
# Get system statistics
curl http://localhost:8000/stats/system

# Check logs
tail -f logs/rag_observability.log
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Test the API manually
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in the documents?"}'
```

## 🔒 Security Considerations

1. **API Key Management**: Store your OpenAI API key securely in environment variables or a secrets manager
2. **Input Validation**: All inputs are validated using Pydantic models
3. **Rate Limiting**: Consider adding rate limiting for production deployments
4. **CORS**: Configure CORS settings for your specific domain in production

## 🚢 Production Deployment

For production deployment:

1. **Use a reverse proxy** (nginx, Traefik) for SSL termination
2. **Set up monitoring** (Prometheus, Grafana) using the observability endpoints
3. **Configure proper CORS origins** in `api/main.py`
4. **Use persistent storage** for vector database (mount Docker volumes)
5. **Set up log rotation** for the observability logs

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Built with ❤️ using LangChain, LangGraph, and OpenAI
