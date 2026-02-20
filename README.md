# 🔍 Multimodal RAG Pipeline for Banking — Text + Image Understanding

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that handles **both text and images** from banking documents. Upload financial reports with charts, tables, and graphs — then ask questions in natural language and get accurate, grounded answers.

Built with **Azure OpenAI GPT-4o**, **Azure AI Search (Vector + Hybrid)**, **Azure AI Vision Image Embeddings**, and **LangChain**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Azure](https://img.shields.io/badge/Azure-OpenAI%20%7C%20AI%20Search%20%7C%20Vision-0078D4)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   PDF Upload ──▶ Document Cracking ──▶ Content Extraction        │
│                     │                      │                     │
│              ┌──────┴──────┐        ┌──────┴──────┐              │
│              │   Text      │        │   Images    │              │
│              │   Content   │        │   & Charts  │              │
│              └──────┬──────┘        └──────┬──────┘              │
│                     │                      │                     │
│              ┌──────┴──────┐        ┌──────┴──────┐              │
│              │   Text      │        │   Image     │              │
│              │  Chunking   │        │  Embeddings │              │
│              │  (Semantic) │        │  (CLIP /    │              │
│              └──────┬──────┘        │  AI Vision) │              │
│                     │               └──────┬──────┘              │
│              ┌──────┴──────┐               │                     │
│              │   Text      │               │                     │
│              │  Embeddings │               │                     │
│              │  (ada-002)  │               │                     │
│              └──────┬──────┘               │                     │
│                     │                      │                     │
│                     ▼                      ▼                     │
│              ┌─────────────────────────────────────┐             │
│              │     Azure AI Search Vector Index     │             │
│              │   (Hybrid: Text + Vector + Semantic) │             │
│              └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query ──▶ Query Embedding ──▶ Hybrid Search               │
│                                          │                       │
│                                   ┌──────┴──────┐               │
│                                   │  Text Chunks │               │
│                                   │  + Images    │               │
│                                   │  (Reranked)  │               │
│                                   └──────┬──────┘               │
│                                          │                       │
│                                   ┌──────┴──────┐               │
│                                   │   GPT-4o    │               │
│                                   │  Multimodal │               │
│                                   │  Generation │               │
│                                   └──────┬──────┘               │
│                                          │                       │
│                                   ┌──────┴──────┐               │
│                                   │  Grounded   │               │
│                                   │  Answer +   │               │
│                                   │  Citations  │               │
│                                   └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Multimodal Ingestion**: Processes PDFs with text, tables, charts, and images
- **Semantic Chunking**: Smart chunking that preserves context around tables and figures
- **Dual Embedding**: Text embeddings (text-embedding-ada-002) + Image embeddings (Azure AI Vision / CLIP)
- **Hybrid Search**: Combines keyword, vector, and semantic ranking for optimal retrieval
- **Image-Aware RAG**: Retrieves relevant charts/images alongside text for multimodal answers
- **GPT-4o Multimodal Generation**: Generates answers using both text context and images
- **Citation Tracking**: Every answer includes source citations with page numbers
- **Banking Domain Focus**: Optimized for financial reports, annual reports, regulatory filings

## 📁 Project Structure

```
project2-multimodal-rag-banking/
├── src/
│   ├── main.py                         # FastAPI application + Web UI serving
│   ├── config.py                       # Configuration
│   ├── services/
│   │   ├── document_cracker.py         # PDF → text + images extraction
│   │   ├── chunker.py                  # Semantic text chunking
│   │   ├── text_embedder.py            # Azure OpenAI text embeddings
│   │   ├── image_embedder.py           # Azure AI Vision image embeddings
│   │   ├── index_manager.py            # Azure AI Search index management
│   │   ├── retriever.py                # Hybrid search retrieval
│   │   ├── generator.py                # GPT-4o multimodal answer generation
│   │   ├── rag_pipeline.py             # End-to-end RAG orchestration
│   │   └── blob_storage.py             # Azure Blob Storage connector
│   ├── models/
│   │   └── schemas.py                  # Pydantic models
│   └── utils/
│       ├── pdf_utils.py                # PDF processing utilities
│       └── image_utils.py              # Image processing utilities
├── static/
│   └── index.html                      # Web UI — chat-style RAG interface
├── data/
│   └── sample_reports/                 # Sample banking PDFs
├── tests/
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Azure AI Search resource (S1 tier+ for vector search)
- Azure OpenAI with `text-embedding-ada-002` and `gpt-4o` deployments
- Azure AI Vision resource (for image embeddings)

### Setup

```bash
git clone https://github.com/yourusername/multimodal-rag-banking.git
cd multimodal-rag-banking
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure credentials
uvicorn src.main:app --reload --port 8001
```

### Open the Web UI

Open `http://localhost:8001` in your browser — a full chat-style RAG interface loads automatically.

### Usage (CLI)

```bash
# 1. Ingest a financial report
curl -X POST "http://localhost:8001/api/v1/ingest" \
  -F "file=@data/sample_reports/annual_report_2024.pdf"

# 2. Ask a question
curl -X POST "http://localhost:8001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue growth trend shown in the Q3 chart?"}'
```

## ☁️ Azure Deployment (Web App)

```bash
# 1. Create resources
az group create --name rg-rag-banking --location uaenorth
az appservice plan create --name plan-rag-banking --resource-group rg-rag-banking --sku B1 --is-linux
az webapp create --name rag-banking-app --resource-group rg-rag-banking \
  --plan plan-rag-banking --runtime "PYTHON:3.11"

# 2. Configure environment
az webapp config appsettings set --name rag-banking-app --resource-group rg-rag-banking --settings \
  AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com/" \
  AZURE_OPENAI_API_KEY="your-key" \
  AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net" \
  AZURE_SEARCH_API_KEY="your-key" \
  AZURE_VISION_ENDPOINT="https://your-vision.cognitiveservices.azure.com/" \
  AZURE_VISION_API_KEY="your-key" \
  AZURE_STORAGE_CONNECTION_STRING="your-connection-string"

# 3. Deploy
zip -r deploy.zip . -x "venv/*" "__pycache__/*" ".env"
az webapp deploy --name rag-banking-app --resource-group rg-rag-banking --src-path deploy.zip --type zip

# 4. Set startup command
az webapp config set --name rag-banking-app --resource-group rg-rag-banking \
  --startup-file "uvicorn src.main:app --host 0.0.0.0 --port 8000"
```

Live at: `https://rag-banking-app.azurewebsites.net`

### Storage Modes

| Mode | Condition | PDFs Stored | Metadata Stored |
|------|-----------|-------------|-----------------|
| **Azure Blob** | Connection string set | `rag-documents/pdfs/` | `rag-documents/metadata/` |
| **Local** | No connection string | `uploads/` | `outputs/` |

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Ingest a PDF document into the RAG index |
| `POST` | `/api/v1/query` | Ask a question — returns grounded answer with citations |
| `GET`  | `/api/v1/documents` | List all ingested documents |
| `DELETE`| `/api/v1/documents/{id}` | Remove document from index |
| `GET`  | `/api/v1/health` | Health check |

## 🧠 Key Design Decisions

### Why Hybrid Search over Pure Vector?
Banking documents contain precise numbers, dates, and codes (SWIFT, IBAN, account numbers). Pure vector search can miss exact matches. Hybrid search combines keyword precision with semantic understanding.

### Why Semantic Chunking?
Fixed-size chunking splits tables and figures mid-content. Our semantic chunker detects section boundaries, keeps tables intact, and links figure captions to their images.

### Why Image Embeddings?
Financial reports contain critical information in charts (revenue trends, risk distributions, portfolio allocations). Text-only RAG misses this entirely. Image embeddings enable retrieval of relevant charts when questions reference visual data.

### Why GPT-4o for Generation?
GPT-4o natively understands images — it can read charts, interpret graphs, and combine visual + textual context in a single response. This is essential for questions like "What does the revenue chart on page 12 show?"

## 🛠️ Tech Stack

- **Python 3.10+** — Core language
- **FastAPI** — REST API
- **Azure OpenAI** — GPT-4o (generation) + text-embedding-ada-002 (text embeddings)
- **Azure AI Vision** — Image embeddings (Florence model)
- **Azure AI Search** — Vector + hybrid search index
- **LangChain** — RAG orchestration
- **PyMuPDF (fitz)** — PDF processing and image extraction
- **Pillow / OpenCV** — Image preprocessing
- **Docker** — Containerization

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Jalal Ahmed Khan** — Senior AI Consultant | Microsoft Certified Trainer
