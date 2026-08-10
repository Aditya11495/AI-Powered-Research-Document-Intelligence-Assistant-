# AI Business & Research RAG Assistant

A portfolio-ready **Retrieval-Augmented Generation (RAG)** application for asking grounded questions over multiple PDF documents.

## Why this project is strong

This is intentionally more than a basic PDF chatbot. It demonstrates:

- Multi-document ingestion
- PDF parsing with PyMuPDF
- Overlapping text chunking
- Local semantic embeddings
- FAISS vector similarity search
- Top-k retrieval
- LLM-grounded generation
- Source/page citations
- OpenAI and Gemini provider support
- A deterministic demo mode
- A simple Streamlit UI

## Architecture

```text
PDFs
  ↓
PyMuPDF extraction
  ↓
Text cleaning + chunking
  ↓
Sentence-Transformer embeddings
  ↓
FAISS vector index
  ↓
User question
  ↓
Query embedding
  ↓
Top-k semantic retrieval
  ↓
Evidence context
  ↓
LLM
  ↓
Grounded answer + source citations
```

## Example use cases

Upload annual reports, research papers, technical documentation, or business reports and ask:

- What were the main revenue drivers?
- Compare two companies' strategies.
- What risks did management identify?
- Find all mentions of AI investment.
- Summarize a company's expansion strategy.
- What evidence supports this conclusion?

## Setup

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure an LLM

Copy `.env.example` to `.env` and add either:

```text
OPENAI_API_KEY=your_key
```

or:

```text
GEMINI_API_KEY=your_key
```

You can also run in **Local / Demo** mode without an API key.

### 4. Run

```bash
streamlit run app.py
```

## Recommended demo dataset

For a portfolio demonstration, use 5–10 public annual reports from 2–3 companies and ask comparative questions.

## Suggested next-level improvements

1. Add hybrid BM25 + vector retrieval.
2. Add a reranker such as a cross-encoder.
3. Add metadata filters for company/year/document type.
4. Add automatic RAG evaluation with faithfulness and answer relevance.
5. Add query rewriting for conversational follow-up questions.
6. Add a FastAPI backend and Docker deployment.
7. Add a retrieval-confidence threshold that refuses unsupported questions.
8. Add an evaluation dataset with 30–50 question/answer pairs.

## Resume bullets

**AI Business & Research RAG Assistant | Python, FAISS, Sentence Transformers, LLM, Streamlit**

- Built a multi-document Retrieval-Augmented Generation system for semantic question answering over PDF-based business and research documents.
- Implemented PDF extraction, overlapping chunking, dense embeddings, and FAISS similarity search to retrieve top-k evidence passages.
- Developed citation-aware LLM responses with source-document and page-level references and added a demo fallback for API-independent testing.
- Designed the system for extensibility with metadata filtering, reranking, hybrid retrieval, and RAG evaluation.

## Interview talking points

### Why RAG instead of fine-tuning?

RAG keeps knowledge outside model parameters, so documents can be updated without retraining the model. It also provides retrieved evidence that can be cited.

### Why FAISS?

FAISS provides efficient vector similarity search and is lightweight enough for a portfolio project.

### Why chunking?

Entire documents are usually too large and noisy to pass directly to an LLM. Chunking creates smaller retrieval units so the system can retrieve only the relevant evidence.

### Main limitation

Retrieval quality directly affects generation quality. Poor chunking or retrieval can lead to incomplete or unsupported answers, which is why reranking and evaluation are useful future improvements.
