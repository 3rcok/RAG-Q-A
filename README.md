# RAG-Q-A

A **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, ChromaDB, and OpenAI — part of an AI/ML class curriculum. The pipeline loads PDFs, chunks and embeds them into a vector store, and answers natural-language questions using a retrieval-backed LLM chain.

## Project Structure

```
RAG-Q-A/
└── Rag-vector-databases/
    ├── rag_vector_databases_part6_working.ipynb  # Main notebook
    ├── requirements.txt                           # Python dependencies
    └── data/                                      # Place PDF files here
```

## What the Notebook Does

| Step | Description |
|------|-------------|
| 1 — Load | Reads all PDFs from `data/` using `PyPDFLoader` |
| 2 — Chunk | Splits text with `RecursiveCharacterTextSplitter`; compares chunk sizes 500 vs 1000 |
| 3 — Embed & Store | Embeds chunks with `text-embedding-3-small` and persists in **ChromaDB** |
| 4 — Retrieve | Runs similarity search for 3 sample queries and annotates relevance |
| 5 — RAG Chain | Wires `RetrievalQA` with a custom prompt template and generates answers via `gpt-*` |

## Setup

### 1. Install dependencies

```bash
pip install -r Rag-vector-databases/requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the `Rag-vector-databases/` folder:

```
OPENAI_API_KEY=sk-...
```

### 3. Add a PDF

Drop one or more `.pdf` files into `Rag-vector-databases/data/`.

### 4. Run the notebook

Open `rag_vector_databases_part6_working.ipynb` in Jupyter or VS Code and run all cells.

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-openai` / `langchain-community` | LLM orchestration & chains |
| `chromadb` | Local vector store |
| `pypdf` | PDF loading |
| `openai` | Embeddings & chat completions |
| `python-dotenv` | API key management |

## Notes

- The ChromaDB store is persisted locally in `chroma_part6_store/` (git-ignored).
- Chunk size experiments show 500-char chunks produce more granular retrieval; 1000-char chunks provide broader context per chunk.
- Requires an active OpenAI API key with access to `text-embedding-3-small` and a chat model.
