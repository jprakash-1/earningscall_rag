# earningscall_rag

Agentic RAG over earnings call transcripts using:
- LangChain
- LangGraph
- LangSmith
- Pinecone
- Groq API (LLM generation/router)

The data source is HuggingFace `lamini/earnings-calls-qa`.

## What This Repo Includes

- Ingestion from HuggingFace with canonical schema normalization
- Two chunking strategies (baseline recursive and structure-aware)
- Pinecone indexing CLI with index auto-create
- Citation-aware RAG chain
- LangGraph routing (`retrieve` vs `clarify` vs `direct`)
- Streamlit chat app with filters + source panels
- LangSmith dataset builder + eval harness
- Pytest suite for loader/chunking/retrieval/router smoke coverage

## Project Layout

```text
src/
  app/              # Streamlit UI
  data/             # HF loader, cleaners, chunking, doc builder
  eval/             # LangSmith dataset + eval harness
  graph/            # LangGraph state + nodes + graph runner
  indexing/         # Embeddings, Pinecone client, indexing CLI
  rag/              # Retriever, prompts, schemas, chain
  utils/            # Logging, tracing, Groq model factory
tests/
reports/
scripts/
```

## Prerequisites

- Python 3.11+
- Pinecone account/API key
- Groq API key
- Optional but recommended: LangSmith API key

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment Setup

Create `.env` from `.env.example` and fill values:

```bash
cp .env.example .env
```

Important vars:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `PINECONE_NAMESPACE`
- `LANGSMITH_API_KEY` (optional but needed for hosted traces/eval)
- `LANGSMITH_PROJECT`
- `LANGCHAIN_TRACING_V2`

## Pinecone Dimension Guidance

The index dimension must match the embedding model output dimension.

Defaults in this repo:
- `HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (commonly 384 dims)
- Fallback deterministic embeddings use `FALLBACK_EMBEDDING_DIM` (default 1536)

How dimension is enforced:
- The code computes a probe embedding at runtime.
- If the Pinecone index does not exist, it is created with that dimension.
- If it exists with a different dimension, indexing fails fast with a clear error.

## One-Command Quickstart (Small Mode)

```bash
bash scripts/dev_small.sh
```

What this does:
1. Indexes a small slice into Pinecone.
2. Starts Streamlit app.

## Manual Workflow

### 1) Build index

```bash
python -m src.indexing.cli_index --mode small --limit 200 --strategy baseline
```

### 2) Run CLI RAG query

```bash
python -m src.rag.chains --query "What were the risks mentioned?" --mode pinecone
```

### 3) Run agentic graph query

```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
```

### 4) Launch app

```bash
streamlit run src/app/streamlit_app.py
```

### 5) Build eval dataset + run experiments

```bash
python -m src.eval.build_dataset --limit 50
python -m src.eval.run_eval --experiment baseline
python -m src.eval.run_eval --experiment improved
```

Outputs:
- `reports/eval_dataset_preview.json`
- `reports/eval_summary.md`

## Testing

```bash
pytest -q
```

External-service smoke behavior:
- Pinecone retrieval tests are skipped when required env vars are missing.
- Evaluation falls back to local scoring when LangSmith credentials are missing.

## LangSmith Tracing

Set:
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGCHAIN_TRACING_V2=true`

Then run graph or chain commands and inspect traces in LangSmith.
Expected trace nodes include:
- `router_node` / `router_llm_decision`
- `retriever_query`
- `synthesize_from_chunks`

## Troubleshooting

- Missing env vars: commands fail fast with explicit variable names.
- Pinecone dimension mismatch: recreate index or switch embedding model to match.
- No citations returned: ensure indexing ran on the same namespace as query/eval.
- Offline/no-network local dev: loader/chunking/eval scripts use fallback behavior for rapid iteration.

## Step 0 verification

```bash
python -c "from src.config import settings; print(settings)"
pytest -q
```

## Step 1 verification

```bash
python -m src.data.hf_loader --mode small --limit 50
pytest -q
```

## Step 2 verification

```bash
python -m src.data.chunking --mode small --print-samples 3
pytest -q
```

## Step 3 verification

```bash
python -m src.indexing.cli_index --mode small --limit 200
pytest -q
```

## Step 4 verification

```bash
python -m src.rag.chains --query "What were the risks mentioned?" --mode pinecone
pytest -q
```

## Step 5 verification

```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
pytest -q
```

## Step 6 verification

```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
```

## Step 7 verification

```bash
streamlit run src/app/streamlit_app.py
```

## Step 8 verification

```bash
python -m src.eval.build_dataset --limit 50
python -m src.eval.run_eval --experiment baseline
python -m src.eval.run_eval --experiment improved
```

## Step 9 verification

Fresh install + `.env` configuration + small mode quickstart should work:

```bash
bash scripts/dev_small.sh
pytest -q
```
