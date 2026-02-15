# DEVLOG

This file tracks exact commands run during implementation.

## Step 0
- `git init`
- `python -c "from src.config import settings; print(settings)"`
- `pytest -q`
- `mkdir -p src/utils tests && cat > ...` (created Step 0 project files)
- `git add README.md DEVLOG.md pyproject.toml .env.example src tests`
- `git commit -m "chore: bootstrap repo skeleton + config"`

## Step 1
- `mkdir -p src/data src/utils tests && cat > ...` (created loader, cleaners, ids, tests, and .gitignore)
- `git rm -r --cached src/__pycache__ tests/__pycache__`
- `python -m src.data.hf_loader --mode small --limit 50`
- `pytest -q`
- `git add .`
- `git commit -m "feat: add HF loader for lamini earnings-calls-qa + tests"`

## Step 2
- `cat > src/data/doc_builder.py`
- `cat > src/data/chunking.py`
- `cat > tests/test_chunking.py`
- `python -m src.data.chunking --mode small --print-samples 3`
- `pytest -q`
- `git add .`
- `git commit -m "feat: add chunking strategies + document builder"`

## Step 3
- `mkdir -p src/indexing`
- `cat > src/indexing/embedder.py`
- `cat > src/indexing/pinecone_client.py`
- `cat > src/indexing/indexer.py`
- `cat > src/indexing/cli_index.py`
- `cat > tests/test_retrieval.py`
- `python -m src.indexing.cli_index --mode small --limit 200` (failed fast as expected without Pinecone key)
- `pytest -q`
- `git add .`
- `git commit -m "feat: add Pinecone indexing pipeline + CLI"`

## Provider update (Groq-first)
- `cat > src/indexing/embedder.py` (migrated from OpenAI embeddings defaults to HuggingFace + fallback)
- `cat > src/utils/llm.py` (Groq LLM factory)
- `cat > src/rag/retriever.py`
- `cat > src/rag/prompts.py`
- `cat > src/rag/schemas.py`
- `cat > src/rag/chains.py`
- `python -m src.rag.chains --query "What were the risks mentioned?" --mode pinecone` (failed fast as expected without Pinecone key)
- `pytest -q`

## Step 5
- `cat > src/graph/state.py`
- `cat > src/graph/nodes.py`
- `cat > src/graph/graph.py`
- `cat > tests/test_graph_routing.py`
- `python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1`
- `pytest -q` (fixed routing heuristics and reran until green)

## Step 6
- `cat > src/utils/tracing.py`
- `apply_patch src/graph/nodes.py` (trace decorators)
- `apply_patch src/rag/retriever.py` (trace decorator)
- `apply_patch src/rag/chains.py` (trace + setup)
- `apply_patch src/graph/graph.py` (trace setup in CLI)
- `python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1`
- `pytest -q`
