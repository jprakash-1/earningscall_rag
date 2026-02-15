# earningscall_rag

A teaching-first, end-to-end agentic RAG project over earnings call transcripts.

## Step 0 verification

Run:

```bash
python -c "from src.config import settings; print(settings)"
pytest -q
```

If both commands succeed, the repo skeleton is healthy and ready for Step 1.

## Step 1 verification

Run:

```bash
python -m src.data.hf_loader --mode small --limit 50
pytest -q
```

If loader output prints a count and at least one sample record, Step 1 is wired.

## Step 2 verification

Run:

```bash
python -m src.data.chunking --mode small --print-samples 3
pytest -q
```

If chunk counts are non-zero and tests pass, chunking strategies are working.

## Step 3 verification

Run:

```bash
python -m src.indexing.cli_index --mode small --limit 200
pytest -q
```

Expected behavior:
- If Pinecone env vars are configured, indexing should upsert vectors.
- If env vars are missing, CLI should fail fast with a clear error message.
