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
