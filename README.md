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

## Step 4 verification

Run:

```bash
python -m src.rag.chains --query "What were the risks mentioned?" --mode pinecone
pytest -q
```

Expected behavior:
- If Pinecone + Groq env vars are configured, command returns JSON answer with citations.
- If keys are missing, command fails fast with a clear error message.

## Step 5 verification

Run:

```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
pytest -q
```

Expected behavior:
- Graph output includes route decision (`retrieve`, `clarify`, or `direct`).
- Tests validate deterministic routing behavior for ambiguous/specific/general queries.

## Step 6 verification

Run any query command, for example:

```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
```

Then confirm in LangSmith:
1. `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` are set.
2. `LANGCHAIN_TRACING_V2=true`.
3. Open your LangSmith project and verify runs include:
   - `router_node` / `router_llm_decision`
   - `retriever_query`
   - `synthesize_from_chunks`

## Step 7 verification

Run:

```bash
streamlit run src/app/streamlit_app.py
```

Then in the app:
1. Ask a question in chat.
2. Verify source citations appear under `Sources`.
3. Toggle `DEBUG mode` and verify router decision, retrieval snippets, and latency appear.

## Step 8 verification

Run:

```bash
python -m src.eval.build_dataset --limit 50
python -m src.eval.run_eval --experiment baseline
python -m src.eval.run_eval --experiment improved
```

Outputs:
- Dataset preview dump: `reports/eval_dataset_preview.json`
- Evaluation report: `reports/eval_summary.md`
