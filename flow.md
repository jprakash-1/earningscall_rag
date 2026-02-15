# Module-by-Module Learning Flow

Follow this exact **module-by-module lab order**.

1. **Start with the map**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/LEARN.md`
- `/Users/jainendra/Desktop/projects/rag-tutorial/README.md`

Run:
```bash
cd /Users/jainendra/Desktop/projects/rag-tutorial
```

2. **Config + fail-fast + logging**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/config.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/utils/logging.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/utils/ids.py`

Run:
```bash
python -c "from src.config import settings; print(settings)"
```

Learn check:
- Understand how `validate_env()` blocks missing keys.
- Understand why JSON logs are used.

3. **Data ingestion layer**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/data/cleaners.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/data/hf_loader.py`

Run:
```bash
python -m src.data.hf_loader --mode small --limit 20
pytest -q tests/test_loader.py
```

Learn check:
- Understand canonical schema: `question, answer, text, doc_id, metadata`.

4. **Document builder + chunking**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/data/doc_builder.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/data/chunking.py`

Run:
```bash
python -m src.data.chunking --mode small --limit 20 --print-samples 3
pytest -q tests/test_chunking.py
```

Learn check:
- Compare `baseline` vs `structure_aware`.

5. **Embeddings + Pinecone indexing**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/indexing/embedder.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/indexing/pinecone_client.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/indexing/indexer.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/indexing/cli_index.py`

Run:
```bash
python -m src.indexing.cli_index --mode small --limit 200 --strategy baseline
pytest -q tests/test_retrieval.py
```

Learn check:
- If keys are missing, confirm fail-fast error is clear.
- If keys exist, verify vectors upsert.

6. **RAG chain with citations**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/rag/prompts.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/rag/schemas.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/rag/retriever.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/rag/chains.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/utils/llm.py`

Run:
```bash
python -m src.rag.chains --query "What were the risks mentioned?" --mode pinecone
```

Learn check:
- See how context is formatted with `S1, S2...`.
- See JSON parsing + citation mapping.

7. **Agentic routing with LangGraph**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/graph/state.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/graph/nodes.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/graph/graph.py`

Run:
```bash
python -m src.graph.graph --query "Can you explain this?" --debug 1
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
python -m src.graph.graph --query "What is operating margin?" --debug 1
pytest -q tests/test_graph_routing.py
```

Learn check:
- Observe route selection: `clarify`, `retrieve`, `direct`.

8. **Tracing + observability**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/utils/tracing.py`

Run:
```bash
python -m src.graph.graph --query "Summarize guidance changes for Tesla Q2" --debug 1
```

Learn check:
- With LangSmith env set, traces should show router/retriever/synthesizer spans.

9. **Streamlit product layer**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/app/streamlit_app.py`

Run:
```bash
PYTHONPATH=. streamlit run src/app/streamlit_app.py
```

Learn check:
- Test filters (`company`, `section`), sources panel, debug panel.

10. **Evaluation pipeline**
Read:
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/eval/build_dataset.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/eval/evaluators.py`
- `/Users/jainendra/Desktop/projects/rag-tutorial/src/eval/run_eval.py`

Run:
```bash
python -m src.eval.build_dataset --limit 50
python -m src.eval.run_eval --experiment baseline
python -m src.eval.run_eval --experiment improved
```

Inspect:
- `/Users/jainendra/Desktop/projects/rag-tutorial/reports/eval_dataset_preview.json`
- `/Users/jainendra/Desktop/projects/rag-tutorial/reports/eval_summary.md`

11. **Final full check**
Run:
```bash
pytest -q
```

If you want, we can create a **daily study checklist** (Day 1 to Day 5) based on this exact sequence.
