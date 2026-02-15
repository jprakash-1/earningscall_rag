#!/usr/bin/env bash
set -euo pipefail

# Teaching-oriented quickstart:
# 1) Build a small index for fast local iteration.
# 2) Launch Streamlit app.

INDEX_LIMIT="${INDEX_LIMIT:-200}"
NAMESPACE="${PINECONE_NAMESPACE:-earnings-call-rag}"

python -m src.indexing.cli_index --mode small --limit "${INDEX_LIMIT}" --strategy baseline --namespace "${NAMESPACE}"
streamlit run src/app/streamlit_app.py
