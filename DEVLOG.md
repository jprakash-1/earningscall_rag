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
