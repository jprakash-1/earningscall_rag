# DEVLOG

This file tracks exact commands run during implementation.

## Step 0
- `git init`
- `python -c "from src.config import settings; print(settings)"`
- `pytest -q`
- `mkdir -p src/utils tests && cat > ...` (created Step 0 project files)

## Step 1
- `mkdir -p src/data src/utils tests && cat > ...` (created loader, cleaners, ids, tests, and .gitignore)
- `git rm -r --cached src/__pycache__ tests/__pycache__`
- `python -m src.data.hf_loader --mode small --limit 50`
- `pytest -q`
