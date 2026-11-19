# BizInsight Agent

BizInsight Agent is an open-source candidate for governed business analytics. It combines a FastAPI backend, LangGraph orchestration, PostgreSQL-backed data access, retrieval, and a Vue web client. It supports metric exploration, root-cause analysis, multi-dimensional analysis, and confirmation-based complex analysis plans.

## Privacy and credentials

This repository contains source code and synthetic tests only. It does not include a database, local `.env`, database dumps, or the original Git history. Configure provider credentials and database passwords in your own ignored `.env` file; never commit credentials or production data.

The semantic interpreter can send user questions to the configured LLM provider when enabled. Review `SEMANTIC_INTERPRETER_MODE` and `SEMANTIC_INTERPRETER_SEND_RAW_QUESTION` before using real user data. Use synthetic data for development unless you have an approved privacy and retention policy.

## Requirements

- Python 3.13
- PostgreSQL for persistent application data
- Optional: Qdrant for retrieval and a compatible LLM provider
- Node.js/npm for the web client

## Local setup

1. Create and activate a Python environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create `.env` from `.env.example` and fill in values for the services you enable. At minimum, configure `POSTGRES_URL`. For LLM features, set `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` (or the provider settings documented in `.env.example`). Do not use example or development passwords.

3. Start the API from the repository root:

   ```powershell
   uvicorn app.main:app --reload
   ```

4. Install and start the web client:

   ```powershell
   cd web
   npm install
   npm run dev
   ```

Docker Compose profiles may require additional database passwords and service tokens. Compose intentionally refuses to start those services until their required environment variables are set.

## Tests

Run backend tests through the project shard runner so tests use an isolated configuration:

```powershell
python scripts/run_test_shard.py --shard no-db --name local -- tests -q
```

Run frontend tests from `web/`:

```powershell
npm test -- --run
```

Tests that require PostgreSQL need a separately provisioned isolated test database. Do not point test settings at a personal or production database.

## Project layout

- `app/api`, `app/graph`: API and agent workflow
- `app/analysis`, `app/data`: analytics and data access
- `app/complex_analysis`: capability and confirmed-plan logic
- `app/llm`, `app/context`: provider client and semantic contracts
- `web`: Vue frontend
- `evaluation`, `tests`: synthetic evaluation cases and automated tests

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE). Copyright 2026 BizInsight contributors.

If you use or deploy BizInsight, I would appreciate a note through the repository issue tracker. This is a voluntary request and is not an additional condition of the Apache-2.0 license.
