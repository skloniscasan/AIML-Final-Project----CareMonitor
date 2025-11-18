# CareMonitor — Clinical Reasoning RAG Streamlit App

Brief: CareMonitor is a prototype clinical reasoning assistant using retrieval-augmented generation (RAG).
It provides a Streamlit-based UI to ingest patient data, retrieve similar historical cases from a Chroma
vector index, and call a local LLM (via Ollama) to produce a structured clinical analysis.

## Key Features
- Build a small Chroma vector store from human-authored example cases (`cases.txt`).
- Simple vitals parser and heuristic 12-hour risk scoring.
- Streamlit dashboard to add patients, view risk, and see AI-generated clinical summaries.

## Repository layout
- `AIML Project/app/app2/` : Primary application code and data used by the prototype
  - `streamlit_app.py` : Main Streamlit UI (runs the app, RAG retrieval, and Ollama invocation)
  - `build_db2.py` : Script to build a Chroma vector index from `cases.txt`
  - `cases.txt` : Example clinical cases used as documents for the vector store
  - `prompt_template2.py` : Prompt template used to format the clinical-reasoning prompt
  - `patients.json` : Local JSON data store created/updated by the Streamlit app (patient records)
  - `chroma_index/` : Directory where Chroma persists its SQLite files (created by `build_db2.py`)

## Requirements
- Python 3.9+ recommended
- Typical Python packages used (not exhaustive):
  - `streamlit`
  - `langchain-core`, `langchain-community` (for `Chroma` wrapper and `HuggingFaceEmbeddings`)
  - `sentence-transformers` (used by `HuggingFaceEmbeddings` model)
  - `transformers`, `torch` (may be required depending on embedding backend)

Install with pip (example):

```bash
python -m pip install --upgrade pip
pip install streamlit langchain-core langchain-community sentence-transformers transformers torch
```

Notes:
- If you plan to run embeddings locally, ensure you have the required ML libs and models downloaded.
- The app uses `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` by default.

## Build the Chroma DB (RAG index)
1. Change directory to the app folder:

```bash
cd "AIML Project/app/app2"
```

2. Run the builder to create `chroma_index/` from `cases.txt`:

```bash
python build_db2.py
```

This reads `cases.txt`, splits on `=== CASE ===`, creates document chunks, computes embeddings,
and persists a Chroma index under `chroma_index/`.

## Run the Streamlit app
1. From the same folder run:

```bash
cd "AIML Project/app/app2"
streamlit run streamlit_app.py
```

2. In the app you can:
- Add a new patient (name, main symptom, history, onset, vitals)
- The app will retrieve similar cases from the Chroma index, format a prompt (see `prompt_template2.py`),
  and call the local Ollama model to generate an AI summary.

Important configuration notes:
- `streamlit_app.py` has a hard-coded `OLLAMA_EXE` pointing to a Windows path. On Linux/macOS, set it to the
  `ollama` executable (or the full path) or modify the `ask_ollama()` implementation to call the model endpoint you use.
- If you do not have Ollama, replace `ask_ollama()` with another model call (OpenAI, local transformer, etc.),
  or stub the function to return a placeholder summary while developing.

## Data and persistence
- `patients.json` keeps patient records, parsed vitals, computed risk and AI summaries. Back up this file if needed.
- `chroma_index/` contains the persisted vector store created by `build_db2.py`.

## Prompt & RAG behavior
- The prompt in `prompt_template2.py` instructs the model to produce a structured clinical analysis with
  sections: relevant case insights, clinical analysis, differential diagnosis, 12-hour risk assessment, and recommended next steps.
- Retrieval is configured with `HuggingFaceEmbeddings` + Chroma; retrieval uses MMR (see `streamlit_app.py`).

## Troubleshooting
- If embeddings fail due to missing model files, install `sentence-transformers` and the model, or use a hosted embeddings API.
- If the Streamlit app fails to call Ollama, check `OLLAMA_EXE`, PATH settings, and whether the model name is available locally.
- If `patients.json` is invalid JSON, the app will recreate an empty DB; the loader in `streamlit_app.py` attempts to handle JSONDecodeError.

## Next steps & suggestions
- Add a `requirements.txt` or `pyproject.toml` to lock dependencies.
- Add CI checks that build the Chroma index and run static checks.
- Add tests for the vitals parser and risk scoring functions in `streamlit_app.py`.

## License
This repository does not currently include a license file. Add one if you intend to publish or share.

---
Created by project inspection on 2025-11-18.
