# Helpdesk Triage Agent — README

A small FastAPI app that classifies incoming support tickets with a tiny PyTorch classifier and uses an LLM (OpenAI) to draft replies or escalate. This README shows how to set up and run the app using **uv** (your `uv` tool/CLI), including the `git clone` step and the `uv init` / `uv add` / `uv pip` workflow you requested.

> Files included in the repository
- `fastapi_triage_agent.py` — main FastAPI app that runs the triage workflow.
- `main.py` — simple Python client (example runs) that calls `/predict` and `/batch`.
- `requirements.txt` — Python dependencies.

---

## 0) Clone the repository (get the `.py` files and others)

Clone the repo that contains `fastapi_triage_agent.py` and `main.py`:

```bash
git clone https://github.com/alyab16/helpdesk-triage-agent.git
cd helpdesk-triage-agent
```

---

## 1) Prepare the project

From the project directory (where `fastapi_triage_agent.py`, `main.py` and `requirements.txt` live):

```bash
# initialize uv workspace
uv init
```

---

## 2) Install the Python dependencies (requirements.txt) **first**

Install everything from `requirements.txt` first. This ensures auxiliary packages (e.g., transformers, langchain, python-dotenv, pydantic, etc.) are in place before you add the large PyTorch wheel — which can help avoid dependency-resolution surprises.

```bash
# install everything in your requirements file using uv's package installer
uv add -r requirements.txt
```

`requirements.txt` (for reference) should contain:
```
fastapi>=0.116.1
langchain>=0.2
langchain-openai>=0.3.32
langgraph>=0.6.6
transformers>=4.56.0
uvicorn>=0.35.0
python-dotenv==1.1.1
pydantic==2.11.7
requests
```

---

## 3) Install PyTorch (CUDA wheel) **after** installing requirements

Now install the PyTorch wheel for CUDA 12.6 (if you have a CUDA-capable GPU and want the CUDA build). Installing PyTorch after the rest of the dependencies reduces the chance of pip trying to resolve large wheels into your `uv` lock step in an unexpected way.

**CUDA (GPU) install:**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU-only fallback (if you don't have a GPU or want CPU-only):**
```bash
uv pip install torch torchvision
```

> Note: the PyTorch wheel is large. Make sure you have enough disk space and that your Python version is supported.

---

## 4) Create a `.env` file with required environment variables

Create a `.env` in the project root with at least the OpenAI API key:

```text
OPENAI_API_KEY=sk-...
# optional
TRIAGE_NEG_THRESHOLD=0.80
```

You can also set `TRIAGE_NEG_THRESHOLD` (e.g., `0.80`) if you want to override the default.

---

## 5) Run the FastAPI application

This is the command you requested for running via `uv`:

```bash
uv run uvicorn fastapi_triage_agent:app --reload --host 0.0.0.0 --port 8000
```

If you prefer to run `uvicorn` directly (without `uv`), you can also use:

```bash
uvicorn fastapi_triage_agent:app --reload --host 0.0.0.0 --port 8000
```

---

## 6) Run the sample client (`main.py`)

`main.py` contains a few example calls to the server (single `/predict` requests and a `/batch` call). Run it like this from the project root while the server is running:

```bash
python main.py
```

What it does:
- Posts a single short ticket asking about billing/plan change.
- Posts a single angry/crash ticket (expected to be NEGATIVE / escalate).
- Posts a small batch of 3 tickets to `/batch`.

Example expected output (JSON objects returned from the API; actual `response` text depends on the LLM and may differ):

```json
{"sentiment":"POSITIVE","score":0.92,"action":"RESPOND","response":"..."}
====================================================================================================
{"sentiment":"NEGATIVE","score":0.95,"action":"ESCALATE","response":"..."}
{"results":[
  {"index":0,"sentiment":"NEGATIVE","score":0.95,"action":"ESCALATE","response":"..."},
  {"index":1,"sentiment":"POSITIVE","score":0.92,"action":"RESPOND","response":"..."},
  {"index":2,"sentiment":"POSITIVE","score":0.98,"action":"RESPOND","response":"..."}
]}
```

Notes:
- The `response` fields contain LLM-generated text; exact wording varies.
- If you get connection errors, ensure the FastAPI server is running on `localhost:8000` and there are no firewall conflicts.

---

## 7) Health check & quick tests

1. Health endpoint:
```bash
curl http://localhost:8000/health
# should return {"status":"ok", "device":"cuda"|"cpu"|"mps", "threshold":...}
```

2. Single predict example (curl):
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticket_text":"My order never arrived and I was charged twice."}'
```

3. Batch predict example (curl):
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"tickets":["Where is my refund?","I cannot login to my account."]}'
```

---

## 8) Tech stack & architecture details

This project combines a small PyTorch classifier with an LLM-driven orchestration layer. Key components:

- **FastAPI** — HTTP API framework serving `/predict`, `/batch`, and `/health`.
- **Uvicorn** — ASGI server used to run the FastAPI app.
- **PyTorch** — used for the local sentiment classifier (`distilbert-base-uncased-finetuned-sst-2-english`).
  - The code auto-selects a device: `"cuda"` (NVIDIA GPU) if available, otherwise `"mps"` (Apple Silicon) if available, otherwise `"cpu"`.
  - If `device == "cuda"` the model is converted to `half()` precision for speed.
  - Install the matching PyTorch CUDA wheel (example `cu126`) if you want GPU acceleration.
- **Transformers (Hugging Face)** — model/tokenizer loading and tokenization.
- **LangGraph** — used to build a small compiled StateGraph workflow (classify → decide → compose reply / escalate).
- **LangChain OpenAI wrapper (ChatOpenAI)** — used to call the OpenAI chat model for drafting responses and escalations.
  - The app is configured to use `gpt-4o-mini` for the ChatOpenAI calls (see `get_llm()` in the code).
- **python-dotenv** — loads `.env` variables like `OPENAI_API_KEY`.
- **pydantic** — input/output schema validation for FastAPI endpoints.
- **requests** — used by `main.py` test client.

Workflow summary:
1. Client sends a ticket to `/predict`.
2. The compiled `StateGraph` invokes `classify_with_torch` (local PyTorch classifier) to produce sentiment and score.
3. `decide` uses configured threshold `TRIAGE_NEG_THRESHOLD` to choose `ESCALATE` vs `RESPOND`.
4. Depending on the routing, the app calls an LLM prompt to compose a reply or escalation summary.
5. The API returns `sentiment`, `score`, `action`, and the drafted `response`.

---