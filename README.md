# DTS407TC A2: Adversarial Debate Argument Generator

This project implements a domain-specific large-model system for debate preparation. A user enters a debate motion and a target side, then the system compares:

1. A direct single-agent generation strategy.
2. A four-agent adversarial debate strategy with pro/con constructive speeches, rebuttals, and final synthesis.

The system uses the open-source local model `qwen3.5:4B` through Ollama and Tavily for web search.

## Quick Start

```bash
cp .env.example .env
# edit .env and add TAVILY_API_KEY
ollama serve
ollama pull qwen3.5:4B

uv sync --extra dev
npm install

uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
npm run dev
```

Open `http://127.0.0.1:5173`.

`OLLAMA_ENABLE_THINKING=false` is the demo-safe default because Qwen thinking can spend a long time in the hidden reasoning channel. Set it to `true` when you want the full `/think` behaviour described in the report.

## API

- `GET /api/health`
- `GET /api/topics?language=en|zh`
- `POST /api/generate`
- `POST /api/generate/stream`

Request body:

```json
{
  "topic": "AI systems should have legal personhood",
  "target_side": "pro",
  "language": "en",
  "use_search": true,
  "use_cache": true
}
```

## Evaluation

Start the backend, then run:

```bash
uv run python evaluation/run_eval.py --limit 2 --language en --side pro
```

Outputs are written to `evaluation/results/<timestamp>/`:

- `results.jsonl`
- `metrics.csv`
- `summary.md`

The CSV includes empty human-scoring columns for manual evaluation.

Search results and full generation results are cached in `.cache/` when `use_cache=true`. Run the demo motions once before presentation to make repeated demos return immediately.

## Deliverables

Draft materials live in:

- `docs/report/dts407tc_a2_report.tex`
- `docs/presentation/dts407tc_a2_slides.md`

Export and package:

```bash
bash scripts/export_docs.sh
bash scripts/package_submission.sh
```

The packaging script excludes `.env`, caches, `node_modules`, build output, and evaluation result folders.
