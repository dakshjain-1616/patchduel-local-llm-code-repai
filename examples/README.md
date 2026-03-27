# PatchDuel Examples

Runnable scripts demonstrating different features of the `patchduel_local_llm_` package.
All examples work from any directory and use mock mode by default (no Ollama required).

## Running an example

```bash
python examples/01_quick_start.py
```

## Scripts

| Script | What it demonstrates |
|---|---|
| [01_quick_start.py](01_quick_start.py) | Minimal working example — load a scenario, get a mock patch, print the diff summary |
| [02_advanced_usage.py](02_advanced_usage.py) | Quality scoring, heuristic winner selection, and persisting results to SQLite |
| [03_custom_config.py](03_custom_config.py) | Runtime configuration via env vars — custom DB path, Ollama URL, provider status checks |
| [04_full_pipeline.py](04_full_pipeline.py) | End-to-end pipeline — all 8 scenarios, scoring, DB persistence, aggregate stats, JSON export |

## Live mode (requires Ollama)

`04_full_pipeline.py` supports real model inference:

```bash
ollama pull llama3.2
USE_LIVE=true python examples/04_full_pipeline.py
```
