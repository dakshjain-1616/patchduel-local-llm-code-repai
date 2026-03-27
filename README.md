# PatchDuel: Local LLM Code Repair Arena – Audit local LLM bug fixes with visual diffs

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-142%20passed-brightgreen.svg)]()

> Stop guessing which local model fixes code best—see the exact diffs and log every verdict locally.

## Install

```bash
git clone https://github.com/dakshjain-1616/patchduel-local-llm-code-repai
cd patchduel-local-llm-code-repai
pip install -r requirements.txt
```

## The Problem

Existing benchmarks like `instruct-eval` provide abstract accuracy scores without preserving the context of specific failure modes, making it impossible to audit why a model failed on a bug fix. Developers cannot reproduce results locally without incurring API costs or relying on stale, centralized leaderboards that lack deterministic logging.

## Who it's for

This tool is for ML engineers and developers deploying local LLMs for coding assistance who need to validate model performance on specific bug fixes without relying on cloud APIs or stale leaderboards. When you are tuning a local Ollama model for a production codebase, you need to see exactly where the patch went wrong, not just a pass/fail score.

## Quickstart

Launch the interactive Gradio arena to compare two local models side-by-side:

```bash
python app.py
```

Or run a specific scenario programmatically:

```python
from examples import 01_quick_start
01_quick_start.run_duel("llama3", "mistral")
```

## Key features

- Split-screen diff view with git-style colored highlights (green/red/gray)
- Dual provider support for Ollama (local) and OpenRouter (cloud)
- Reproducible SQLite logging for every run, input, and verdict
- Built-in evaluation workflow with win/loss/tie tracking and notes

## Run tests

```bash
pytest tests/ -q
# 142 passed
```

## Project structure

```
patchduel-local-llm-code-repai/
├── patchduel_local_llm_/      ← main library
├── core/                      ← engine logic (diff, db, ollama)
├── examples/                  ← usage demos (01-04)
├── tests/                     ← test suite
├── app.py                     ← Gradio interface
└── requirements.txt
```

---