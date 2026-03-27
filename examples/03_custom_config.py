"""
03_custom_config.py — Customising PatchDuel via environment variables.

Demonstrates:
- Reading and overriding configuration at runtime
- Changing Ollama server URL and timeout
- Using a custom SQLite database path
- Checking provider availability before making requests
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

# ── Override config before importing the clients ──────────────────────────────
# These env vars are read at import time, so set them before the first import.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_TIMEOUT", "30")          # shorter timeout for demo
os.environ.setdefault("OLLAMA_HEALTH_TIMEOUT", "3")

# Use a temp DB so this example never touches the project's main patchduel.db
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    custom_db = f.name
os.environ["DB_PATH"] = custom_db

from patchduel_local_llm_.ollama_client import is_ollama_running, list_models
from patchduel_local_llm_.openrouter_client import is_configured, list_models as or_list_models
from patchduel_local_llm_.database import init_db, run_count

# ── 1. Show active configuration ──────────────────────────────────────────────
print("=== Active configuration ===")
print(f"  OLLAMA_BASE_URL     : {os.environ.get('OLLAMA_BASE_URL')}")
print(f"  OLLAMA_TIMEOUT      : {os.environ.get('OLLAMA_TIMEOUT')}s")
print(f"  OLLAMA_HEALTH_TIMEOUT: {os.environ.get('OLLAMA_HEALTH_TIMEOUT')}s")
print(f"  DB_PATH             : {custom_db}")
print()

# ── 2. Provider availability check ───────────────────────────────────────────
print("=== Provider status ===")
if is_ollama_running():
    models = list_models()
    print(f"  Ollama: online — {len(models)} model(s) available")
    for m in models[:5]:
        print(f"    • {m}")
else:
    print("  Ollama: offline (start with: ollama serve)")

if is_configured():
    or_models = or_list_models()
    print(f"  OpenRouter: configured — {len(or_models)} curated models")
else:
    print("  OpenRouter: no API key set (add OPENROUTER_API_KEY to .env)")
print()

# ── 3. Initialise the custom DB ───────────────────────────────────────────────
init_db(custom_db)
print(f"=== Custom DB initialised at {custom_db} ===")
print(f"    Runs in DB: {run_count(custom_db)}")

os.unlink(custom_db)
print("    (cleaned up temp DB)")
