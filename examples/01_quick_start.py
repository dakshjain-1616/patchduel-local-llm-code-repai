"""
01_quick_start.py — Minimal PatchDuel example.

Uses mock mode (no Ollama needed). Shows how to:
- Get a patch for a buggy function
- Compute and display the diff
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from patchduel_local_llm_.scenarios import get_scenario_by_id, mock_patch
from patchduel_local_llm_.diff_engine import render_html_diff, diff_summary

# Load a built-in bug scenario
scenario = get_scenario_by_id("sc-001")
buggy_code = scenario["buggy_code"]

print("Buggy code:")
print(buggy_code)
print()

# Get a mock patch (deterministic — no model required)
patched_code = mock_patch(buggy_code)

print("Patched code:")
print(patched_code)
print()

# Show diff summary
summary = diff_summary(buggy_code, patched_code)
print(f"Diff: {summary}")
