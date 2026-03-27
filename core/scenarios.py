"""Predefined bug scenario library for PatchDuel benchmarks."""
from __future__ import annotations

SCENARIOS: list[dict] = [
    {
        "id": "sc-001",
        "name": "Arithmetic Operator",
        "description": "Subtraction used instead of addition",
        "tags": ["arithmetic", "easy"],
        "language": "python",
        "buggy_code": "def add(a, b):\n    return a - b  # Bug: should be a + b",
        "expected_fix": "def add(a, b):\n    return a + b",
    },
    {
        "id": "sc-002",
        "name": "Off-by-One Loop",
        "description": "Loop iterates one extra time causing IndexError",
        "tags": ["loop", "off-by-one", "medium"],
        "language": "python",
        "buggy_code": (
            "def sum_list(items):\n"
            "    total = 0\n"
            "    for i in range(len(items) + 1):  # Bug: +1 causes IndexError\n"
            "        total += items[i]\n"
            "    return total"
        ),
        "expected_fix": (
            "def sum_list(items):\n"
            "    total = 0\n"
            "    for i in range(len(items)):\n"
            "        total += items[i]\n"
            "    return total"
        ),
    },
    {
        "id": "sc-003",
        "name": "Wrong Guard Return",
        "description": "Guard clause returns the divisor instead of a constant zero",
        "tags": ["guard", "logic", "easy"],
        "language": "python",
        "buggy_code": (
            "def safe_divide(a, b):\n"
            "    if b == 0:\n"
            "        return b  # Bug: should return 0\n"
            "    return a / b"
        ),
        "expected_fix": (
            "def safe_divide(a, b):\n"
            "    if b == 0:\n"
            "        return 0\n"
            "    return a / b"
        ),
    },
    {
        "id": "sc-004",
        "name": "Missing None Check",
        "description": "Function crashes when passed None",
        "tags": ["none", "guard", "medium"],
        "language": "python",
        "buggy_code": (
            "def get_length(text):\n"
            "    return len(text)  # Bug: crashes if text is None"
        ),
        "expected_fix": (
            "def get_length(text):\n"
            "    if text is None:\n"
            "        return 0\n"
            "    return len(text)"
        ),
    },
    {
        "id": "sc-005",
        "name": "Wrong Logical Operator",
        "description": "'and' used where 'or' is needed in boundary check",
        "tags": ["logic", "boolean", "medium"],
        "language": "python",
        "buggy_code": (
            "def is_valid_age(age):\n"
            "    # Bug: 'and' means both conditions must be true simultaneously\n"
            "    if age < 0 and age > 150:\n"
            "        return False\n"
            "    return True"
        ),
        "expected_fix": (
            "def is_valid_age(age):\n"
            "    if age < 0 or age > 150:\n"
            "        return False\n"
            "    return True"
        ),
    },
    {
        "id": "sc-006",
        "name": "Mutable Default Argument",
        "description": "Default list shared across all calls — a classic Python gotcha",
        "tags": ["python-gotcha", "hard"],
        "language": "python",
        "buggy_code": (
            "def append_item(item, items=[]):  # Bug: mutable default arg\n"
            "    items.append(item)\n"
            "    return items"
        ),
        "expected_fix": (
            "def append_item(item, items=None):\n"
            "    if items is None:\n"
            "        items = []\n"
            "    items.append(item)\n"
            "    return items"
        ),
    },
    {
        "id": "sc-007",
        "name": "String vs Integer Comparison",
        "description": "User input stays as string, compared to integer — always False",
        "tags": ["types", "comparison", "medium"],
        "language": "python",
        "buggy_code": (
            "def check_answer(user_input):\n"
            "    \"\"\"user_input comes from input() — it's always a string.\"\"\"\n"
            "    if user_input == 42:  # Bug: '42' != 42\n"
            "        return 'Correct!'\n"
            "    return 'Wrong answer.'"
        ),
        "expected_fix": (
            "def check_answer(user_input):\n"
            "    \"\"\"user_input comes from input() — it's always a string.\"\"\"\n"
            "    if int(user_input) == 42:\n"
            "        return 'Correct!'\n"
            "    return 'Wrong answer.'"
        ),
    },
    {
        "id": "sc-008",
        "name": "Wrong String Method",
        "description": "strip() called on result of split() — TypeError at runtime",
        "tags": ["strings", "medium"],
        "language": "python",
        "buggy_code": (
            "def parse_tags(tag_string):\n"
            "    \"\"\"Split comma-separated tags and strip whitespace.\"\"\"\n"
            "    return tag_string.split(',').strip()  # Bug: list has no .strip()"
        ),
        "expected_fix": (
            "def parse_tags(tag_string):\n"
            "    \"\"\"Split comma-separated tags and strip whitespace.\"\"\"\n"
            "    return [t.strip() for t in tag_string.split(',')]"
        ),
    },
]


def get_all_scenarios() -> list[dict]:
    """Return all predefined scenarios."""
    return SCENARIOS


def get_scenario_by_id(scenario_id: str) -> dict | None:
    """Return a scenario by its ID, or None if not found."""
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    return None


def get_scenarios_by_tag(tag: str) -> list[dict]:
    """Return all scenarios that have the given tag."""
    return [s for s in SCENARIOS if tag in s.get("tags", [])]


def scenario_choices() -> list[tuple[str, str]]:
    """Return (label, value) tuples suitable for a Gradio Dropdown."""
    _ICONS = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
    choices: list[tuple[str, str]] = [("✏️  Custom — paste your own code", "custom")]
    for s in SCENARIOS:
        tags = s.get("tags", [])
        icon = next((_ICONS[t] for t in ("easy", "medium", "hard") if t in tags), "⚪")
        label = f"{icon} [{s['id']}] {s['name']} — {s['description']}"
        choices.append((label, s["id"]))
    return choices


def mock_patch(buggy_code: str) -> str:
    """
    Return a mock patch for demo/offline mode.
    Matches known scenario buggy code → returns the expected fix.
    Falls back to returning the input unchanged with a demo comment.
    """
    for s in SCENARIOS:
        if s["buggy_code"].strip() == buggy_code.strip():
            return s["expected_fix"]
    # Unknown code: return as-is so the diff shows "no changes"
    return buggy_code
