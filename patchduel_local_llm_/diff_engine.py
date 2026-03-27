"""Diff computation and HTML rendering using diff-match-patch."""
import html
from diff_match_patch import diff_match_patch

_dmp = diff_match_patch()


def compute_char_diff(original: str, patched: str) -> list:
    """Character-level diff list: [(op, text), ...]  op: -1 del, 0 equal, 1 insert."""
    diffs = _dmp.diff_main(original, patched)
    _dmp.diff_cleanupSemantic(diffs)
    return diffs


def compute_line_diff(original: str, patched: str) -> list:
    """
    Line-level diff for more readable code comparisons.
    Returns list of (op, text) tuples.
    """
    a_chars, b_chars, line_array = _dmp.diff_linesToChars(original, patched)
    diffs = _dmp.diff_main(a_chars, b_chars, False)
    _dmp.diff_charsToLines(diffs, line_array)
    return diffs


def render_html_diff(original: str, patched: str) -> str:
    """
    Render a git-style colored HTML diff between original (buggy) and patched code.
    Lines prefixed with '+' are additions (green), '-' are deletions (red).
    """
    diffs = compute_line_diff(original, patched)

    rows: list[str] = []
    for op, text in diffs:
        # text may contain multiple lines; split and handle each
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # Skip the trailing empty string produced by a trailing newline
            if i == len(lines) - 1 and line == "":
                continue
            escaped = html.escape(line)
            if op == 1:   # insertion
                rows.append(
                    f'<div class="diff-line diff-add">'
                    f'<span class="diff-gutter">+</span>'
                    f'<span class="diff-text">{escaped}</span>'
                    f'</div>'
                )
            elif op == -1:  # deletion
                rows.append(
                    f'<div class="diff-line diff-del">'
                    f'<span class="diff-gutter">−</span>'
                    f'<span class="diff-text">{escaped}</span>'
                    f'</div>'
                )
            else:           # equal
                rows.append(
                    f'<div class="diff-line diff-eq">'
                    f'<span class="diff-gutter">&nbsp;</span>'
                    f'<span class="diff-text">{escaped}</span>'
                    f'</div>'
                )

    content = "\n".join(rows) if rows else '<div class="diff-line diff-eq"><span class="diff-text">(no changes)</span></div>'

    return f"""
<style>
  .diff-block {{
    font-family: 'Courier New', Courier, monospace;
    font-size: 13px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    overflow-x: auto;
    padding: 8px 0;
    margin: 4px 0;
  }}
  .diff-line {{
    display: flex;
    white-space: pre;
    line-height: 1.5;
    padding: 0 12px;
  }}
  .diff-add  {{ background: #0d2a1a; color: #3fb950; }}
  .diff-del  {{ background: #2a0d0d; color: #f85149; }}
  .diff-eq   {{ color: #8b949e; }}
  .diff-gutter {{
    display: inline-block;
    width: 16px;
    min-width: 16px;
    user-select: none;
    opacity: 0.8;
    margin-right: 8px;
  }}
  .diff-text {{ flex: 1; }}
</style>
<div class="diff-block">
{content}
</div>
"""


def diff_stats(original: str, patched: str) -> dict:
    """Return insertion/deletion character counts and whether any change occurred."""
    diffs = compute_char_diff(original, patched)
    insertions = sum(len(t) for op, t in diffs if op == 1)
    deletions = sum(len(t) for op, t in diffs if op == -1)
    return {
        "insertions": insertions,
        "deletions": deletions,
        "changed": insertions > 0 or deletions > 0,
    }


def diff_summary(original: str, patched: str) -> str:
    """Human-readable one-liner summary of the diff."""
    s = diff_stats(original, patched)
    if not s["changed"]:
        return "No changes"
    parts = []
    if s["insertions"]:
        parts.append(f"+{s['insertions']} chars added")
    if s["deletions"]:
        parts.append(f"−{s['deletions']} chars removed")
    return "  |  ".join(parts)
