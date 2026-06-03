"""result_writer.py — generic RESULT.md + .json sidecar writer for the candle-language panel.

Lifts the markdown formatting idiom from tools/tokenizer_c0/run_gate.py:336-394 into a single
generic ``write_result`` entry point shared by every panel member (L1..L5, L4). Each member hands
us a title, an ordered list of (heading, lines) sections, a verdict string, and a free-form meta
dict; we render a stable RESULT.md AND a machine-readable JSON sidecar.

Public API (contract):
    write_result(out_md_path, title, sections, verdict, meta) -> str
        Writes ``out_md_path`` and ``out_md_path`` with the ``.md`` suffix replaced by ``.json``
        (``json.dump(default=float)``). Creates parent dirs. Prints + returns the .md path written.

No JAX / DB / GPU dependency — pure stdlib, importable on CPU-only hosts.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["write_result"]


def write_result(
    out_md_path: str,
    title: str,
    sections: Sequence[Tuple[str, Sequence[str]]],
    verdict: str,
    meta: Dict[str, Any],
) -> str:
    """Render a RESULT.md + .json sidecar.

    Parameters
    ----------
    out_md_path
        Destination markdown path. The JSON sidecar replaces the ``.md`` suffix with ``.json``
        (if ``out_md_path`` does not end in ``.md`` the sidecar simply appends ``.json``).
    title
        H1 title line (no leading ``#`` — we add it).
    sections
        Ordered list of ``(heading, lines)`` pairs. Each ``heading`` is rendered as an H2
        (``## heading``); each entry of ``lines`` is written verbatim as one markdown line.
    verdict
        Short verdict token (e.g. ``"PASS"`` / ``"FAIL"`` / ``"PROCEED"`` / ``"KILL"``). Rendered
        in a dedicated ``## VERDICT`` section and stored in the JSON sidecar.
    meta
        Arbitrary machine-readable payload (config, metrics, etc.). Serialized into the JSON
        sidecar under ``"meta"`` and rendered as a final ``## Meta`` key/value section.

    Returns
    -------
    str
        The markdown path written (also printed).
    """
    generated = time.strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    L = lines.append

    L(f"# {title}")
    L("")
    L(f"_Generated {generated} — candle-language panel (branch C)_")
    L("")

    for heading, body in sections:
        L(f"## {heading}")
        for ln in body:
            L(str(ln))
        L("")

    L("## VERDICT")
    L(f"### **{verdict}**")
    L("")

    if meta:
        L("## Meta")
        for k, v in meta.items():
            L(f"- **{k}**: {_fmt_meta_value(v)}")
        L("")

    out_dir = os.path.dirname(os.path.abspath(out_md_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_path = _json_path_for(out_md_path)
    payload = {
        "title": title,
        "generated": generated,
        "verdict": verdict,
        "sections": [
            {"heading": heading, "lines": [str(ln) for ln in body]}
            for heading, body in sections
        ],
        "meta": meta,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print(f"wrote {out_md_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    return out_md_path


def _json_path_for(out_md_path: str) -> str:
    """Sidecar path: replace a trailing ``.md`` with ``.json``, else append ``.json``."""
    if out_md_path.endswith(".md"):
        return out_md_path[: -len(".md")] + ".json"
    return out_md_path + ".json"


def _fmt_meta_value(v: Any) -> str:
    """Compact, human-readable rendering of a meta value for the markdown body."""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt_meta_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=float, sort_keys=True)
    return str(v)
