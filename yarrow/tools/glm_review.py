#!/usr/bin/env python3
"""
Yarrow Root-Digger — GLM-4.7 Code Review Tool

Sends code to GLM via Cerebras API for structural review.
Uses the Root-Digger harvester prompt for focused analysis.

Usage:
    python yarrow/tools/glm_review.py <file_path> [--focus <area>] [--raw]
    python yarrow/tools/glm_review.py <file1> <file2> ...  # multi-file
    cat file.py | python yarrow/tools/glm_review.py -       # stdin
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = PROJECT_ROOT / "yarrow" / "harvester_glm_prompt.md"
FINDINGS_DIR = PROJECT_ROOT / "yarrow" / "workspace" / "findings"

def load_api_key():
    key = os.environ.get("CEREBRAS_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CEREBRAS_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: CEREBRAS_API_KEY not found in env or .env file", file=sys.stderr)
    sys.exit(1)


def load_system_prompt(focus=None):
    if PROMPT_PATH.exists():
        prompt = PROMPT_PATH.read_text()
    else:
        prompt = "You are a code reviewer focused on structural correctness: types, signatures, API misuse, imports."

    if focus:
        prompt += f"\n\n## Focus for this scan\n{focus}"

    return prompt


def read_files(paths):
    """Read one or more files, return combined content with headers."""
    parts = []
    for p in paths:
        if p == "-":
            content = sys.stdin.read()
            parts.append(f"# stdin\n```python\n{content}\n```")
        else:
            path = Path(p)
            if not path.exists():
                print(f"WARNING: {p} not found, skipping", file=sys.stderr)
                continue
            content = path.read_text()
            rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
            parts.append(f"# {rel}\n```python\n{content}\n```")
    return "\n\n".join(parts)


def call_glm(system_prompt, user_content, api_key, max_tokens=65000):
    """Call Cerebras API with GLM model via SDK."""
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)

    try:
        result = []
        stream = client.chat.completions.create(
            model="zai-glm-4.7",
            stream=True,
            max_completion_tokens=max_tokens,
            temperature=0.3,  # Low temp for precise review
            top_p=0.95,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            result.append(content)
            print(content, end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)  # newline after streaming
        return "".join(result)
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


def save_finding(content, scan_id, target):
    """Save raw GLM response to findings dir."""
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"glm_{scan_id}_{ts}.md"
    out_path = FINDINGS_DIR / filename
    out_path.write_text(f"# GLM Review: {target}\n\nScan: {scan_id}\nTimestamp: {ts}\n\n{content}")
    print(f"\nSaved to: {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Yarrow Root-Digger: GLM code review")
    parser.add_argument("files", nargs="+", help="Files to review (use - for stdin)")
    parser.add_argument("--focus", type=str, default=None, help="Specific focus area for review")
    parser.add_argument("--raw", action="store_true", help="Skip system prompt, use raw review")
    parser.add_argument("--scan-id", type=str, default="manual", help="Scan ID for tracking")
    parser.add_argument("--save", action="store_true", help="Save findings to workspace")
    parser.add_argument("--max-tokens", type=int, default=65000, help="Max response tokens")
    args = parser.parse_args()

    api_key = load_api_key()

    if args.raw:
        system_prompt = "You are an expert code reviewer. Analyze the code for bugs, type errors, API misuse, and structural issues. Be specific — cite line numbers and show evidence."
    else:
        system_prompt = load_system_prompt(args.focus)

    code_content = read_files(args.files)
    if not code_content:
        print("No files to review", file=sys.stderr)
        sys.exit(1)

    user_msg = f"Review the following code. Output your findings as structured JSON per the schema in your instructions.\n\n{code_content}"

    print(f"Sending to GLM-4.7 ({len(user_msg)} chars)...", file=sys.stderr)
    t0 = time.time()
    result = call_glm(system_prompt, user_msg, api_key, args.max_tokens)
    elapsed = time.time() - t0
    print(f"Response received in {elapsed:.1f}s ({len(result)} chars)", file=sys.stderr)

    print(result)

    if args.save:
        target = ", ".join(args.files)
        save_finding(result, args.scan_id, target)


if __name__ == "__main__":
    main()
