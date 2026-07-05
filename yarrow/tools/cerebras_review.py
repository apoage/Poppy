#!/usr/bin/env python3
"""
Yarrow Harvester — Cerebras Multi-Model Code Review Tool

Sends code to any Cerebras-hosted model for structural review.

Usage:
    python yarrow/tools/cerebras_review.py <file> --model gpt-oss-120b
    python yarrow/tools/cerebras_review.py <file> --model llama3.1-8b
    python yarrow/tools/cerebras_review.py <file> --model zai-glm-4.7
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINDINGS_DIR = PROJECT_ROOT / "yarrow" / "workspace" / "findings"

# Model → harvester prompt mapping
HARVESTER_PROMPTS = {
    "zai-glm-4.7": PROJECT_ROOT / "yarrow" / "harvester_glm_prompt.md",
    "gpt-oss-120b": PROJECT_ROOT / "yarrow" / "harvester_openai_prompt.md",
    "llama3.1-8b": PROJECT_ROOT / "yarrow" / "harvester_claude_prompt.md",  # Generic structural review
}

# Short names for output
MODEL_NAMES = {
    "zai-glm-4.7": "glm",
    "gpt-oss-120b": "gpt-oss",
    "llama3.1-8b": "llama",
    "qwen-3-235b-a22b-instruct-2507": "qwen",
}


def load_api_key():
    key = os.environ.get("CEREBRAS_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CEREBRAS_API_KEY="):
                return line.split("=", 1)[1].strip()
    print("ERROR: CEREBRAS_API_KEY not found", file=sys.stderr)
    sys.exit(1)


def load_system_prompt(model, focus=None, raw=False):
    if raw:
        prompt = "You are an expert code reviewer. Analyze the code for bugs, type errors, API misuse, and structural issues. Be specific — cite line numbers and show evidence."
    else:
        prompt_path = HARVESTER_PROMPTS.get(model)
        if prompt_path and prompt_path.exists():
            prompt = prompt_path.read_text()
        else:
            prompt = "You are a code reviewer focused on structural correctness: types, signatures, API misuse, imports. Output findings as JSON with fields: file, line_start, line_end, type, confidence, severity, description, evidence, suggested_fix."

    if focus:
        prompt += f"\n\n## Focus for this scan\n{focus}"
    return prompt


def read_files(paths):
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


def call_model(model, system_prompt, user_content, api_key, max_tokens=16000):
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)
    short = MODEL_NAMES.get(model, model)

    try:
        result = []
        stream = client.chat.completions.create(
            model=model,
            stream=True,
            max_completion_tokens=max_tokens,
            temperature=0.3,
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
        print(file=sys.stderr)
        return "".join(result)
    except Exception as e:
        print(f"[{short}] API error: {e}", file=sys.stderr)
        return None


def save_finding(content, model, scan_id, target):
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    short = MODEL_NAMES.get(model, model)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{short}_{scan_id}_{ts}.md"
    out_path = FINDINGS_DIR / filename
    out_path.write_text(f"# {short} Review: {target}\n\nModel: {model}\nScan: {scan_id}\nTimestamp: {ts}\n\n{content}")
    print(f"\nSaved to: {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Yarrow: Cerebras multi-model code review")
    parser.add_argument("files", nargs="+", help="Files to review (use - for stdin)")
    parser.add_argument("--model", type=str, default="gpt-oss-120b", help="Model ID")
    parser.add_argument("--focus", type=str, default=None, help="Specific focus area")
    parser.add_argument("--raw", action="store_true", help="Skip harvester prompt, use generic review")
    parser.add_argument("--scan-id", type=str, default="manual", help="Scan ID for tracking")
    parser.add_argument("--save", action="store_true", help="Save findings to workspace")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max response tokens")
    args = parser.parse_args()

    api_key = load_api_key()
    system_prompt = load_system_prompt(args.model, args.focus, args.raw)
    code_content = read_files(args.files)

    if not code_content:
        print("No files to review", file=sys.stderr)
        sys.exit(1)

    short = MODEL_NAMES.get(args.model, args.model)
    user_msg = f"Review the following code. Find bugs, type errors, shape mismatches, API misuse, and structural issues. Cite line numbers.\n\n{code_content}"

    print(f"[{short}] Sending ({len(user_msg)} chars)...", file=sys.stderr)
    t0 = time.time()
    result = call_model(args.model, system_prompt, user_msg, api_key, args.max_tokens)
    elapsed = time.time() - t0

    if result:
        print(f"[{short}] Done in {elapsed:.1f}s ({len(result)} chars)", file=sys.stderr)
        print(result)
        if args.save:
            save_finding(result, args.model, args.scan_id, ", ".join(args.files))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
