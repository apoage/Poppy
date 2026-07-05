#!/usr/bin/env python3
"""
Yarrow Harvest — Multi-Backend Code Review Tool

Supports Cerebras (OpenAI-compat) and Z.AI (Anthropic-compat) backends.

Usage:
    # Cerebras models (free, OpenAI API)
    python yarrow/tools/harvest.py <file> --model gpt-oss-120b
    python yarrow/tools/harvest.py <file> --model llama3.1-8b

    # Z.AI GLM models (paid Lite plan, Anthropic API)
    python yarrow/tools/harvest.py <file> --model GLM-4.7          # Best quality, ~17s/file
    python yarrow/tools/harvest.py <file> --model GLM-4.7-Flash    # Fast, ~5s/file
    python yarrow/tools/harvest.py <file> --model GLM-4.5-Air      # Budget option

    # Multi-file, with focus
    python yarrow/tools/harvest.py file1.py file2.py --model GLM-4.7 --focus "race conditions" --save
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINDINGS_DIR = PROJECT_ROOT / "yarrow" / "workspace" / "findings"

# Backend configs
BACKENDS = {
    "cerebras": {
        "type": "openai",
        "base_url": "https://api.cerebras.ai/v1",
        "key_env": "CEREBRAS_API_KEY",
    },
    "zai": {
        "type": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic",
        "key_env": "ZAI_API_KEY",
    },
}

# Model → backend auto-detection (case-insensitive lookup)
# Only models confirmed working on Lite plan as of 2026-02-17
MODEL_BACKEND = {
    # Cerebras (free tier)
    "gpt-oss-120b": "cerebras",
    "llama3.1-8b": "cerebras",
    # Z.AI GLM — confirmed working on Lite plan (Anthropic-compatible endpoint)
    "glm-4.7": "zai",
    "glm-4.7-flash": "zai",
    "glm-4.6": "zai",
    "glm-4.6v": "zai",           # vision model
    "glm-4.6v-flash": "zai",     # vision model (fast)
    "glm-4.5": "zai",
    "glm-4.5v": "zai",           # vision model
    "glm-4.5-air": "zai",
    "glm-4.5-flash": "zai",
    # Blocked on Lite plan: GLM-5, FlashX/AirX variants, GLM-4-Plus, media gen models
}

# Model → harvester prompt mapping
HARVESTER_PROMPTS = {
    # Z.AI GLM family — all use same harvester prompt
    "glm-4.7": "harvester_glm_prompt.md",
    "glm-4.7-flash": "harvester_glm_prompt.md",
    "glm-4.6": "harvester_glm_prompt.md",
    "glm-4.6v": "harvester_glm_prompt.md",
    "glm-4.6v-flash": "harvester_glm_prompt.md",
    "glm-4.5": "harvester_glm_prompt.md",
    "glm-4.5v": "harvester_glm_prompt.md",
    "glm-4.5-air": "harvester_glm_prompt.md",
    "glm-4.5-flash": "harvester_glm_prompt.md",
    # Cerebras
    "gpt-oss-120b": "harvester_openai_prompt.md",
    "llama3.1-8b": "harvester_claude_prompt.md",
}

# Short names for output files
MODEL_SHORT = {
    # Cerebras
    "gpt-oss-120b": "gpt-oss",
    "llama3.1-8b": "llama",
    # Z.AI GLM — confirmed working
    "glm-4.7": "glm47",
    "glm-4.7-flash": "glm47f",
    "glm-4.6": "glm46",
    "glm-4.6v": "glm46v",
    "glm-4.6v-flash": "glm46vf",
    "glm-4.5": "glm45",
    "glm-4.5v": "glm45v",
    "glm-4.5-air": "glm45air",
    "glm-4.5-flash": "glm45f",
}


def load_api_key(backend_name):
    cfg = BACKENDS[backend_name]
    key = os.environ.get(cfg["key_env"])
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith(f"{cfg['key_env']}="):
                return line.split("=", 1)[1].strip()
    print(f"ERROR: {cfg['key_env']} not found in env or .env", file=sys.stderr)
    sys.exit(1)


def load_system_prompt(model, focus=None, raw=False):
    if raw:
        return "You are an expert code reviewer. Analyze the code for bugs, type errors, API misuse, and structural issues. Be specific — cite line numbers and show evidence."

    model_lower = model.lower()
    prompt_file = HARVESTER_PROMPTS.get(model_lower)
    if prompt_file:
        prompt_path = PROJECT_ROOT / "yarrow" / prompt_file
        if prompt_path.exists():
            prompt = prompt_path.read_text()
        else:
            prompt = _default_prompt()
    else:
        prompt = _default_prompt()

    if focus:
        prompt += f"\n\n## Focus for this scan\n{focus}"
    return prompt


def _default_prompt():
    return "You are a code reviewer focused on structural correctness: types, signatures, API misuse, imports. Output findings as JSON with fields: file, line_start, line_end, type, confidence, severity, description, evidence, suggested_fix."


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


def call_model_anthropic(model, system_prompt, user_content, api_key, base_url, max_tokens=16000):
    """Call via Anthropic-compatible API (Z.AI)."""
    import anthropic

    short = MODEL_SHORT.get(model.lower(), model)
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    try:
        result = []
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for text in stream.text_stream:
                result.append(text)
                print(text, end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        return "".join(result)
    except Exception as e:
        print(f"[{short}] API error: {e}", file=sys.stderr)
        return None


def call_model_openai(model, system_prompt, user_content, api_key, base_url, max_tokens=16000):
    """Call via OpenAI-compatible API (Cerebras)."""
    from openai import OpenAI

    short = MODEL_SHORT.get(model.lower(), model)
    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        result = []
        stream = client.chat.completions.create(
            model=model,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.95,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                result.append(content)
                print(content, end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        return "".join(result)
    except Exception as e:
        print(f"[{short}] API error: {e}", file=sys.stderr)
        return None


def call_model(model, backend_name, system_prompt, user_content, api_key, max_tokens=16000):
    cfg = BACKENDS[backend_name]
    if cfg["type"] == "anthropic":
        return call_model_anthropic(model, system_prompt, user_content, api_key, cfg["base_url"], max_tokens)
    else:
        return call_model_openai(model, system_prompt, user_content, api_key, cfg["base_url"], max_tokens)


def save_finding(content, model, scan_id, target):
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    short = MODEL_SHORT.get(model.lower(), model)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{short}_{scan_id}_{ts}.md"
    out_path = FINDINGS_DIR / filename
    out_path.write_text(
        f"# {short} Review: {target}\n\nModel: {model}\nScan: {scan_id}\nTimestamp: {ts}\n\n{content}"
    )
    print(f"\nSaved to: {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Yarrow Harvest: multi-backend code review")
    parser.add_argument("files", nargs="+", help="Files to review (- for stdin)")
    parser.add_argument("--model", type=str, default="GLM-4.7", help="Model ID")
    parser.add_argument("--backend", type=str, default=None, help="Force backend (cerebras/zai)")
    parser.add_argument("--focus", type=str, default=None, help="Specific focus area")
    parser.add_argument("--raw", action="store_true", help="Skip harvester prompt")
    parser.add_argument("--scan-id", type=str, default="manual", help="Scan ID")
    parser.add_argument("--save", action="store_true", help="Save to workspace")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max response tokens")
    args = parser.parse_args()

    # Auto-detect backend from model name (case-insensitive)
    backend = args.backend or MODEL_BACKEND.get(args.model.lower())
    if not backend:
        print(f"Unknown model '{args.model}'. Use --backend to specify.", file=sys.stderr)
        sys.exit(1)

    api_key = load_api_key(backend)
    system_prompt = load_system_prompt(args.model, args.focus, args.raw)
    code_content = read_files(args.files)

    if not code_content:
        print("No files to review", file=sys.stderr)
        sys.exit(1)

    short = MODEL_SHORT.get(args.model.lower(), args.model)
    user_msg = f"Review the following code. Find bugs, type errors, shape mismatches, API misuse, and structural issues. Cite line numbers.\n\n{code_content}"

    print(f"[{short}] Sending to {backend} ({len(user_msg)} chars)...", file=sys.stderr)
    t0 = time.time()
    result = call_model(args.model, backend, system_prompt, user_msg, api_key, args.max_tokens)
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
