# Yarrow — Backend Reference

## Z.AI Backend (paid, Lite plan $27/qtr)
- Anthropic-compatible endpoint: `https://api.z.ai/api/anthropic`
- Key: `ZAI_API_KEY` in `.env` (key name "Apothecary")
- Uses `anthropic` Python SDK with `base_url` override
- Two API ecosystems: International (api.z.ai, Anthropic SDK) vs China (open.bigmodel.cn, OpenAI SDK + CNY)

### Working on Lite (confirmed 2026-02-17)

| Model | Short | Notes |
|-------|-------|-------|
| GLM-4.7 | glm47 | ~17s/file, structured JSON |
| GLM-4.7-Flash | glm47f | Faster, slight quality drop |
| GLM-4.6 | glm46 | Available |
| GLM-4.6V / 4.5V | glm46v/glm45v | Vision — works on Lite |
| GLM-4.5-Air/Flash | glm45air/glm45f | Budget options |

**Blocked on Lite:** GLM-5 (Pro $81/qtr), media models (OCR, Image, ASR, CogView, Vidu)

## Cerebras Backend (free)
- OpenAI-compatible `https://api.cerebras.ai/v1`
- Key: `CEREBRAS_API_KEY` in `.env` (personal account)
- `gpt-oss-120b` — usable reviewer, has reasoning tokens
- `llama3.1-8b` — canary/smoke-test only

## Review Guidance
- Always send full files, not snippets (models flag `...` truncation as bugs)
- All reviewers have domain blind spots — always validate findings against codebase context

## claude-glm (Zlaude)
- Claude Code pointed at Z.AI endpoint — see `docs/claude-glm-setup.md`
