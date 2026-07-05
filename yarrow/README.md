# Yarrow — Multi-Agent Orchestration System

**Named for Achillea (yarrow)** — a compound flower where dozens of tiny blooms work as one.
Used in the I Ching for decision-making. Coordinates the body's healing response.

## What It Is

Multi-model agent system for code review, bug hunting, and implementation.
Multiple LLM agents (different providers) work independently, their findings
are distilled by an orchestrator, and implementation is delegated to a
dedicated executor. No agent crosses role boundaries.

## The Apothecary's Workshop

| Role | Name | Model(s) | Purpose | Edits Code? |
|------|------|----------|---------|-------------|
| **Orchestrator** | The Apothecary | Claude Opus | Validate, prioritize, prescribe | NO |
| **Assistant** | The Apprentice | Haiku / Sonnet | Deduplicate, cluster, score | NO |
| **Executor** | The Compounder | Claude Opus / Gemini | Implement prescriptions | YES |

## The Harvesters

| Name | Model | Territory | Key Strength |
|------|-------|-----------|-------------|
| **Wildcrafter** | Gemini | Data flow, race conditions, edge cases | Creative, lateral |
| **Simpler** | Claude | Logic errors, JAX pitfalls, math | Deep reasoning |
| **Root-Digger** | GLM | Types, signatures, API misuse | Structural precision |
| **Herb-Walker** | Kimi K2 | Duplication, dead code, patterns | Broad coverage |
| **Distiller** | DeepSeek | Algorithms, numerics, invariants | Proof by execution trace |
| **Physick** | OpenAI | Security, resilience, error handling | System-level diagnosis |
| **Chronicler** | MiniMax | Cross-file consistency, drift | Long-context cross-reference |

## Operational Modes

```
PLAN MODE  ──→  APPROVAL  ──→  YOLO MODE  ──→  PLAN MODE
(research)      (you decide)    (autonomous)     (cycle complete)
```

User controls gear shifts. The Apothecary never self-promotes to YOLO.

## Prompts

| Agent | Prompt File |
|-------|------------|
| Apothecary | [orchestrator_prompt.md](orchestrator_prompt.md) |
| Apprentice | [assistant_prompt.md](assistant_prompt.md) |
| Compounder | [executor_prompt.md](executor_prompt.md) |
| Wildcrafter | [harvester_gemini_prompt.md](harvester_gemini_prompt.md) |
| Simpler | [harvester_claude_prompt.md](harvester_claude_prompt.md) |
| Root-Digger | [harvester_glm_prompt.md](harvester_glm_prompt.md) |
| Herb-Walker | [harvester_kimi_prompt.md](harvester_kimi_prompt.md) |
| Distiller | [harvester_deepseek_prompt.md](harvester_deepseek_prompt.md) |
| Physick | [harvester_openai_prompt.md](harvester_openai_prompt.md) |
| Chronicler | [harvester_minimax_prompt.md](harvester_minimax_prompt.md) |

## Reference Docs

- Architecture: [architecture.md](architecture.md)
- Agent Definitions: [agents.md](agents.md)
- Workflow & Modes: [workflow.md](workflow.md)
- Communication Protocol: [communication.md](communication.md)
