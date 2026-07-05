# dark-factory

```
        *                    .                          ☾                    .
   .           .                       *                          .
                    .                            .        *              *
        ▐█▌            ▐█▌                ▐█▌
        ▐█▌ ░░         ▐█▌ ░░░            ▐█▌ ░░
        ▐█▌            ▐█▌                ▐█▌
   ┌────███────────────███────────────────███──────────────────────────────┐
   │ /\      /\      /\      /\      /\      /\      /\      /\      /\    │
   │/  \    /  \    /  \    /  \    /  \    /  \    /  \    /  \    /  \   │
   ├────────────────────────────────────────────────────────────────────────┤
   │                                                                        │
   │   ██████╗  █████╗ ██████╗ ██╗  ██╗                                     │
   │   ██╔══██╗██╔══██╗██╔══██╗██║ ██╔╝                                     │
   │   ██║  ██║███████║██████╔╝█████╔╝                                      │
   │   ██║  ██║██╔══██║██╔══██╗██╔═██╗                                      │
   │   ██████╔╝██║  ██║██║  ██║██║  ██╗                                     │
   │   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝                                     │
   │   ███████╗ █████╗  ██████╗████████╗ ██████╗ ██████╗ ██╗   ██╗          │
   │   ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝          │
   │   █████╗  ███████║██║        ██║   ██║   ██║██████╔╝ ╚████╔╝           │
   │   ██╔══╝  ██╔══██║██║        ██║   ██║   ██║██╔══██╗  ╚██╔╝            │
   │   ██║     ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║   ██║             │
   │   ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝             │
   │                                                                        │
   │      ▓▓░░▓▓     ▓▓░░▓▓     ▓▓░░▓▓     ▓▓░░▓▓     ▓▓░░▓▓     ▓▓░░▓▓     │
   └────────────────────────────────────────────────────────────────────────┘
      ══╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════
        ║          ║          ║          ║          ║          ║

              t h e   l i g h t s   a r e   o f f .
              t h e   l i n e   i s   r u n n i n g .
```

*Codename: **moonflower** — the night-blooming datura: powerful, careful-handling-required,
and it only opens in the dark. Its entry in the apothecary's garden ledger; the working name
stays dark-factory.*

An autonomous implementer loop for Claude Code: a ranked task queue, a build agent, a
risk-tiered panel of adversarial verifiers, and a release ritual — running unattended while
the humans (and the planner thread) sleep. Extracted from a real project where it shipped a
working queue end-to-end; rebuilt project-neutral, with the contract's high-risk rules moved
from prompt-discipline into **hooks the loop cannot ignore**.

## What's in the box

```
dark-factory/
├── skills/dark-factory/
│   ├── SKILL.md                     ← the loop itself (/dark-factory)
│   └── references/
│       ├── config.md                ← every {{variable}}, key by key
│       ├── config.template.json     ← per-project config the setup step fills in
│       ├── two-thread-contract.md   ← the operating doctrine (planner ↔ factory)
│       ├── QUEUE.template.md        ← the ranked work order
│       └── DECISIONS.template.md    ← the blocked-on-human surface
├── agents/                          ← the roster
│   ├── dark-factory-implementer.md  ← builds ONE task file, exactly as specced
│   ├── dark-factory-judge.md        ← adversarial reviewer, ONE lens, refutation-first
│   ├── dark-factory-flow-verifier.md← runs the real flow, inspects the artifacts
│   └── dark-factory-visual-judge.md ← boots the app headless, screenshots, JUDGES
├── hooks/                           ← the guard rail (armed/disarmed at runtime)
│   ├── hooks.json                   ← wiring (PreToolUse / PostToolUse / Stop)
│   ├── lane_guard.py                ← single-writer lanes, enforced both directions
│   ├── bash_gate.py                 ← arm/disarm · no-push wall · fail-closed commit gate
│   ├── stop_gate.py                 ← no ending the turn armed: ritual, then disarm
│   └── run_log.py                   ← JSONL telemetry for the morning reconcile
└── docs/
    ├── RESEARCH.md                  ← the comparative review that shaped v0.2
    └── examples/                    ← live queue/decisions/playbook files from the source project
```

## The assembly line

```
  {{paths.queue}} ──► pick top unblocked task
        │
        ▼
  dark-factory-implementer ── task file = the contract; smallest faithful build
        │
        ▼
  fan-out verify ──── NEVER self-review; panel sized by task risk
        ├── dark-factory-judge ×1–4   (parallel; lenses: correctness / fail-closed /
        │                              regression / security — refute or list attacks)
        ├── dark-factory-flow-verifier (run the real pass on a fixture copy;
        │                              task-file claims = test cases)
        └── dark-factory-visual-judge (headless boot + Playwright + screenshot
                                       judgment; findings with evidence, never pass/fail)
        │
        ▼
  verdict file ────── .dark-factory/verdicts/current.json — green + fresh, or no commit
        │
        ▼
  release ritual ──── version bump · changelog · commit · tag · build · smoke
        │                               (git gate open only behind a green verdict;
        │                                push always denied)
        ▼
  banner the task ─── dated STATUS line 1, mapped to the changelog, never guessed
        │
        └──► next task … until only human decisions remain, then wrap up, disarm, STOP
```

## Hooks, not honor

v0.1 enforced its contract by prompt; v0.2 arms a guard rail. `true # dark-factory-arm`
binds the running session as the implementer; from then on:

| High-risk action | Guard | Outcome |
|---|---|---|
| Out-of-lane write (either thread) | `lane_guard` | denied with the lane map |
| `git commit` / `git tag` without a fresh green verdict | `bash_gate` | denied, fail closed |
| `git push` | `bash_gate` | denied, always |
| Ending the turn without the wrap-up ritual | `stop_gate` | blocked until bannered, handed off, clean, disarmed |
| Any mutation | `run_log` | appended to `.dark-factory/run-<date>.jsonl` |

Disarmed (or in projects with no `.dark-factory/config.json`), every hook is a fast no-op —
the plugin costs nothing outside factory runs.

## Portability — variables, not paint

Nothing project-specific lives in the skill or agents. Every specific — test command,
version file, fixture path, invariants, protected volumes, lane map, risk tiers — is a
`{{variable}}` resolving from `.dark-factory/config.json`. First run of `/dark-factory` in
a fresh repo walks the owner through writing that file from the template; see
`skills/dark-factory/references/config.md` for the full key reference.

## The three rules that make it work

1. **Single-writer areas.** The planner thread writes tasks/docs/prototypes; the factory
   writes code/changelog/version. Nobody crosses. *(hook-enforced)*
2. **Never self-review.** The context that wrote a change is its weakest reviewer. Every
   load-bearing change faces separate adversarial verifiers with named lenses — and
   objective checks (suite, artifact diffs, real execution) outrank judge opinion.
   *(commit-gated)*
3. **Decisions are walls.** Anything needing a human lands as a recommendation-first row
   in DECISIONS and the factory stops cleanly — it never builds on a guessed decision.
   *(stop-gated)*

Full doctrine: `skills/dark-factory/references/two-thread-contract.md`.
