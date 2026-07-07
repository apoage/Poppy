# Agent permissions — Trader project

This file is the canonical spec for what each project-scoped agent can read and write. It documents both **hard enforcement** (tool allowlist in frontmatter) and **soft discipline** (system prompt path scoping). Update this file when you change either.

## Universal rules (apply to every agent including the main thread)

**Read:** anywhere in the repo, plus standard read locations (home dir, /tmp). Already permitted via `Read(//home/lukas/**)` in `settings.local.json`.

**Write — always forbidden, regardless of agent:**
- `.env*` files anywhere in the repo
- `.git/objects/**` (would corrupt the storage disk symlinked git store)
- Any path outside the repo unless explicitly scoped (e.g. `~/.claude/skills/` is intentional, `~/some-other-project/` is not)
- `trader_env/` — virtualenv, regeneratable, never edit-in-place
- `trading_data.db` — the production sqlite; touch only via explicit migration scripts, never agent Edit/Write

**Bash — always forbidden:**
- `rm -rf` on anything outside `/tmp` or `runs/<id>/` (and even there, ask first)
- Anything that pushes to git remotes (the user pushes)
- Anything that touches deploy infrastructure without going through `tools/deploy/deploy.sh`

## Per-agent scope

### campaign-shipper

**Tools (hard):** `Read, Edit, Write, Grep, Glob, Bash`

**Write — allowed:**
- Production code paths the brief names — typically `mcts_trader/**`, `diff_teacher/**`, `tools/**`, top-level `*.py` (`train_jax_fast.py`, `train_easy.py`, `easy_trainer.py`, `real_trade_jax.py`)
- Commit messages (via `git commit`)

**Write — forbidden:**
- `lore/**` — that's `lore-writer`'s lane
- `tasks/**` — task specs have their own discipline (`chain-quest` updates status fields only)
- `experiments/**` — that's `chain-quest`'s lane
- `.claude/**` — config, not code
- `bloom/**`, `yarrow/**` — side projects, not in scope for trading campaigns
- Anything not named in the brief's file:line ranges (scope creep)

**Bash — allowed:**
- `git add`, `git commit` (already in project allowlist)
- Smoke commands the brief names
- `python tools/...` for verification

**Bash — forbidden:**
- Long-running training (`train_phase2.py`, `train_jax_fast.py` for >5min)
- Deploy commands
- `git push`

---

### qa-hunt

**Tools (hard):** `Read, Grep, Glob, Bash, WebFetch, WebSearch`

**Write — none.** Layer 1 enforces: `Edit` / `Write` / `NotebookEdit` are not in the allowlist. Cannot be prompt-injected around.

**Bash — allowed (read-only inspection only):**
- `git log`, `git diff`, `git show`, `git status`
- `grep`, `rg`, `cat`, `head`, `tail`, `wc`, `ls`, `find`
- `python -c '<read-only inspection>'`

**Bash — forbidden (prompt discipline, since Bash is the tool):**
- Any state-changing command (`git add`, `git commit`, `mv`, `cp` to a non-tmp path, `rm`, `python <script>` that produces files)
- Training runs of any duration
- Network mutations

---

### qa-crossval

**Tools (hard):** `Read, Grep, Glob, Bash, WebFetch, WebSearch`

**Write — none.** Same Layer 1 enforcement as `qa-hunt`.

**Bash:** same allowed/forbidden as `qa-hunt`.

---

### lore-writer

**Tools (hard):** `Read, Write, Edit, Grep, Glob, Bash`

**Write — allowed:**
- `lore/**` — primary output
- `bloom/**` — secondary output when the lore section is being shaped for publication

**Write — forbidden:**
- All code paths (`mcts_trader/`, `diff_teacher/`, `tools/`, top-level `*.py`)
- `tasks/**` — task specs
- `experiments/**` — campaign artifacts
- `.claude/**`
- `~/.claude/projects/*/memory/**` — that's a separate artifact with its own discipline; lore is not memory

**Bash — allowed:**
- `git log`, `git diff`, `git status`, `git show` — for capturing concrete anchors into lore
- `ls`, `cat`, `head` of files being summarized
- Date / pwd / branch detection

**Bash — forbidden:**
- Anything that modifies repo state
- Any training, any deploy

---

### chain-quest

**Tools (hard):** `Read, Write, Edit, Grep, Glob, Bash, Agent`

**Write — allowed (orchestration artifacts only):**
- `experiments/<id>/SPEC.md`
- `experiments/<id>/AUTHORIZATION.md`
- `experiments/QUEUE.md`
- `tasks/TASK_*.md` — **status fields only** (e.g. `Status: SHIPPED → DEPLOYED`); not body rewrites
- `lore/**` — only via spawned `lore-writer` agent, not directly

**Write — forbidden:**
- All production code paths — spawn `campaign-shipper` instead. If chain-quest finds itself about to Edit `mcts_trader/foo.py`, that's a failure of orchestration.
- `.claude/**`
- `bloom/**`, `yarrow/**`

**Agent — allowed:**
- `campaign-shipper` (for the ship step)
- `qa-hunt` (parallel for QA step)
- `qa-crossval` (parallel per finding)
- `lore-writer` (when the orchestrator needs lore shaping rather than just status updates)

**Agent — forbidden:**
- Spawning another `chain-quest` (no recursion)
- Spawning anything not in the project agents directory

**Bash — allowed:**
- `git log`, `git diff`, `git status` — for SPEC anchoring
- `ls`, `cat`, `head` for inspection
- Reading run results (`cat runs/<id>/result.md`)

**Bash — forbidden:**
- Training runs (delegated to shipper or the human at deploy boundary)
- Deploy commands
- Direct file modification via `mv`, `cp`, `echo >`, etc. — use Edit/Write tools, which respect the per-agent write rules above

---

## Optional Layer 2 guard — global denylist

If you want belt-and-suspenders enforcement of "always forbidden" paths above, add this to `.claude/settings.local.json` under `permissions`:

```json
"deny": [
  "Edit(/home/lukas/Trader/.env*)",
  "Write(/home/lukas/Trader/.env*)",
  "Edit(/home/lukas/Trader/trader_env/**)",
  "Write(/home/lukas/Trader/trader_env/**)",
  "Edit(/home/lukas/Trader/trading_data.db)",
  "Write(/home/lukas/Trader/trading_data.db)",
  "Edit(/home/lukas/Trader/.git/objects/**)",
  "Write(/home/lukas/Trader/.git/objects/**)",
  "Bash(git push:*)",
  "Bash(rm -rf:*)"
]
```

This applies to **every** agent including the main thread. Adds a refusal at tool-call time rather than relying on prompt discipline. Reversible — just remove the entries.

Not auto-applied; review before adding.

## When this file disagrees with an agent's SKILL.md

This file is the canonical reference. If an agent prompt and this file disagree, fix the prompt — but update this file first if the rule itself needs to change. Tool allowlists in the agent frontmatter must match the "Tools (hard)" line here.
