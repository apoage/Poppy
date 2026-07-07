# Handoff 2026-07-07 — factory line ready, token-size mandate

**Previous:** `lore/handoff_2026-07-06_vector-action-pivot.md` (seq +1)
**Branch:** master · **Working dir:** /home/lukas/Trader
**Session role:** PLANNER / line designer (operator-corrected mid-session: "you are dark factory
line designer not executor"). Factory = separate session via /dark-factory.
**Session focus:** dark-factory setup + repo prune to the reasoning-trading-system core +
D1–D5 decided + repo-wide emotion strip (cheap-agent fan-out).

## Compaction rubric

```
Compaction rubric (chandoff 2026-07-07):

KEEP IN FULL:
- Role contract: this session = PLANNER (queue/decisions/specs/docs); factory session executes.
  Never implement queue tasks in the planner session.
- Factory line state: .dark-factory/config.json (8 invariants, selftest = green baseline),
  tasks/QUEUE.md (828 St1→2→3 ready, 829 filler, St4 gated server-run), tasks/DECISIONS.md
  (ZERO open rows), fixtures/trading_fixture.db 51MB (regen fixtures/make_fixture.py).
- Decisions: D1 828 sole focus (824/825/826 parked, non-binding); D2 827 substance = promising →
  action-conditional reward head folded into 828 Stage 3 + variance-attribution probe; D3 paper
  timer killed (trader/ stays as test baseline + the +0.97 bar); D4 OHLC-only, 24GB DB is the
  world; D5 Bella stays resident, Foxglove retires via TASK_829.
- TOKEN-SIZE MANDATE (operator, load-bearing): optimize ACTIVE TOKEN SIZE of the codebase, not
  MB. Measured: always-loaded ~8.7k tok; 828 hot set ~135k tok (train_phase2.py 44.6k,
  train_jax_fast.py 23k, jax_trading_env.py 18.6k, benchmark_easy.py 18k); repo py+md 3.78M tok.
  Design consequence: vector-action code lands in NEW SMALL MODULES importing from the monsters,
  never threaded through them; 829 deletes train_easy.py (33k tok).
- Next action: operator will deliver a DETAILED TASK OUTLINE for the actual goal (reasoning
  trading system over OHLC) — that is design/queue-grooming work for the planner.

COMPRESS + POINT:
- Repo prune (~46GB + 23GB stale worktrees; keep-calls: candle_lang wired at train_phase2.py:703+,
  Hindsight-optimal = Sampled-MuZero refs, Bella ckpts) → lore/prune_2026-07-07.md.
- Emotion strip (lore 32 files, tasks/README 8 files; 12 cheap agents; diff-audited, evidence
  untouched) → commits 319ddba + c110c68.
- Lore role reversal (Bloom retired, neutral technical record only) → memory
  feedback_lore_dual_role (rewritten).

DROP:
- Cheap-agent classification tables and per-batch strip reports, worktree unlock mechanics,
  systemd timer commands, fixture-build SQL details.

VERBATIM ANCHORS:
- Commits: 430b4ef (setup+prune) → 0fda1ff (828 factory contract) → 0c92cda (D1–D5 + 829) →
  319ddba (lore strip) → c110c68 (tasks strip). Tree clean, nothing pushed.
- TASK_829 decouple points: train_phase2.py:484 (from train_easy import SegmentPool → real home
  mcts_trader/data/segment_pool.py), tools/benchmark_easy.py:41 (easy_config imports).
- 828 Stage 3 D2-carry: reward head action-conditional; 60× action-variance gap is the probe.

UNCERTAIN (operator: correct me):
- Code-comment cosmetic pass (11 files, mostly functional alarms): recommended SKIP; not
  explicitly ruled on.
- trading_data.db.zst (7.7GB in-repo backup): kept — is it still wanted on this disk?
```

## Git state
```
c110c68 tasks/README/experiments de-dramatized: 8 files, drama-only edits
319ddba Lore de-emotionalized: 32 files stripped of drama/emoji/editorializing
0c92cda Decisions D1-D5 recorded; queue regroomed; TASK_829 retire-Foxglove seeded
0fda1ff TASK_828: factory execution contract (risk tier, fixture, canon gate per stage)
430b4ef Dark-factory setup + repo prune to the reasoning-trading-system core
```
Status: clean. Nothing pushed (owner pushes after review).

## Next action
Receive the operator's detailed task outline for the reasoning trading system; groom it into
queue rows / task specs under the token-size mandate (new small modules, small active sets).
Factory launch (`/dark-factory`, fresh session) can proceed on the existing queue meanwhile.
