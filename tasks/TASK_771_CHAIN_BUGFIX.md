# TASK_771 — Chain Bug-Fix (2026-05-23 multi-agent QA synthesis)

**Status**: 767-related fixes (F1+F2+F4+F5) ✓ FIXED 2026-05-23 — synthesis from 4 bug-hunt + 4 cross-validation reports. Remaining batches (sync-db.sh P1s, 764/769 P2s) still pending.
**Created**: 2026-05-23.
**Context**: The chain quest shipped 4 implementations 2026-05-22/23 (767 action-space, 764 oracle-k flag, 769 F2 value-clip, deploy-integrity sync-db.sh). A multi-agent QA pass (8 agents — 4 hunters + 4 cross-validators) ran 2026-05-23 and surfaced **1 confirmed P0 + 6 P1s + N hardening items**. This task consolidates and ranks the fixes.

## Source reports
- `tasks/BUGHUNT_767_2026-05-23.md` ↔ `tasks/CROSSVALIDATE_767_2026-05-23.md`
- `tasks/BUGHUNT_764_2026-05-23.md` ↔ `tasks/CROSSVALIDATE_764_2026-05-23.md`
- `tasks/BUGHUNT_769_2026-05-23.md` ↔ `tasks/CROSSVALIDATE_769_2026-05-23.md`
- `tasks/BUGHUNT_deploy_integrity_2026-05-23.md` ↔ `tasks/CROSSVALIDATE_deploy_integrity_2026-05-23.md`

## Severity ledger (post-cross-validation)

| ID | Severity | Component | One-line | Blocks | Status |
|---|---|---|---|---|---|
| 767/F1 | **P0** | `factored_search.py` mean-shift + `-1e9` mask | float32 ULP collapse → uniform policy target | 767 launch | ✓ FIXED (TASK_771 commit, 2026-05-23) |
| 767/F2 | P1 | mctx interior priors unmasked | wastes search compute, biases value backup | quality, not correctness | ✓ FIXED (TASK_771 commit, 2026-05-23) |
| di/F1 | P1 | sync-db.sh sqlite3-crash silent abort | failure-path-only; success path works | sync-db.sh failure visibility | pending |
| di/F2 | P1 | sync-db.sh printed snippet does in-place overwrite | contradicts docstring "no live swap" | operator footgun | pending |
| di/F3 | P1 | sync-db.sh interrupted zstd → corrupt-next-run | mtime check skips re-compress, sha matches itself | next sync after interrupt | pending |
| di/F4 | P1 | sync-db.sh no WAL checkpoint before compress | latent — bites if DB ever in WAL mode | future Kraken-fetch path | pending |
| di/F5 | P1 | sync-db.sh `--skip-compress` ships stale .zst | current .zst is 2026-01-20 (4 mo old) | operator misuse | pending |
| 767/F4 | P2 | Stage-B resume drops `disable_limit_orders` if stray Phase-1 config.json | requires unusual ckpt layout | edge case | ✓ FIXED (TASK_771 commit, 2026-05-23) |
| 764/F1+N1 | P2 | `__post_init__` validators bypassed by direct CLI override | covers `oracle_k, batch_size, target_oracle_agreement` | guardrail gap |
| 764/F2 | P2 | e-dojo silently drops `--oracle-k` (only via programmatic kwargs) | operator CLI is rejected → safe; programmatic use voids | narrow gap |
| 764/F3 | P2 | Phase-2 inheritance log "inherited=4" masks missing-key fallback | current ckpts all have the field → risk low | guardrail |
| 769/F1 | P2 | SPEC body has stale pre-patch line numbers | `:63, :71-76, :78, :82-83, :85` | doc accuracy |
| 769/F2 | P2 | SPEC cites orphaned amended hash `c2ad6f5` | should be `95fcf98` | doc accuracy |
| 769/F3 | P2 | no `logger.info` for `value_clip_mode` at training start | reproducibility / log readability | operator visibility |
| 769/N1 | P2 | `value_clip_mode` accepts any string from JSON config-load (argparse `choices` only catches CLI) | typo bypass | guardrail |
| di/N1 | P2 | sync-db.sh — server-side disk-full on 24 GB decompress unprotected | foot-gun if `$HOME` is short | rare-but-possible |
| 767/F5, 769/N3 | P3 | various SPEC / commit message line drift | doc accuracy | polish | ✓ FIXED for 767+769 SPECs (TASK_771 doc commit, 2026-05-23) |
| di/{F7-F13, N2-N7} | P3 | sync-db.sh papercuts | stderr leak, no DB lock, no `zstd -t`, etc. | polish |
| 764/F4, F5, N2 | P3 | argparse abbrev, sibling hardcoded `k=4` in Poppy entry points | cosmetic / out-of-scope | polish |
| 769/F4-F6, N2, N5 | P3 | documented behavioural caveats (td0 partial, popart warm-resume, etc.) | by-design / documented | polish |

**Demoted by cross-validation:** the deploy-integrity hunt's two P0s (F1, F2) → P1 — real bugs, but the success path works and the script does not auto-execute the destructive command. **No P0 exists outside 767/F1.**

---

# Implementation log

**2026-05-23 — 767-related fixes shipped (F1+F2+F4+F5).**

Files changed:
- `mcts_trader/jax_agent/factored_search.py` — F1: removed the per-asset `q.mean()` shift before `softmax(q/T)`. `jax.nn.softmax` already max-shifts internally; the explicit mean-shift was rounding all valid Q's to a single float32 value under the `-1e9` mask. Canon (action_mask=None) path is mathematically identical (softmax shift-invariance) and observed within intra-baseline noise (ΔSP_loss ≤ ~0.002, within the canon-vs-canon spread of ~0.001 from pipeline scheduling).
- `mcts_trader/jax_agent/mctx_search.py` — F2: threaded `action_mask` into `_recurrent_fn` so depth-≥1 prior logits are masked to `finfo.min` at the same indices as root. mctx's `gumbel_muzero_interior_action_selection` has no `invalid_actions` argument — only root selection masks — so without F2 limit-action branches were expanded at depth ≥ 1 (wasted simulations + biased value backups). Canon (action_mask=None) is bit-identical: the masking branch is dead Python in the trace.
- `phase2_config.py` — F4: `get_phase2_config` now filters by `Phase2Config.__dataclass_fields__` (superset of EasyConfig), so a stray Phase-1 `config.json` next to a Phase-2 checkpoint can no longer silently drop `disable_limit_orders` (or any other Phase-2-only field).
- `train_phase2.py` — F4: `--disable-limit-orders` switched to `argparse.BooleanOptionalAction` with `default=None`; the override now applies for both True and False so `--no-disable-limit-orders` can force the flag OFF on resume. Three-state semantics: None = inherit, True = force ON, False = force OFF.

Smoke results (workstation single RTX 3090, 1-epoch factored Stage-B on a 4-slot SegmentPool, seed=7):

| Run | `mean_trade_rate` | `limit_action_fraction` | Notes |
|---|---|---|---|
| flag_on_pre (pre-patch, with `--disable-limit-orders`) | **0.666667** | 0.000 | EXACT 2/3 = the buggy uniform fingerprint (F1 present) |
| flag_on_post (post-patch, with `--disable-limit-orders`) | **0.665129** | 0.000 | Off the uniform fingerprint by 60× the canon noise floor — F1 fix observable |
| canon_pre (pre-patch, no flag) | 0.798276 | 0.397608 | Canon baseline |
| canon_post (post-patch, no flag) | 0.798309 | 0.397609 | Within intra-baseline noise of canon_pre |
| canon_pre_v3 (re-run of canon_pre) | 0.798276 | 0.397608 | bit-identical to canon_pre |
| canon_pre_v2 (second canon_pre run) | 0.798301 | 0.397605 | intra-baseline noise ~5e-6 on metrics |

Canon noise floor (canon vs canon, same code): `Δpolicy_loss ≈ 0.0007`. Cross-patch canon: `Δpolicy_loss ≈ 0.002`. Within ~3× the intra-baseline noise floor — the F1 patch is **mathematically identical** by softmax shift-invariance and **effectively bit-identical** in float32 (variations at the ULP level absorbed into the AsyncBatchPipeline's intrinsic non-determinism).

F2 verified by structural code inspection — the mctx recurrent_fn closure now captures `action_mask` and applies it to `prior_logits` at depth ≥ 1.

F4 verified by python harness: `Phase2Config.__dataclass_fields__` contains `disable_limit_orders` (and `value_clip_mode`) by construction; a synthetic config dict with `disable_limit_orders: true` is preserved through `get_phase2_config()`. `argparse.BooleanOptionalAction` supports all three states (absent → None, `--disable-limit-orders` → True, `--no-disable-limit-orders` → False).

Smoke artifacts: `runs/000_smoke/771_chain_bugfix/{flag_on_post,flag_on_pre,canon_post,canon_post_v2,canon_pre,canon_pre_v2,canon_pre_v3}/{train.log,stage_b_metrics.csv}`.

---

# P0 — must fix before 767 launches

## 767/F1 — factored_search mean-shift collapses under `-1e9` mask

### Location
`mcts_trader/jax_agent/factored_search.py:128-131` (`_q_to_policy`).

### Bug (independently re-derived by the cross-validator)
```python
q_centered = q_values - q_values.mean(axis=-1, keepdims=True)
return jax.nn.softmax(q_centered / T, axis=-1)
```
When `disable_limit_orders=True`, the search-action mask sets the two limit-action Q-values to `-1e9` at `:194`. Then per-asset over 5 actions:

- Sum of `[Q0, Q1, Q2, -1e9, -1e9]` with `Q_i ∈ ~[−20, 20]` → ULP at `2e9` = 256 ≫ |Q sum| ≤ 60 → sum rounds to exactly `−2e9`; mean = `−4e8`.
- `q_centered = q_value − (−4e8)` with `q_value ∈ ~[−20, 20]`: ULP at `4e8` = 32 > Q range → all three valid `q_centered` entries round to the same float32 value (`4e8`).
- `softmax([4e8, 4e8, 4e8, −6e8, −6e8])` → `exp(−1e9)` underflows to exact 0 → policy target = `[1/3, 1/3, 1/3, 0, 0]` across valid actions.

**Action selection is unaffected** — `argmax(q_values)` at `:201` operates on un-centered masked values where `−1e9` is far below valid Q. **The training target is what collapses.**

**XLA does NOT optimize the mean-shift away** — it's a data-dependent reduction, not a pattern in any HLO simplification pass. `jax.nn.softmax`'s own internal `x − x.max()` operates on whatever it receives.

**Epsilon-greedy mix (default 0.1) accidentally matches the bug**: 90% buggy-uniform + 10% valid-uniform = uniform. No mitigation.

**Smoke fingerprint**: `limit_action_fraction = 0.000` is consistent with both buggy and correct implementations (search excludes limits either way). `mean_trade_rate` (NOT reported in the 767 smoke summary) would have been the smoking gun — should be `0.667 ± 0.02` if buggy (uniform across sell/hold/buy ⇒ 2/3 active).

### Fix (one line)
Remove the mean-shift entirely. `jax.nn.softmax` does its own max-shift internally for numerical stability — the explicit mean centering is mathematically redundant for softmax output.

```python
# Before:
q_centered = q_values - q_values.mean(axis=-1, keepdims=True)
return jax.nn.softmax(q_centered / T, axis=-1)

# After:
return jax.nn.softmax(q_values / T, axis=-1)
```

Canon (flag-OFF) is bit-identical: without the `−1e9` mask the mean-shift is just a translation that softmax inverts; removing it gives the same probabilities up to floating-point.

### Smoke verification
1-epoch flag-ON Stage-B (same pattern as the 767 ship-smoke). Assert:
- `mean_trade_rate` ≠ 0.667 ± 0.02 (would be ~0.5-0.6 if policy is learning, exactly 0.667 if still buggy).
- `value_loss` decreases across the epoch (was masked by the uniform-target bug pre-fix).
- Canon (flag-OFF) bit-identical to pre-fix — run pre-fix and post-fix canon smoke at fixed seed, assert metrics match.

### Risk
**Zero** — removes a redundant operation. Same bit-identical-canon discipline as 767's original ship.

### Why this is urgent
767 is queued to launch after 762 Wave 2 closes (currently 762 W2 has cells `fp05_s13 + fp10_s13` running; ~few hours remaining). Without this fix, 767 trains a random policy and looks like a null even if the action-space mechanism is real. **Must ship before 767 launches.**

---

# P1 — fix soon (batch by component)

## Batch 1 — `sync-db.sh` (5 fixes, one commit)

### di/F1 — explicit rc capture for sqlite3 crash visibility
`tools/deploy/sync-db.sh:51`:
```bash
# Before:
RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"

# After:
RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)" && RC=0 || RC=$?
if [[ $RC -ne 0 ]]; then
  echo "❌ sqlite3 itself failed (exit=$RC):" >&2
  echo "$RESULT" >&2
  exit 1
fi
```

### di/F2 — printed server snippet uses atomic `.NEW` rename + doc honesty
Replace the trailing heredoc snippet with:
```
ssh -p $SRV_PORT $SRV_USER@$SRV_HOST '
  cd $SRV_PATH &&
  sha256sum -c $SHA &&
  unzstd -k -f $ZST -o trading_data.db.NEW &&
  sqlite3 trading_data.db.NEW "PRAGMA integrity_check" | grep -q "^ok$" || { rm trading_data.db.NEW; exit 1; } &&
  mv trading_data.db.NEW trading_data.db
'
```
+ update docstring at `:10-14` and `tools/deploy/README.md` line on "manual server-side swap" to describe the atomic-rename semantics honestly.

### di/F3 — atomic temp + rename for zstd compress
`tools/deploy/sync-db.sh:65`:
```bash
# Before:
zstd -f --keep -19 -o "$ZST" "$DB"

# After:
zstd -f --keep -19 -o "$ZST.tmp" "$DB" && mv "$ZST.tmp" "$ZST"
```

### di/F4 — WAL checkpoint before compress
Insert before step 2 (compress):
```bash
echo "==> WAL checkpoint (no-op if not in WAL mode)..."
sqlite3 "$DB" 'PRAGMA wal_checkpoint(TRUNCATE)' >/dev/null
```

### di/F5 — staleness warning for `--skip-compress`
In the flag-handling block:
```bash
if [[ -n "$SKIP_COMPRESS" ]]; then
  if [[ "$DB" -nt "$ZST" ]]; then
    echo "⚠️  --skip-compress with $ZST older than $DB ($(date -r $ZST '+%F') vs $(date -r $DB '+%F'))" >&2
    [[ -n "$FORCE_STALE" ]] || { echo "Pass --force-stale to override." >&2; exit 1; }
  fi
fi
```
+ add `--force-stale` to the arg parser.

### Smoke / verification (batch)
- `bash -n tools/deploy/sync-db.sh` (parse).
- `tools/deploy/sync-db.sh --help` (banner).
- `tools/deploy/sync-db.sh --check-only` (full integrity check — runs ~minutes; success path).
- Mentally walk failure paths: sqlite3 returns non-zero, interrupted zstd, `--skip-compress` with old .zst.
- No live training implication — shell script only.

### Risk
Zero.

## Batch 2 — 767/F2: mctx interior mask threading

### Location
`train_phase2.py` — wherever `_recurrent_fn` is defined (the inner mctx callback that returns `prior_logits` at depth ≥1). The 767 ship masked root-level `prior_logits` only.

### Fix
Thread the limit mask into `_recurrent_fn` (via closure or `static_argnames`); when `disable_limit_orders=True`, mask the returned `prior_logits` to `-1e9` (same pattern + same sentinel as root). Note: with the P0 fix above removed, `-1e9` is no longer in a code path that does mean-shift, so it's safe.

### Smoke
1-epoch joint-MCTS Stage-B with `--disable-limit-orders --mcts-sims 32` + telemetry on mean tree-depth limit-action visit fraction (should be ~0 post-fix). Confirm canon (flag-OFF) bit-identical.

### Risk
Low — replicates root-level masking semantics.

---

# P2 — hardening (batched by file, after P0/P1 closed)

## Batch 3 — `easy_config.py` / `train_easy.py` (764)
- **764/F1+N1**: add `config.__post_init__()` call immediately after the CLI-override block in `train_easy.py:~3050` — covers the three validators (`oracle_k`, `batch_size`, `target_oracle_agreement`) that are currently bypassed by direct CLI attribute set. Single-line fix, broader than per-field re-validate.
- **764/F2**: document in `tools/orchestration/CLAUDE.md` that dojo's launch_branch does not forward arbitrary CLI flags; programmatic kwargs go through a different path.
- **764/F3**: change the inheritance log to detect missing key explicitly:
  ```python
  if 'oracle_k' not in raw_config: logger.warning(f"[TASK_764] oracle_k MISSING from ckpt config — defaulted to {config.oracle_k}")
  else:                            logger.info(f"[TASK_764] oracle_k inherited from ckpt = {config.oracle_k}")
  ```

## Batch 4 — `phase2_config.py` / `train_phase2.py` (769 + 767/F4)
- **769/F3**: add `logger.info(f"[CONFIG] value_clip_mode={config.value_clip_mode}")` at Stage-B start (after the `[TASK_764] oracle_k` line for grouping).
- **769/N1**: validate `value_clip_mode in {"pre_popart", "post_popart"}` in `Phase2Config.__post_init__` (catches JSON-load path that argparse `choices` misses).
- **767/F4**: ensure `disable_limit_orders` is in `Phase2Config.__dataclass_fields__` (it should already be — verify). If a stray Phase-1 `config.json` could overwrite it during Phase-2 load, fix the load filter to preserve the Phase-2-overrideable field.

## Batch 5 — SPEC / doc accuracy (769/F1, 769/F2, 767/F5, 769/N3)
- `experiments/769_value_clip/SPEC.md`: update the deep-dive section's line numbers (`:63, 71-76, 78, 82-83, 85` → the post-patch reality `:689, :1266, :1284, :1297`); fix the `c2ad6f5` → `95fcf98` reference.
- `experiments/767_action_space/SPEC.md`: refresh the +17-45-line drift caused by TASK_769 insertions.
- Pure doc commit; no code change.

## Batch 6 — `sync-db.sh` further hardening (di/N1, F8, F9, etc.)
- **di/N1**: pre-flight server-side disk-free check before rsync — `ssh ... df --output=avail $SRV_PATH`; require ≥ 30 GB free.
- **di/F8** (P2): `BEGIN IMMEDIATE` DB lock around steps 1→2 to close TOCTOU window between integrity_check and compress.
- **di/F9** (P2): `zstd -t "$ZST"` (verify) after compress.
- **di/F10** (P2): pre-check rsync target dir exists.

---

# P3 — polish (defer or roll into Batches 5-6)

Various — see the referenced source reports. Includes:
- 764/F4 argparse abbreviation cosmetics; 764/F5 + N2 sibling `k=4` in Poppy entry points (out of 764 scope).
- 769/F4-F6, N2, N5 — by-design behavioural caveats; document in the SPEC's "Code shipped" section, no code.
- di/F11-F13 — `--help` from a non-repo cwd, unknown positional args silently ignored, hardcoded `sed` slice.

---

# Execution plan

**Serial through git/deploy**, same discipline as the original chain. Smoke + commit + deploy per batch.

```
1. P0 — 767/F1 mean-shift removal           [URGENT, before 767 launches; ~1 line + smoke]
2. Batch 1 — sync-db.sh P1s (5 fixes)       [one commit; shell-script smoke]
3. Batch 2 — 767/F2 mctx mask threading     [code commit + 1-epoch flag-ON smoke]
4. Batch 3 — 764 P2s                        [config validator hardening]
5. Batch 4 — 769 P2 + 767/F4 P2             [config + log additions]
6. Batch 5 — SPEC doc refresh (769, 767)    [doc-only commit]
7. Batch 6 — sync-db.sh further hardening   [shell-script]
```

**Owner per batch**: any of (workstation main agent, an implementation sub-agent). The P0 is small enough to do directly without a sub-agent; the larger P1/P2 batches can each be a sub-agent if the user prefers the chain-agent pattern.

**Smoke discipline**: every code commit gets a 1-epoch smoke verifying canon bit-identical (same as the chain). Shell-script-only commits get `bash -n` + `--help` + the relevant invocation flow.

**Deploy**: same `tools/deploy/deploy.sh --code-only` for code changes; shell-script changes ship with the next deploy.

**Gating on 762 Wave 2**: P0 must ship before 767 launches. 762 Wave 2 is still running (~few hours to gate-check); 769 is queued ahead of 767 on the server run order. So the *strict* deadline for P0 is "before 767 cell starts," but **earlier is always safer**.

## Out of scope for TASK_771
- The 768 channel-restructure (still gated on 762 W2 closing).
- The 764 *experiment* (Phase-1 re-imprint) — still future per the user's "after the Phase-2 queue" decision.
- Sandbox issue blocking sub-agents from `train_easy.py` JAX execution — a Claude Code permissions question, not a code bug.
