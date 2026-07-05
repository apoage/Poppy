# CROSSVALIDATE 764 — bug-hunt verdicts

Read-only adversarial cross-check of `tasks/BUGHUNT_764_2026-05-23.md`.
All claims re-derived from current source (post `2ee38f7` + `72146f8`).

## Per-finding verdicts (F1-F6)

### F1: CONFIRMED

- **Independent evidence**: `easy_config.py:153-159` defines `__post_init__` with `assert self.oracle_k > 0` at line 158. Per dataclass spec `__post_init__` runs exactly once at construction. `train_easy.py:3049-3051` is a direct attribute assignment AFTER `config = get_<preset>_config()` (i.e. after `__post_init__` already ran). No re-validate hook is called. Argparse declares `type=int, default=None` only — no `choices=`, no custom validator.
- **Confirmed bypass paths**: `--oracle-k 0` (`0 is not None`, override applied, assertion never re-runs); `--oracle-k -1` (same); `--oracle-k 65` (passes through to `jax.lax.top_k(arr, k=65)` — invalid because `n_assets=64`, would be caught by XLA but with an opaque error).
- **Recommended fix verdict**: The hunt's 1-liner `if args.oracle_k is not None: assert 0 < args.oracle_k <= config.n_assets, ...; config.oracle_k = args.oracle_k` is the cleanest option *for this codebase*. The "property setter on EasyConfig" alternative would be cleaner architecturally but is high-blast-radius (the dataclass is also constructed via `EasyConfig(**dict)` for JSON loads, presets, Phase-2 inheritance — a property setter would need to dance with `field()` defaults). The "call `config.__post_init__()` after override" alternative is one line shorter but couples train_easy to the implementation detail of the dataclass. Hunt's recommendation stands.
- **Wider pattern (see "New findings" §1)**: F1 is the most operator-reachable instance of a broader `__post_init__`-bypass pattern. The hunt should generalise but the F1 fix-text covers the immediate hazard.

### F2: CONFIRMED

- **Independent evidence**: `tools/orchestration/e-dojo_v1.py:1228-1236` (`DEFAULT_PARAMS`) — no `oracle_k`. `:1276-1326` (`launch_branch` cmd builder) — explicit `cmd = [...]` list with the exact forwarded flags the hunt enumerates. The only kwargs-driven extensions are `--ar-max-blend` (state-driven, `:1291-1294`), `--oracle-multipass` (`:1320-1321`), benchmark flags (`:1324-1326`), and historical/periods args (`:1306-1317`). No `**kwargs` → `cmd.extend` fallback. A `grep -n "oracle_k" tools/orchestration/e-dojo_v1.py` returns ZERO hits.
- **Call chain verified**: `launch_single_branch` (`:1661-1701`) → `launch_branch(**kwargs)` (`:1682-1688`); `**kwargs` is propagated as Python-call kwargs but `launch_branch` never reads `kwargs['oracle_k']` so it's silently dropped at the cmd-build step.
- **CLAUDE.md confirms dojo is advertised**: `tools/orchestration/CLAUDE.md` lists `e-dojo_v1.py` as "Active (edit freely) — Easy-mode dual-branch (Bella + Synthetic). Current research sandbox" and includes it in "Entry commands" (single-branch + dual-bella variants). Operator could absolutely reach for it.
- **Impact claim**: If a user runs `python tools/orchestration/e-dojo_v1.py --run --single --preset large --oracle-k 8 ...` — argparse at `:1980-1995` would actually error out (`--oracle-k` is not a registered flag in the e-dojo parser; `unrecognized arguments` is the failure mode). The "silent k=4 void" scenario requires either (a) an operator who hand-edits `DEFAULT_PARAMS` to add `oracle_k=8` (won't propagate without the cmd-builder edit), or (b) a script that calls `launch_branch(..., oracle_k=8)` programmatically (would silently drop). Either way the bug is real, but the silent failure path is narrower than the hunt implies — the argparse layer catches the most likely operator-error.
- **Recommended fix verdict**: Both hunt-suggested fixes are valid. Fix (b) — `logger.info(f"[CONFIG] oracle_k={config.oracle_k}")` unconditional at startup in `train_easy.py` — is the more defensive option because it surfaces the actual k regardless of launch path (currently the log fires only on explicit CLI override at `:3051`). Recommend (b) over (a) if only one is shipped; ideally both.

### F3: CONFIRMED

- **Independent evidence**: `train_phase2.py:2570-2580` — `if p1_config_path.exists(): config = get_phase2_config(...)` else `config = Phase2Config(**filtered)`. `get_phase2_config` (`phase2_config.py:297-307`) filters by `easy_fields = {f.name for f in EasyConfig.__dataclass_fields__.values()}` then `Phase2Config(**p1_filtered)`. Since `Phase2Config(EasyConfig)` (`phase2_config.py:14`) inherits all `EasyConfig` fields including `oracle_k: int = 4` (`easy_config.py:64`), a missing `oracle_k` key in `p1_data` will silently default to 4. Same for the fallback branch (`Phase2Config.__dataclass_fields__` includes inherited fields).
- **Log message is technically misleading**: `"[TASK_764] oracle_k inherited from Phase-1 checkpoint = {config.oracle_k}"` will read "= 4" both when the ckpt explicitly says 4 AND when the ckpt has no `oracle_k` field at all. The very ambiguity the log was added to prevent (k=4 vs k=8/12 silent mismatch) is unresolved when the ckpt is missing the field entirely.
- **Practical-risk verification (2 ckpts checked)**:
  - `jax_ckpt_easy_phase1_xs32/easy_medium/config.json:30` — `"oracle_k": 4` present
  - `jax_ckpt_easy_phase1_v3/easy_medium/config.json:30` — `"oracle_k": 4` present
  - These are the only two `config.json` files under `*/jax_ckpt*/` paths on the workstation. Both have the field. `easy_trainer.py:440` writes `self.config.__dict__` to the ckpt pickle, and `EasyConfig.__dict__` since TASK_666 has always included `oracle_k`. So the practical risk is genuinely low for ckpts produced by this repo. Risk lives only in (i) externally produced ckpts, (ii) hand-edited `config.json`, (iii) corrupted ckpt files where the pickle deserialises but the dict is mangled.
- **Recommended fix verdict**: Hunt's `k_in_p1 = p1_config.get('oracle_k', '<MISSING — default 4>')` rewrite is sound. Note one subtlety the hunt doesn't address: when `p1_config_path.exists()`, the function reads the JSON file but the log probes `p1_config` (the dict from the pickle, loaded by `load_phase1_checkpoint`). If JSON and pickle disagree, the log message would say one value while `config.oracle_k` (built from JSON via `get_phase2_config`) says another. The fix should probe whichever source actually built `config` for an honest report — i.e. inside both branches.

### F4: CONFIRMED (cosmetic, accurate scope)

- **Independent evidence**: `train_easy.py:2900` instantiates `argparse.ArgumentParser(description=...)` with no `allow_abbrev=False`. Default in Python 3.5+ is `allow_abbrev=True`.
- **Enumerated `--oracle-*` flags** (lines 2958, 2970, 2972):
  - `--oracle-multipass` (action='store_true')
  - `--oracle-mask-head` (action='store_true')
  - `--oracle-k` (type=int)
- **Ambiguity analysis**:
  - `--oracle 8` → ambiguous (matches prefix of all three) → argparse exits with `ambiguous option` error. The error message argparse produces lists all candidates, so the user CAN see `--oracle-k` as one of the options.
  - `--oracle-m 8` → ambiguous between `--oracle-multipass` and `--oracle-mask-head`. `--oracle-k` is NOT a candidate (the `-k` suffix is distinct).
  - `--oracle-k 8` → unambiguous, works as intended.
- **Verdict**: F4's "cosmetic only" assessment is accurate. No false-positive bug, just operator-UX opacity if someone abbreviates too aggressively. Not 764-introduced.

### F5: CONFIRMED + EXPANDED (hunt missed a second site)

- **Independent evidence**: `train_jax_fast.py:619-621` — `compute_optimal_path(jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0)` inside `on_episode_complete_callback` (TASK_125 expert imprinting). Live code, but only fires when `getattr(config, 'imprint', False)` is True.
- **Status as dead-code vs sibling**: This is the Poppy training path (per `CLAUDE.md` top-level "train_jax_fast.py — Poppy training entry (router)" and `mcts_trader/CLAUDE.md` "Common commands"). It's a sibling entry point, NOT dead. A future user running Poppy training with imprinting would silently use k=4. Out of scope for 764 (Foxglove Phase-1), but the hunt's flag is correct.
- **NEW: hunt missed `mcts_trader/training/self_play.py:818-821`**. Identical call site: `compute_optimal_path(jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0)` inside `episode_optimal_alpha` computation. Same out-of-scope status (Poppy lineage), same issue. Both are siblings to F5 — listed under "New findings".
- **Inert function-signature defaults**: Hunt correctly identifies `train_easy.py:88, 188, 327` as inert (every live call passes `k=self.config.oracle_k`). Independently re-verified: `compute_optimal_path` itself in `path_solver.py` also has `k: int = 4` as its top-level default — making the explicit `k=4` at `train_jax_fast.py:620` and `self_play.py:820` redundant inert overrides. They could omit `k=` entirely with no behavioural change.

### F6: CONFIRMED (by-design)

- **Independent evidence**: `train_easy.py:122-131` — `n_active = jnp.sum(jnp.abs(pos) > 0.5); capacity = n_active / k; out = jnp.stack([..., jnp.full(n_assets, capacity)])`. `train_easy.py:231-238` (model-state rollout) — identical formulation. Channel 32 IS `n_active / k`.
- **Distribution implication**: When oracle picks exactly k assets every step (the canonical case), `n_active = k` and the channel saturates at 1.0 always. At k=8/12 the channel is more informative because the oracle may select fewer than k when min_alpha_threshold filters out marginal picks. The hunt's by-design framing is correct — k flows through generically, but the encoder learns a different channel-32 distribution at different k. This is downstream of the experiment hypothesis (different k = different label density), not a bug.
- **Verdict**: Informational, not actionable. Worth keeping in experiment notes.

## New findings (missed by the hunt)

### N1 (P3): F1 is a specific instance of a broader `__post_init__`-bypass pattern

`easy_config.py:153-159` validates FOUR invariants in `__post_init__`:
1. `action_space_size == 5 * n_assets` — no direct CLI override of either field, BUT `config.n_assets` is mutated indirectly in some code paths (worth checking; not exercised by the 764 patch).
2. `0 < target_oracle_agreement <= 1.0` — directly overridden by `train_easy.py:3046-3047` (`config.target_oracle_agreement = args.target_agreement` after `if args.target_agreement is not None`). `--target-agreement 1.5` or `-0.1` bypasses the validator silently.
3. `oracle_k > 0` — F1.
4. `batch_size > 0` — directly overridden by `train_easy.py:3022-3023` (`if args.batch_size: config.batch_size = args.batch_size`). Note the truthy check (`if args.batch_size`) — `--batch-size 0` skips override (default behaviour preserved), but `--batch-size -1` passes and bypasses the validator.

If the F1 fix is shipped (re-validate via `__post_init__()` call or per-field guard), it should plausibly cover the other two operator-facing fields (`target_agreement`, `batch_size`) — one `config.__post_init__()` call after all the overrides would re-run all four assertions at once.

### N2 (P3): Sibling out-of-scope hardcoded `k=4` in `mcts_trader/training/self_play.py:820`

Sibling of F5. Same call signature, same lineage (Poppy), same out-of-scope status. Hunt enumerated F5 but missed this one. Listing for completeness in case TASK_764 ever expands beyond Phase-1.

### N3 (info): F2 silent-drop scenario narrower than hunt frames

The hunt suggests an operator who reaches for `python tools/orchestration/e-dojo_v1.py --run --single ... --oracle-k 8` would void the experiment silently. In reality, the e-dojo's own argparse (`:1980-1995`) does not register `--oracle-k`, so the CLI would error with `unrecognized arguments: --oracle-k 8`. The silent-drop path requires:
- Programmatic call to `launch_branch(..., oracle_k=8)` from Python (would silently drop), OR
- A future code change that adds `--oracle-k` to e-dojo's argparse but forgets to forward it via the cmd builder.

Risk is real but bounded to "code-side regression in e-dojo" rather than "operator misuse". The hunt's recommended unconditional `logger.info` in `train_easy.py` still covers both cases — recommend.

### N4 (info): `train_jax_fast.py:620` and `self_play.py:820` are inert-default OVERRIDES, not just signature defaults

Because `compute_optimal_path`'s own default is `k=4` (`path_solver.py:def compute_optimal_path(returns, k=4, method='greedy', **kwargs)`), the explicit `k=4` at both call sites is redundant. Removing the explicit `k=4` would be cosmetic — same behaviour. Worth a one-line clean-up in any future Poppy-side TASK_764 expansion (would make the sites greppable as "uses default k" rather than "hardcodes k=4").

### N5 (verified clean — explicit confirmation requested by spec)

- **CLI help string vs behaviour**: `--oracle-k` help reads `"TASK_764: oracle top-k (sweep {4,8,12}); default None = use preset value"`. Behaviour matches: `default=None`, override only when not None, preset/`--config` preserved otherwise. No drift.
- **`oracle_cfg` dict pickup at `train_easy.py:3157-3166`**: re-traced. Line 3050 override → line 3158 `'oracle_k': config.oracle_k` read. Override sequenced strictly before dict construction. Saved `config.json` (line 3327) reflects the actual k used.

## Verified-clean re-confirmed

All of the hunt's "Verified clean" list re-verified against source:

- Override placement / ordering — re-checked `train_easy.py:3049-3051` < `:3157` (oracle_cfg) < `:3228` (`OnTheFlyGenerator`) < `:3327` (`config.save`). Correct.
- `default=None` invariant — re-checked. `args.oracle_k` is only read inside the `if args.oracle_k is not None:` guard.
- Phase-2 inheritance when ckpt has the key — re-verified. `Phase2Config(EasyConfig)` carries `oracle_k`. Both `get_phase2_config` and the fallback branch produce the same result. `easy_trainer.py:440` writes the value round-trips correctly.
- `k_count` per-step normalisation — re-verified. `path_solver.py:130` and `train_phase2.py:629` both compute `k_count = jnp.maximum(jnp.sum(mask), 1.0)` / `np.maximum(...)`. Adapts to k automatically.
- `n_active / k` in `build_state_channels` — re-verified. Static-arg `k` flows through every call site.
- JIT `static_argnames=['k', ...]` — re-verified at `path_solver.py:38, 182, 219, 342, 484, 580` and `train_easy.py:83, 181, 320`. New k triggers fresh JIT compile.
- `jax.lax.top_k(masked_returns, k)` — re-verified. Documented to support any k ≤ array length; valid for any k ≤ `n_assets`.
- No disk-cached oracle — re-verified. `_precompute_oracle` runs `greedy_lookahead_path` on-the-fly with current k. No invalidation needed.
- `oracle_cfg` dict in `train_phase2.py:452-490` — re-verified. Inside `load_segment_pool(config: Phase2Config, ...)`, called from `main()` after config inheritance + `[TASK_764] oracle_k inherited from ...` log fires. Reads `config.oracle_k` consistently.
- Phase-2 inheritance log placement — re-verified at `train_phase2.py:2580`. Both `if/else` branches (`:2570-2576`) assign `config` before the log fires.
- Benchmark scripts pick up `config.oracle_k` from ckpt — re-verified at `tools/benchmark_easy.py:506, 607, 973, 1122` and `tools/benchmark_oracle_vs_model.py:239, 624`.

## Summary

**Verdict on hunt**: All 6 findings (F1-F6) **CONFIRMED** under independent re-derivation. The hunt is sound, well-evidenced, and the recommended fixes are appropriate for this codebase.

**Net new findings**: 1 P3 (N1 — F1 is part of a broader `__post_init__`-bypass pattern covering also `target_agreement` and `batch_size`); 1 P3 (N2 — sibling hardcoded `k=4` at `self_play.py:820`); 2 info-level (N3, N4). No new P0 or P2 findings.

**One nuance the hunt over-states**: F2's "silent void via dojo CLI" scenario. The e-dojo's argparse rejects `--oracle-k` outright (unrecognized arg), so the silent-drop path requires programmatic kwarg use or a future broken e-dojo edit. Hunt's recommended unconditional `logger.info` in `train_easy.py` is still the right fix and covers both N3 and the original F2 risk.

**Blocking status**: Hunt's "None of the P2s block running the 764 sweep IF the operator launches Phase-1 directly with a sane `--oracle-k {4,8,12}`" — agreed. P2 guardrails should be tightened opportunistically; no hard block on the 764 experiment as currently scoped.
