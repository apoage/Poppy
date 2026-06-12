# TASK_805 — Server-state reconciliation (pre-deploy gate)

**Status**: 🔍 INVENTORIED 2026-06-11 — **corrected**: NOT a pull-back; the server is stale-HEAD +
franken-tree, all code superseded by workstation master. Blocks the [[TASK_804]] arm deploy.
Distinct from [[TASK_790]] (realized-PnL-value — which turns out to be the *content* on both sides,
see below).

## Corrected finding (the earlier "pull TASK_790 back" framing was WRONG)
The server's `M` files are NOT un-pulled-back server work — they are the **workstation's own
already-committed code**, delivered to the server by direct rsync but sitting **uncommitted on the
server's stale HEAD** (`4879982`, deploy 2026-05-31, which predates the workstation commits).

Inventory (every server item verified present in workstation master):
| server state | what it is | in workstation master? |
|---|---|---|
| `M phase2_config.py / train_phase2.py / train_phase2_cli.py` | TASK_790 A2 (`realized_value_weight`) | **yes — committed `936ce31`** |
| `M tools/benchmark_easy.py` | the 803-P1 gate (my rsync) | yes — `e0ff145` |
| `?? tools/candle_lang/` | 800/801/803 work | yes — 39 tracked files |
| `?? tools/safe_run.sh` | sandbox wrapper | yes — tracked |
| `?? tools/tokenizer_c0/` | C0 tokenizer | yes — 5 tracked files |
| `?? data/`, jax caches, bella manifests | runtime artifacts | excluded by `.training-exclude` |

**Conclusion: nothing on the server is unique. Applying any "pull-back" hunks would DUPLICATE code the
workstation already has.** The workstation is fully authoritative.

## Reconciliation = clean deploy (not a merge)
Bring the server to the workstation's current HEAD. The server's franken working tree is entirely
superseded → **discard it, don't preserve it**:
1. On the server: `git reset --hard` + `git clean -fd` (drops the rsync'd-uncommitted-on-stale-HEAD
   franken state — safe, it's all in workstation master), so the push to the checked-out `training`
   branch isn't rejected by a dirty tree.
2. `tools/deploy/deploy.sh` from the workstation → server `training` branch advances to the
   workstation state; verify server HEAD == the deployed `deploy <ts>` commit and the candle_lang /
   train_phase2 / model files match the workstation.

## OPEN caveat — the WORKSTATION dirty tree (operator decision)
`deploy.sh` ships the workstation **working tree** (uncommitted state included — by design, per
`tools/deploy/README.md`). The workstation has many **pre-existing uncommitted `M` code files**
unrelated to branch C — `mcts_trader/data/history.py`, `env/jax_trading_env.py`,
`jax_agent/path_solver.py`, `diff_teacher/*`, `training/cli.py`, `config/real_trade_config.py`, …
These would ship with the 804 deploy. Two implications:
- "Ship from a **known commit**" is only loosely true while the tree carries these deltas.
- Arm-vs-baseline cleanliness: if any `M` touches code the 760 a1b1 baseline trained under
  (history.py / env / path_solver are all trainer-path), the arms would not be a pure control.
**Operator call before deploy**: ship the working tree as-is (deploy-contract default, if these are
intended dev state) OR commit/snapshot to a clean known commit first. Needs provenance knowledge of
those `M` files.

## Gate status
Reconciliation is **safe and understood** (no server work to lose; workstation authoritative). The
remaining pre-deploy item is the workstation-dirty-tree decision above. The deploy itself executes at
arm-launch on the operator's go.
