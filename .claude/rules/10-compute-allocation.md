---
description: How to launch any job — cgroup caps, VRAM envelope, tmux, GPU selection. Applies when touching training, experiment, or tooling code.
paths:
  - "experiments/**"
  - "tools/**"
  - "mcts_trader/**"
  - "diff_teacher/**"
  - "train_*.py"
  - "real_trade_jax.py"
---

# Compute allocation contract

**Never a bare `python … &`** on either host. Every job is wrapped, on the workstation and on the
server alike.

## The wrapper

```bash
tools/safe_run.sh --name <tag> --mem <cap> --gpu <N> --gpu-frac 0.7 -- <cmd>
```

inside `tmux` for anything long-running. `systemd-run --user --scope -p MemoryMax=<cap>` is an
acceptable substitute for the RAM cap alone.

## RAM (mandatory, every job)

Hard cgroup cap, always. Current workstation capacity is **53.8 GiB visible** (operator replaced
the modules 2026-08-01 and runs a custom-calculated profile): **single job ≤32G, aggregate ≤40G,
keep ≥12G free.** Any older note quoting ≤18G/≤22G/32GB-total describes the retired 4-DIMM config.

RAM headroom — not VRAM — is the binding constraint on the workstation. Swap does not save the
box; thrash goes straight to a hard lock.

## VRAM is NOT capped by `safe_run`

`safe_run.sh` caps RAM via cgroup (`MemoryMax`, `MemorySwapMax=0`) and provides **no VRAM
envelope**. On the shared server, `XLA_PYTHON_CLIENT_MEM_FRACTION` is **per-visible-device** with
PREALLOCATE default-on, so a blind `--gpu 1,2,3` grabs that fraction of **each** GPU at init and
OOMs the co-tenant — at init, not gradually.

- Default to **single-GPU on a verified-idle GPU** (`nvidia-smi --query-gpu=memory.used` ≈ 0).
- Treat multi-GPU as opportunistic, gated on the whole pool being empty. Lower `--gpu-frac` when
  co-tenanting.
- Unbounded parallelism lives in the **batch/vmap axis, not the device count** — one GPU
  iterating batches already satisfies an embarrassingly-parallel sample axis.
- `gpu-server`: GPU 0 is VirtualBox-reserved; 1/2/3 are training. ≤2 concurrent
  `train_phase2.py` per host (TASK_742 — rank-head contention deadlock). Evals can fill the rest.

## JAX settings

`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` is enough. **Leave `XLA_PYTHON_CLIENT_PREALLOCATE` at its
default (true)** and do **not** set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` — preallocation + BFC
gives smoother behaviour across long runs. The three-var "VRAM safety" recipe was proposed after
the 2026-04-22 kernel lockup, but the real cause there was reth Ethereum-sync IO contention, not
JAX preallocation. Any task doc still carrying `PREALLOCATE=false ALLOCATOR=platform` is stale.

## Workstation GPU jobs are allowed

The blanket "GPU → server only" ban was **retired by operator override 2026-07-05**: the
requirement is correct allocation, not abstinence. Check free VRAM first (the desktop compositor
holds some), run **one** GPU job at a time, and prefer not to launch while heavy desktop GPU apps
are active. The server remains preferred for long/large training when reachable.

## Server launches go in tmux

Any training/heavy job on `gpu-server` starts inside **tmux**, not `screen`, not bare background:

```bash
tmux new-session -d -s <tag> "bash /tmp/<launch>.sh"
```

The job runs on the server; the workstation is only a remote control. tmux keeps it alive when the
workstation powers off and keeps state recoverable (`tmux attach -t <tag>`, `train.log`,
`/tmp/<tag>_done.flag`). Pack multi-cell queues into **one** self-advancing orchestrator session
that waits on done-flags between cells, so the queue completes with the workstation off.

## Keep the server loaded

An idle `gpu-server` is unrecoverable sunk cost — server GPU time is the campaign bottleneck
(12–35 h per experiment). Server-agent autonomy to launch queued or operator-authorised
experiments when idle is **pre-approved**. When authorising an experiment, stage the whole
continuation queue so cells launch back-to-back; don't authorise cell-by-cell.

## Recovery gotcha

After any reboot `/dev/sdb1` (`storage disk`) does not auto-mount, so the `.git/objects` symlink
dangles and every git command fails with `fatal: not a git repository`. Fix:
`udisksctl mount -b /dev/sdb1`. See `docs/infrastructure.md`.
