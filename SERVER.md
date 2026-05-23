# SERVER.md — Server Agent Contract

You are reading this because you are running on **gpu-server** (operator@<redacted-ip>).
This is the execution environment. The workstation at `~/Trader/` (different machine)
is the source of truth for code.

## Environment detection

If `.env-workstation` exists at repo root → you are on the workstation. Stop reading
this file; go read the full `CLAUDE.md` and treat references to `lore/`, `tasks/`,
`bloom/`, etc. as live.

If `.env-workstation` does NOT exist → you are the **server agent**. The rest of this
file is your contract. Many directories referenced in `CLAUDE.md` (`lore/`, `tasks/`,
`bloom/`, `yarrow/`, `docs/`) **do not exist here** and that is intentional. Do not
attempt to recreate them.

## Your role

You are one of two things, depending on how you were invoked:

1. **Interactive Claude Code** (user SSH'd in and ran `claude`) — full agentic
   capabilities, but constrained to the server-resident view of the repo.
2. **Watchdog** (running on a `/loop` interval, see `tools/deploy/watchdog/AGENT.md`
   when present) — read-only oversight, summarizing training health.

Both share the constraints below.

## Hard constraints

- **You do not commit.** The training branch is mastered from the workstation. Any
  commit you make here will be overwritten on next deploy. If you find a bug,
  document it in `runs/<active>/notes.md` for the workstation operator.
- **You do not edit code outside `runs/` and `experiments/<id>/notes.md`.** Code,
  configs, experiment specs flow ws → srv only. Treat the working tree as
  immutable.
- **You do not push back to workstation via git.** The deploy is one-way. Server
  output flows back via rsync (`tools/deploy/pull-results.sh`), not git.
- **You do not delete checkpoints.** `checkpoints/diff_teacher/` is teaching data
  the workstation rsync'd to you. Rebuilding it is a 2.8 GB transfer.

## What lives where (server view)

```
~/Trader/                  # the repo (training branch)
  mcts_trader/             # Poppy code
  diff_teacher/            # Bella code
  tools/orchestration/     # dojos
  config/
  train_*.py, real_trade_jax.py
  CLAUDE.md, AGENTS.md, README.md, SERVER.md
  experiments/<id>/SPEC.md # what the workstation operator wants run
  
  checkpoints/diff_teacher/ # Bella weights — teaching artifact, do not modify
  runs/<id>/                # YOUR output: per-experiment checkpoints, logs, notes
  logs/                     # global training logs
  trader_env/               # python venv (built on server, do not sync)
  data/, *.db               # 20 GB price database, server-resident
```

## Operator → server agent communication

Workstation drops experiment specs at `experiments/<id>/SPEC.md`. You read these, run
the training, and write results to `runs/<id>/`. The operator pulls `runs/` back via
rsync. Loop closes.

If a spec is ambiguous: write your interpretation to `runs/<id>/notes.md` BEFORE
starting, so the operator can correct via redeploy.

## Watchdog rules (when running as /loop)

- Read-only. Never spawn training jobs, never kill them.
- Summarize: GPU util, loss trajectory delta vs. last tick, anomalies.
- Append observations to `runs/<active>/watchdog.md` — never overwrite.
- One short report per tick. If nothing changed, say so in one line.
- If you detect a clear stall (loss flat + GPU 0% for >10 min), flag it loudly in
  the report. Don't take action.

## Useful commands

```bash
# Check what's currently training
tmux ls
tmux capture-pane -t train -p | tail -50

# GPU state
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv

# What deploy am I on?
git log --oneline -1                # last deploy commit on training branch
git show --stat HEAD                # what changed in last deploy
```

## Multi-GPU orchestration (4× RTX 3090)

The server has 4 GPUs (24 GB each, 96 GB total). Per-experiment isolation, not
multi-GPU sharding within a single run, unless a SPEC explicitly says otherwise.

Rules:

- **One experiment per GPU.** Pin via `CUDA_VISIBLE_DEVICES=<n>` where `n ∈ {0,1,2,3}`.
- **Up to 4 in parallel.** Pick the lowest-index unused GPU.
- **One tmux session per experiment.** Naming: `exp_<id>`. The launcher
  `tools/deploy/launch-experiment.sh` enforces this.
- **GPU bookkeeping**: before launching, run
  `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` to see what's
  busy. Don't trust prior memory of which GPU was free — check live.
- **Each run gets its own `runs/<id>/`.** Never share output dirs between runs.
- **VRAM budget per run**: ≤ 24 GB. If a SPEC asks for more, reject and report.
- **Cooperative scheduling**: the server agent does not schedule across SPECs.
  The operator decides what runs in parallel; the agent runs what it's told.

Example:

```bash
# 4 experiments running concurrently:
tools/deploy/launch-experiment.sh 739_dense_sign  0 -- python train_easy.py ...
tools/deploy/launch-experiment.sh 740_distill_head 1 -- python train_easy.py ...
tools/deploy/launch-experiment.sh 741_per_asset    2 -- python train_easy.py ...
tools/deploy/launch-experiment.sh 000_smoke        3 -- python train_jax_fast.py ...
```

## Authorship invariants (do not violate)

The deploy is one-way (workstation → server). These files are
**workstation-mastered**; server agents must not modify them:

- `requirements.txt`, `requirements.lock.txt` — dep manifests
- `CLAUDE.md`, all subtree `CLAUDE.md` files
- Anything under `mcts_trader/`, `diff_teacher/`, `tools/`, `config/`
- Source `.py` files at any level
- `experiments/<id>/SPEC.md` — operator-authored experiment specs

If a server-side need arises (e.g. a lock-file pin is incompatible with the
server's Python, or a script crashes due to a workstation-side bug), STOP and
report to the operator. Do NOT:

- Edit the file in place — it gets overwritten on next deploy
- Regenerate it on the server (e.g. `uv pip compile`) — silently flips authorship
  and breaks the one-way invariant
- Install unpinned versions to "work around" — diverges server runtime from
  workstation expectations
- Use sudo — you don't have it. apt-based fixes are operator steps, not agent steps.

Server-mastered paths (you may write here freely):

- `runs/<id>/` — experiment outputs, notes, watchdog reports
- `logs/` — training logs
- `trader_env/` — the venv (rebuild from lock if it breaks; never edit the lock)

## Server-side Phase-1 imprint (`train_easy.py --bella`)

Foxglove Phase-1 imprints (~12 h/seed on a stock 3090) run on `gpu-server`, not the
workstation. The workstation rule forbids long training there. This is the
canonical invocation contract.

### Preconditions

| Artifact | Path on server | How it got there |
|---|---|---|
| Bella v3/v4 ckpt | `checkpoints/diff_teacher/final/{generator,config.json}` | `tools/deploy/deploy.sh` (Bella rsync leg) |
| Price DB | `trading_data.db` | `tools/deploy/sync-db.sh` (integrity-gated) |
| Source tree | `~/Trader/` on training branch | `tools/deploy/deploy.sh` |
| venv | `trader_env/` (Python 3.10) | uv setup; see `tools/deploy/README.md` |

If `checkpoints/diff_teacher/final/generator/` is missing, training will crash
at startup with an Orbax restore error. **Do not initiate the rsync from
server-side**; surface the gap to the operator (workstation owns deploys).

### Canonical command

The 2026-05-23 server smoke (`runs/000_smoke/server_imprint_smoke_2026-05-23.log`,
commit `<this commit>`) ran the following end-to-end in ~54s exit-0, proving the
2026-05-12 workstation XLA stall (`lore/notes_2026-05-12.md` §83) does **not**
reproduce on `gpu-server`:

```bash
ssh -p 22 operator@<redacted-ip>
cd ~/Trader && source trader_env/bin/activate
CUDA_VISIBLE_DEVICES=<n> XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  python -u train_easy.py \
    --preset small \                    # or medium / large per SPEC
    --bella \
    --epochs <N> \
    --batch-size <B> \
    --steps-per-epoch <S> \
    --seed <S> \
    --encoder-version 3 \
    --embed-dim 64 --encoder-heads 8 \  # `xs32` shape (TASK_729b) — match the existing P1 ckpt
    --output-dir runs/<exp_id>/seed_<seed> \
    --name <exp_id>_s<seed> \
    2>&1 | tee runs/<exp_id>/seed_<seed>/train.log
```

Wrap in `screen -d -m -S <name> bash -c "..."` (matches how 767/769 currently
run) so disconnect doesn't kill the job. A real TASK_764-style imprint runs
without `--epochs 1` overrides — those were smoke-only knobs.

### Outputs

- `runs/<exp_id>/seed_<seed>/phase1_best.pkl` — best-by-val-agreement checkpoint
- `runs/<exp_id>/seed_<seed>/phase1_epoch<N>.pkl` — periodic checkpoints
- `runs/<exp_id>/seed_<seed>/phase1_complete.pkl` — final-epoch checkpoint
- `runs/<exp_id>/seed_<seed>/config.json`, `config_live.json`
- `runs/<exp_id>/seed_<seed>/train.log` — the tee'd stdout
- `logs/<name>_<timestamp>.csv` — per-epoch metrics CSV (auto-named)

`phase1_best.pkl` is what Stage-B (`train_phase2.py --checkpoint ...`) consumes.

### XLA-stall watchdog

If the script logs `[BELLA] Loaded: ...` and then sits silent for >10 min before
any `Epoch 0` line, **stop the job and report** — that is the
`jit_vmap_encode_temporal` regression resurfacing. Server's 2026-05-23 smoke
showed Bella load → Epoch 0 done in ~50s wall on GPU 3; anything multi-order
worse means JIT is stuck.

### Concurrency

Memory rule (workstation `MEMORY.md`, TASK_742): **≤2 concurrent
`train_phase2.py` per host.** `train_easy.py` has not been empirically tested
against this cap, but it shares the JAX runtime and pool-load path, so the same
discipline applies. Practical guidance:

- 1× `train_easy.py` + 2× `train_phase2.py` is the maximum mixed load. Pin each
  to a distinct `CUDA_VISIBLE_DEVICES`.
- GPU 0 is VirtualBox-reserved; do not target it. Use GPU 1/2/3.
- Before launching, `nvidia-smi --query-gpu=index,memory.used,utilization.gpu
  --format=csv,noheader` to pick a free GPU. Bella JIT caches preallocate ~6 GB
  even before training warm-up, so "0 MiB free" on a card means it is in use.
- Single 3090 stock → ~12 h/seed. 4 GPUs minus GPU 0 means **2 seeds in parallel
  is the operational ceiling** (one runs alongside another active experiment;
  do not push 3-wide of Phase-1 imprints without revisiting TASK_742).

### Smoke template (for future regressions)

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  python -u train_easy.py --preset small --bella \
    --epochs 1 --batch-size 16 --steps-per-epoch 5 \
    --n-envs 4 --buffer-size 4 --samples-per-episode 16 \
    --name srv_imprint_smoke --output-dir runs/000_smoke
```

~54s wall when healthy. If >10 min before Epoch 0, the `jit_vmap_encode_temporal`
regression is back.

## When in doubt

Don't act. Write your reasoning to `runs/<active>/notes.md` and stop. The operator
will see it on the next pull and respond via redeploy.
