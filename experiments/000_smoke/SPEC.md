# 000_smoke — full bring-up sanity check

## Goal

Confirm the server's Python env, GPU stack, and project code can run a real
training step end-to-end. This is the gate between "env installed" and "ready
for actual experiments."

## Source task

None — infrastructure smoke, not a research experiment.

## Resources

- 1 GPU (use `CUDA_VISIBLE_DEVICES=0`)
- ≤ 8 GB VRAM
- Wallclock budget: 15 min total (most spent on first JIT compile)

## Phases

The agent runs these **in order**, stopping at the first failure and writing the
failure report to `runs/000_smoke/notes.md`.

### Phase A — env

```bash
python --version                                        # expect 3.10.x (matches workstation)
python -c "import jax; print(jax.devices())"            # expect 4× cuda gpus
python -c "import flax, optax, orbax.checkpoint, numpy, pandas"
python -c "import lzero, gymnasium; print(gymnasium.__version__)"   # 1.2.2 expected
```

### Phase B — project imports

```bash
cd ~/Trader
python -c "from mcts_trader import jax_agent"           # Poppy module loads
python -c "from diff_teacher import teacher, generation"  # Bella module loads
```

If any project import fails, that's a missing-dep or python-version bug. STOP
and report; do NOT pip-install anything (per SERVER.md invariants).

### Phase C — Bella inference (NOT training)

**Important**: Bella is a frozen teacher. Smoke test loads it and runs ONE
forward pass to verify the checkpoint is intact and Bella's inference path
works. Do NOT start a Bella training loop — `train_diff_teacher_*.sh`,
`diff_teacher/train.py`, and similar are off-limits for this smoke.

```python
# Inline check the agent can run:
from diff_teacher.teacher import MarketGenerator  # adjust import per actual API
import jax, jax.numpy as jnp
gen = MarketGenerator.from_checkpoint("checkpoints/diff_teacher/final")
print("loaded:", gen)
# Tiny synthetic batch — adjust shape/dtype per the actual API:
# out = gen.sample(rng=jax.random.PRNGKey(0), batch_size=2, ...)
# print("inferred shape:", out.shape)
```

If the import or constructor signature differs, read `diff_teacher/CLAUDE.md`,
`diff_teacher/teacher.py`, and `evaluate_bella.py` (which already loads Bella
for inference) to find the actual API. Report what was used.

### Phase D — 1-step training, Foxglove (`train_easy.py`)

This is the real smoke. All current experiments are Foxglove training that
consumes frozen-Bella outputs as the teaching signal.

```bash
CUDA_VISIBLE_DEVICES=0 python train_easy.py --help | head -60
```

Read the available flags. Identify a "smoke mode" (max-steps=1, smallest batch,
or equivalent). If no explicit smoke flag exists, run the smallest configured
run and **kill after one optimizer step lands in the log**. Capture timing of:
- JIT compile (first step)
- Steady-state step (if budget allows 2–3 steps)

Verify in the log: (a) Bella loaded successfully as teacher, (b) loss is finite,
(c) optimizer step counter advances.

### Phase E — DEFERRED

Poppy via `train_jax_fast.py` is not currently active. Skip. If you want to
double-check the Poppy path is reachable, run `python -c "from mcts_trader
import jax_agent"` (already covered in Phase B) and stop.

## Success criteria

All of:

- [ ] Phase A: prints 4 GPU devices, lzero + gymnasium 1.2.2 imports OK
- [ ] Phase B: both project imports succeed
- [ ] Phase C: Bella checkpoint loads AND one inference forward pass succeeds
- [ ] Phase D: one Foxglove optimizer step completes (log shows step count > 0,
      Bella loaded as teacher, finite loss)
- [ ] No CUDA OOM, no NaN loss, no shape mismatch

Phase E is deferred — not part of pass/fail.

## Output expectations

`runs/000_smoke/` contains:
- `notes.md` — the agent's interpretation, decisions, anomalies (start before
  running; append as you go)
- `train.log` — captured stdout/stderr of phases D and E
- `result.md` — final pass/fail per success criterion + 2-line summary

## Notes

- This spec is intentionally vague about exact flags because they're subject to
  drift. The agent reads the actual `--help` output and decides.
- If JAX device discovery returns `[CpuDevice(...)]` instead of CUDA, that's a
  CUDA-toolkit / JAX-cuda-extra mismatch. STOP and report — operator regenerates
  the lock with proper extras on workstation.
- Bella's canonical checkpoint path is `checkpoints/diff_teacher/final/`. If
  that directory doesn't exist or is empty, the rsync is incomplete. STOP and
  report; operator reruns `tools/deploy/deploy.sh`.
