# 778_bella_rollouts — Bella-Conditioned MCTS Rollouts

**STATUS: BLOCKED on code-budget authorization.** No build, no smoke, no run until reviewer signs off on the ~630 LOC code budget per `tasks/TASK_778_BELLA_AS_ROLLOUTS.md`.

**Created**: 2026-05-25
**Owner**: workstation builds + reviews; server executes after both review gates clear
**Refs**: `tasks/TASK_778_BELLA_AS_ROLLOUTS.md` (design), `tasks/TASK_722_BELLA_DYNAMICS_ROLLOUTS.md` (parent design, parked Feb-2026), `lore/ship_path_recommendations_2026-05-25.md` §4

---

## Goal

Replace the MCTS dynamics-MLP expansion (`mcts_trader/jax_agent/muzero_model.py:701-728 recurrent_inference`) with Bella flow-matching rollouts during Stage-B Phase-2 training. Dynamics MLP repurposed as consistency filter (per-branch MSE between MLP prediction and Bella-encoded latent). MCTS searches over real encoded Bella futures, weighted by consistency.

This is the first mechanism that touches the **dynamics class** (genuinely untouched per `project_single_mechanism_exhausted.md`).

---

## Variants (planned)

| cell | bella_rollout_k | bella_horizon | bella_ode_steps | bella_conditioning | seeds |
|---|---:|---:|---:|---|---|
| 778-bella-k16 | 16 | 20 | 10 | skeleton | {7, 13, 23} |

**Wave 1** = 1 cell × 3 seeds against the existing 760 a1b1 paired baselines under eval-MCTS sims=16 joint scorecard.

**Wave 2** (conditional on Wave 1 mechanism gate firing per TASK_778 §7): sweep K ∈ {8, 16, 32} × horizon ∈ {10, 20, 40} — additional ~45h wall at 2-wide.

---

## Planned launch command (post-authorization, do NOT run before gate 2 clears)

```bash
ssh -p 22 operator@<redacted-ip> << 'EOF'
cd ~/Trader && source trader_env/bin/activate

# 778-bella-k16, seed 7, GPU 1 (NOT GPU 0 — VirtualBox-reserved per MEMORY.md)
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  python train_phase2.py \
    --stage B \
    --load-checkpoint runs/760_train_mcts_sims/a1b1_seed_7/phase2_best.pkl \
    --seed 7 \
    --use-bella-rollouts \
    --bella-rollout-k 16 \
    --bella-ode-steps 10 \
    --bella-horizon 20 \
    --bella-conditioning skeleton \
    --bella-ckpt-path checkpoints/bella_conv_768d_12L \
    --output-dir runs/778_bella_rollouts/bella_k16_seed_7

# 778-bella-k16, seed 13, GPU 2 (parallel)
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  python train_phase2.py \
    --stage B \
    --load-checkpoint runs/760_train_mcts_sims/a1b1_seed_13/phase2_best.pkl \
    --seed 13 \
    --use-bella-rollouts \
    --bella-rollout-k 16 \
    --bella-ode-steps 10 \
    --bella-horizon 20 \
    --bella-conditioning skeleton \
    --bella-ckpt-path checkpoints/bella_conv_768d_12L \
    --output-dir runs/778_bella_rollouts/bella_k16_seed_13

# seed 23 runs sequentially after seed 7 or 13 frees a GPU (NEVER GPU 0)
EOF
```

**Placement rule**: GPU 1/2/3 only. **GPU 0 is VirtualBox-reserved per `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md`** (Hardware section). 2-wide per host per TASK_742 (≤2 concurrent `train_phase2.py` per host).

**Eval per ckpt** (after each Stage-B finishes):
```bash
python benchmark_easy.py \
  --checkpoint runs/778_bella_rollouts/bella_k16_seed_${S}/phase2_best.pkl \
  --eval-mcts-sims 16 joint \
  --paired-against runs/760_train_mcts_sims/a1b1_seed_${S}/phase2_best.pkl
```

---

## Expected wall (after build authorization clears)

| Phase | Wall |
|---|---|
| Code (workstation) | 3-5 days |
| Gate 1 (qa-hunt + qa-crossval static review + mitigations) | 0.5-1 day |
| Gate 2 (bit-identity smoke + flag-ON 1-epoch smoke) | 0.5 day |
| Paired 3-seed × 2-wide Stage-B on server (~2-3× canon step) | 12-15 h |
| Eval-MCTS sims=16 joint per ckpt + result.md authoring | 3-4 h |
| **Total post-authorization wall to first measurable result** | **5-6 days** |

---

## Promotion gate (full detail in TASK_778 §7)

- Paired Δα ≥ +3pp vs 760 a1b1 under eval-MCTS sims=16 joint.
- No seed < −5pp.
- Both hold → 5-seed extension. Else FAIL.

**Mechanism diagnostic (REQUIRED, even if standard gate fails)**:
- Consistency-score distribution across K branches must be non-uniform (not all 0, not all 1). Log histogram per batch in `metrics.csv`.
- Per-window oracle-PnL ratio on Bella branches: tag `result.md` if > 2× historical (TASK_758 protocol per `lore/notes_2026-05-18.md` §134).
- Anti-correlation early-exit at epoch 50 of seed 7: abort cell if Δ-Bal.Acc > +1pp AND Δα < 0.

---

## Pre-launch checklist (for the code-budget reviewer)

The reviewer authorizing the build MUST verify each item before approving the ~630 LOC budget:

- [ ] **Bella ckpt deployed to server** at the path used by `--bella-ckpt-path`. Verify per `tools/deploy/README.md` (Bella rsync flow shipped during TASK_758). Confirm file size ≥ 1.2 GB and `config.json` says `architecture: conv`, `hidden_dim: 768`, `num_layers: 12`, `n_steps: 10` (per `lore/notes_2026-05-18.md` §134).
- [ ] **Dynamics-MLP-consistency baseline measured** on current 760 a1b1 checkpoints. Compute `dynamics_consistency_mse` over a 1-epoch evaluation pass on real historical data. This is the calibration target: if the Bella-future consistency MSEs are an order of magnitude higher than the historical baseline, the dynamics MLP cannot distinguish Bella from real — design assumption breaks pre-launch. Save the histogram to `runs/778_bella_rollouts/CALIBRATION.md`.
- [ ] **qa-hunt review of `mcts_trader/jax_agent/bella_dynamics.py`** complete (gate 1 sign-off). Required findings list: JAX RNG via `jax.random.fold_in` not `split` per `diff_teacher/CLAUDE.md` HIGH-009; vmap shape contracts for `(K × T)` sliding window; ckpt double-nesting handling (`diff_teacher/CLAUDE.md` "Checkpoint double-nesting"); sub-batch sizing for the consistency-score loop; consistency score type stability (no NaN under near-zero dynamics MSE).
- [ ] **qa-crossval review of `train_phase2.py` integration** complete (gate 1 sign-off). Required findings: bit-identical canon path when `--use-bella-rollouts` OFF; interaction with TASK_771/F1+F2 limit-action masking when `--disable-limit-orders` is also on; interaction with TASK_767 action-space restructure; interaction with TASK_763 entropy term (joint-only).
- [ ] **Bit-identity smoke** PASSES: flag OFF, 5 epochs, loss curves bit-identical to current a1b1 canon (allow ULP-scale float32 drift only).
- [ ] **Flag-ON smoke** PASSES at production batch: 1 epoch, no OOM, consistency-score distribution non-degenerate (variance > 0.01), per-batch wall ≤ 500 ms.
- [ ] **VRAM headroom verified**: ~2.5 GB combined (per TASK_722 §VRAM Budget). Confirm via `nvidia-smi` during flag-ON smoke. If > 4 GB, investigate before authorizing 3-seed run.
- [ ] **762 W2 / 763 / 764 status snapshot recorded** at authorization time. TASK_778 build is large enough that any in-flight experiment results that close other mechanism classes (e.g., 763 entropy promotes → less need for dynamics-class probe) should be considered before committing 5-6 days of build.
- [ ] **Operator confirms ship-path memo §4 candidates #1-3, #5 (TASK_775/776/777/779) have all closed or have been deprioritized.** Per ship-path memo §"Recommended execution order": TASK_778 is the post-small-probe rewrite, NOT the first probe.

---

## Cancellation triggers (during build)

- Any of TASK_775/776/777/779 returns paired Δα ≥ +3pp at 3-seed AND clears 5-seed → freeze TASK_778 build, ship the survivor.
- Gate 1 review surfaces more than 15 distinct findings (signal that the design has gaps the spec didn't capture) → re-scope TASK_778 before continuing build.
- Gate 2 bit-identity smoke fails (flag OFF drifts from canon) → bug in the integration; investigate before flag-ON smoke.
- VRAM > 6 GB headroom violation at production batch → reduce K or shorten horizon; if irreducible, **freeze TASK_778** (the alternative is degraded canon-batch or canon-window, which breaks paired comparability).

---

## References

- `tasks/TASK_778_BELLA_AS_ROLLOUTS.md` — full design (this SPEC's parent).
- `tasks/TASK_722_BELLA_DYNAMICS_ROLLOUTS.md` — Feb-2026 parent design, parked.
- `lore/ship_path_recommendations_2026-05-25.md` §4 — ship-path candidate framing.
- `lore/notes_2026-05-18.md` §134 (Bella state + 1440 pad bug + 7× too-easy), §138 (two-gate review protocol + smoke at production batch).
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — Bella class CLOSED with explicit world-model rollout carve-out.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closure; dynamics-class genuinely untouched.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — direct-risk LOW, indirect-risk MEDIUM rationale.
- `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md` — server GPU placement rule (GPU 0 reserved), TASK_742 2-wide rule.
- `experiments/758_bella_augment/SPEC.md` — `too_easy` tagging protocol template.
- `tools/deploy/README.md` — Bella ckpt deploy contract.
