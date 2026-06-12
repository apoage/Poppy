"""
train_phase2_cli.py — Argparse for train_phase2.py.

Extracted from train_phase2.py (TASK_772.1, 2026-05-23) as a low-risk
modularity refactor: 332 LOC of pure argparse with zero JAX/numpy/Flax
dependencies. The function `parse_args()` returns the same `argparse.Namespace`
the inline version did — drop-in replacement, bit-identical CLI surface.

Importing this module is cheap (only `argparse` from stdlib); the caller
`train_phase2.py` continues to own all JAX initialization and config wiring.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Dynamics Training + MCTS ExIt")

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Phase 1 checkpoint path (.pkl)')

    # Stage
    parser.add_argument('--stage', type=str, default='A', choices=['A', 'B'],
                        help='Training stage: A=dynamics warmup, B=MCTS ExIt')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Steps per epoch override')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed override (config.seed). Used for multi-seed validation runs.')

    # Data
    parser.add_argument('--periods-file', type=str, default='/tmp/foxglove_periods.json',
                        help='Periods JSON for SegmentPool')
    parser.add_argument('--db-path', type=str, default='trading_data.db',
                        help='SQLite database path')

    # Learning rates
    parser.add_argument('--dynamics-lr', type=float, default=None)
    parser.add_argument('--policy-lr', type=float, default=None)
    parser.add_argument('--value-lr', type=float, default=None)

    # MCTS (Stage B)
    parser.add_argument('--mcts-sims', type=int, default=None,
                        help='MCTS simulations per step (Stage B)')
    parser.add_argument('--mcts-discount', type=float, default=None,
                        help='MCTS tree search discount (should match td_discount)')

    # Factored per-asset search (TASK_724)
    parser.add_argument('--factored', action='store_true',
                        help='Use factored per-asset search instead of joint MCTS')
    parser.add_argument('--factored-temperature', type=float, default=None,
                        help='Q-value softmax temperature (default: 1.0)')
    parser.add_argument('--factored-sub-batch', type=int, default=None,
                        help='Sub-batch size for factored search (default: 4096)')
    parser.add_argument('--no-reset-value-head', action='store_true',
                        help='Keep existing value head (don\'t re-init with He normal)')
    parser.add_argument('--bc-weight-min', type=float, default=None,
                        help='BC weight floor (default: 0.03)')
    parser.add_argument('--bc-decay-steps', type=int, default=None,
                        help='BC decay steps (default: 20000)')
    # TASK_726: Class-Balanced BC
    parser.add_argument('--no-bc-balance', action='store_true',
                        help='Disable class-balanced BC (use plain per-asset CE mean)')
    parser.add_argument('--bc-max-trade-weight', type=float, default=None,
                        help='Max trade weight boost for class-balanced BC (default: 64.0)')
    # TASK_781 / TASK_725-D: Return-weighted BC
    parser.add_argument('--bc-return-weighted', action='store_true',
                        help='TASK_781 / TASK_725-D: weight each BC sample by '
                             '|oracle_values| / mean(|oracle_values|). Bull-period '
                             'samples (high |remaining PnL|) get more gradient. '
                             'Default OFF = canon (bit-identical to pre-781 BC loss).')
    # TASK_762: Reward shaping (reward_scale + fee_penalty_bps)
    parser.add_argument('--reward-scale', type=float, default=None,
                        help='Per-step reward scaling factor (default: 100.0). '
                             'Sweep ∈ {25, 50, 100, 200} per experiments/762/SPEC.md.')
    parser.add_argument('--fee-penalty-bps', type=float, default=None,
                        help='Extra per-trade fee penalty in basis points (default: 0). '
                             'Sweep ∈ {0, 2, 5, 10} per experiments/762/SPEC.md.')
    # TASK_763: Policy entropy regularization
    parser.add_argument('--entropy-coef', type=float, default=None,
                        help='Entropy bonus coefficient on policy loss (default: 0.0 = canon). '
                             'Sweep ∈ {0.005, 0.01, 0.02, 0.05, 0.1} per experiments/763/SPEC.md.')
    parser.add_argument('--min-entropy', type=float, default=None,
                        help='Entropy floor below which 5× penalty fires (default: 0.0). '
                             'Recommended 3.0 for joint MCTS (ln(320) ≈ 5.77).')
    # TASK_787 (experiment A): over-trade penalty in Stage-B loss
    parser.add_argument('--trade-rate-penalty-weight', type=float, default=None,
                        help='Per-asset L2 over-trade penalty weight on Stage-B loss '
                             '(default: 0.0 = canon, term not traced). '
                             'Sweep ∈ {0.05, 0.1, 0.2} per tasks/TASK_787_PNL_IN_LOSS.md.')
    # TASK_790 (experiment A2): realized-PnL value term in Stage-B loss
    parser.add_argument('--realized-value-weight', type=float, default=None,
                        help='Immediate realized-PnL value weight on Stage-B loss '
                             '(default: 0.0 = canon, term not traced). Grounds the '
                             'policy in executed next-step return. Sweep ∈ {0.05, 0.1, '
                             '0.2, 0.5} per tasks/TASK_790_REALIZED_PNL_VALUE.md.')
    # TASK_804-B1: token front-end config as CLI overrides (arms 2-3).
    # Each defaults to None so absence = inherit the Phase2Config / checkpoint
    # default (canon: use_token_frontend=False, vocab_file='', language='L1_bpe',
    # cache_dir=''). The bool mirrors --disable-limit-orders' three-state
    # BooleanOptionalAction (None=inherit, True=force ON, False=force OFF) so a
    # resumed ckpt carrying use_token_frontend=True can be forced OFF.
    parser.add_argument('--use-token-frontend',
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help='TASK_804: enable the candle-language token '
                             'front-end (per-timestep token embedding added to '
                             'the encoder temporal surface). Default = None = '
                             'inherit from checkpoint config (canon = False). '
                             'Use --no-use-token-frontend to force OFF on resume.')
    parser.add_argument('--token-frontend-vocab-file', type=str, default=None,
                        help='TASK_804: path to the frozen codebook JSON loaded '
                             'when --use-token-frontend is ON (e.g. runs/799_'
                             'grammar/L1_RESULT_vocab.json). Default = None = '
                             'inherit (canon = "").')
    parser.add_argument('--token-frontend-language', type=str, default=None,
                        help='TASK_804: candle language driving the serving-lag '
                             '(L1_bpe -> lag 2, L8_haar -> lag 0). Default = None '
                             '= inherit (canon = "L1_bpe").')
    parser.add_argument('--token-frontend-cache-dir', type=str, default=None,
                        help='TASK_804: OPTIONAL per-asset timeline cache dir '
                             '(load-once-to-RAM across runs, keyed by vocab_sha). '
                             'Default = None = inherit (canon = "").')
    # TASK_767: Action-space restructure (drop limit orders).
    # TASK_771/F4 (P2): switched to BooleanOptionalAction so --no-disable-limit-orders
    # can force the flag OFF even when the resumed checkpoint config carries
    # disable_limit_orders=True. Default (flag absent) = None = inherit from
    # the checkpoint / config defaults.
    parser.add_argument('--disable-limit-orders',
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help='TASK_767: drop the two limit-order actions '
                             '(LimitSell=3, LimitBuy=4) from the Phase-2 action '
                             'space. Their policy logits are masked to -inf and '
                             'MCTS never expands them — 64x5=320 -> effective '
                             '64x3=192. Default = None = inherit from checkpoint '
                             'config (canon = False). Use --no-disable-limit-orders '
                             'to explicitly force the flag OFF on resume.')
    # TASK_769: Value-clip mode (F2 fix — PopArt pre-clip reorder)
    parser.add_argument('--value-clip-mode', type=str, default=None,
                        choices=['pre_popart', 'post_popart'],
                        help='TASK_769: when to clip value targets to [-10, 10]. '
                             'pre_popart (default = canon, bit-identical to '
                             'pre-769): clip BEFORE PopArt normalization '
                             '(censors PopArt dynamic range — the F2 bug). '
                             'post_popart: feed PopArt the unclipped target so '
                             'running statistics absorb the scale; only the '
                             'post-normalized target is clipped to the '
                             'categorical support.')
    # TASK_727: Auxiliary direction head
    parser.add_argument('--no-direction-head', action='store_true',
                        help='Disable per-asset direction head (default: on)')
    parser.add_argument('--direction-weight', type=float, default=None,
                        help='Direction loss weight (default: 0.2)')
    parser.add_argument('--direction-horizon', type=int, default=None,
                        help='Horizon for per-asset future cum return (default: 20)')
    # TASK_728: Direction → policy structural bias + encoder unfreeze
    parser.add_argument('--no-direction-policy-bias', action='store_true',
                        help='Disable structural direction→policy bias refiner')
    parser.add_argument('--encoder-lr', type=float, default=None,
                        help='Encoder learning rate (0=frozen, default: config)')
    # TASK_782 / TASK_780 Phase 4: epoch at which to unfreeze encoder mid-training
    # (None = never; if set, encoder joins training at lr=encoder_lr after this epoch).
    parser.add_argument('--encoder-unfreeze-epoch', type=int, default=None,
                        help='Epoch to unfreeze encoder (None = encoder_lr behavior from epoch 0)')
    # TASK_728b: Warm-start refiner kernel to break zero-init dead-gradient basin
    parser.add_argument('--warm-direction-refiner', action='store_true',
                        help='Overwrite refiner kernel with action priors (BUY=+α, SELL=−α)')
    parser.add_argument('--direction-prior-alpha', type=float, default=0.5,
                        help='Prior magnitude α for refiner warm-start (default: 0.5)')
    # TASK_729a: Reset direction head (+ refiner) on load — new head arch
    parser.add_argument('--reset-direction-head', action='store_true',
                        help='Drop direction_head + policy_direction_refiner from checkpoint; fresh-init both')
    # TASK_730: Cross-sectional rank oracle head
    parser.add_argument('--no-rank-head', action='store_true',
                        help='Disable per-asset rank-score head (default: on)')
    parser.add_argument('--rank-weight', type=float, default=None,
                        help='Rank loss weight (default: 1.0)')
    parser.add_argument('--rank-loss', type=str, default=None, choices=['mse'],
                        help='Rank loss form (default: mse; listwise/pairwise held as stretch)')
    parser.add_argument('--reset-rank-head', action='store_true',
                        help='Drop rank_head from checkpoint; fresh-init')
    # TASK_734: Flow-matching dynamics loss
    parser.add_argument('--flow-matching', action='store_true',
                        help='Replace MSE consistency loss with velocity-matching (avoids identity collapse)')
    parser.add_argument('--flow-matching-velocity-weight', type=float, default=None,
                        help='Flow matching: weight on velocity loss vs state loss (default: 0.8)')
    # TASK_735: Aux log-return prediction head
    parser.add_argument('--no-aux-price-head', action='store_true',
                        help='Disable aux log-return prediction head (default: on)')
    parser.add_argument('--aux-price-weight', type=float, default=None,
                        help='Aux log-return loss weight (default: 0.1)')
    parser.add_argument('--reset-aux-price-head', action='store_true',
                        help='Drop aux_price_head from checkpoint; fresh-init')
    # TASK_740: Bella distillation prediction head (Path β-v2)
    parser.add_argument('--no-bella-distill-head', action='store_true',
                        help='Disable Bella distillation head (default: follows config '
                             'use_bella_distill_head, typically False)')
    parser.add_argument('--bella-distill-head', action='store_true',
                        help='Enable Bella distillation head (TASK_740)')
    parser.add_argument('--bella-distill-weight', type=float, default=None,
                        help='Bella distillation loss weight (default: 0.1)')
    parser.add_argument('--bella-preds-manifest', type=str, default=None,
                        help='Path to Bella predictions manifest JSON '
                             '(default: config/bella_preds_manifest.json)')
    parser.add_argument('--reset-bella-distill-head', action='store_true',
                        help='Drop aux_bella_head from checkpoint; fresh-init')
    # TASK_754: Bella trajectory-shape distillation head (β-v3 successor to TASK_740)
    parser.add_argument('--bella-shape-distill', action='store_true',
                        help='TASK_754: enable trajectory-shape distillation head. '
                             'Mutually exclusive with --bella-distill-head.')
    parser.add_argument('--no-bella-shape-distill', action='store_true',
                        help='TASK_754: disable trajectory-shape head '
                             '(default: follows config use_bella_shape_head).')
    parser.add_argument('--bella-shape-mode', type=str, default=None,
                        choices=['moments', 'lowrank', 'contrastive'],
                        help='TASK_754: descriptor mode. Only "moments" '
                             'implemented; lowrank/contrastive raise NotImplementedError.')
    parser.add_argument('--bella-shape-weight', type=float, default=None,
                        help='TASK_754: loss weight on the shape MSE (default 0.1).')
    parser.add_argument('--bella-shape-descriptor-dim', type=int, default=None,
                        help='TASK_754: expected descriptor dimensionality '
                             '(default 14; mismatch is a hard error at startup).')
    parser.add_argument('--bella-shape-component-weights', type=str, default=None,
                        help='TASK_754: comma-separated 6-tuple of per-component '
                             'weights (mean, std, skew, kurt, s1, s1_ratio). '
                             'Default (2.0, 1.0, 0.5, 0.25, 1.0, 2.0).')
    parser.add_argument('--bella-shape-manifest', type=str, default=None,
                        help='TASK_754: path to the trajectory-shape manifest '
                             '(default config/bella_shape_manifest.json).')
    parser.add_argument('--reset-bella-shape-head', action='store_true',
                        help='TASK_754: drop aux_bella_shape_head from checkpoint; fresh-init')
    # TASK_753: Bella as data augmentation (β-v4 successor to 740/754)
    parser.add_argument('--bella-augment', action='store_true',
                        help='TASK_753: enable Bella-as-data augmentation. With this flag, '
                             'a fraction (--bella-augment-ratio) of SegmentPool slots are '
                             'sourced from Bella-generated synthetic OHLCV instead of '
                             'historical market data. No new heads / no new loss surfaces. '
                             'Mutually exclusive with --bella-distill-head (740) and '
                             '--bella-shape-distill (754).')
    parser.add_argument('--no-bella-augment', action='store_true',
                        help='TASK_753: explicit disable (default; same as not passing '
                             '--bella-augment).')
    parser.add_argument('--bella-augment-ratio', type=float, default=None,
                        help='TASK_753: fraction of pool slots sourced from Bella '
                             '(default 0.25). Range [0, 1].')
    parser.add_argument('--bella-augment-mode', type=str, default=None,
                        choices=['append', 'replace', 'interleave'],
                        help='TASK_753: slot-injection mode (default append). '
                             'Only "append" is implemented in v1; replace/interleave '
                             'are reserved for TASK_753b.')
    parser.add_argument('--bella-augment-regime-mode', type=str, default=None,
                        choices=['uniform', 'match', 'unconditional'],
                        help='TASK_753: regime sampling per Bella slot. uniform (default), '
                             'match (uses period regime; falls back to uniform in v1), '
                             'unconditional (no regime conditioning).')
    parser.add_argument('--bella-augment-checkpoint', type=str, default=None,
                        help='TASK_753: path to Bella generator checkpoint '
                             '(default checkpoints/diff_teacher/generator_best/generator).')
    parser.add_argument('--bella-augment-k-samples', type=int, default=None,
                        help='TASK_753: Bella samples per generate_on_device call (default 1).')
    parser.add_argument('--bella-augment-seq-length', type=int, default=None,
                        help='TASK_753: target seq_length per Bella slot (default 800 = '
                             '2 chained chunks of native 400). 0 = native. Bella ckpt '
                             'has native seq_length=400; --generate(seq_length=N) does not '
                             'work for the current conv ckpt — chained calls are used.')
    parser.add_argument('--bella-augment-refresh-epochs', type=int, default=None,
                        help='TASK_753: refresh all Bella slots every N epochs (default 0 = '
                             'generate once at pool init, reuse for the whole run).')
    parser.add_argument('--bella-augment-volatility-scale', type=float, default=None,
                        help='TASK_753: z-score → log-return scale for Bella raw output '
                             '(default 0.015, matches setup.py:660).')
    parser.add_argument('--bella-augment-asset-policy', type=str, default=None,
                        choices=['remap', 'canonical', 'random'],
                        help='TASK_753: column-remap policy. remap (default, mandatory '
                             'per TASK_745), canonical, random.')
    parser.add_argument('--bella-augment-log-every', type=int, default=None,
                        help='TASK_753: log Bella slot stats every N _make_bella_slot '
                             'calls (default 100).')
    # TASK_738: Legacy 27-ch encoder mode (ablate TASK_732)
    parser.add_argument('--legacy-27ch', action='store_true',
                        help='Use 27-channel encoder (skip cross-sectional channels). '
                             'Required when loading pre-TASK_732 checkpoints.')
    parser.add_argument('--value-weight', type=float, default=None,
                        help='Value loss weight (default: 1.0)')
    # TASK_739: Dense sign-of-forward-return oracle (Tier A1)
    parser.add_argument('--oracle-mode', type=str, default=None,
                        choices=['topk_path', 'dense_sign', 'hybrid'],
                        help='BC target oracle: topk_path (default, pre-739), '
                             'dense_sign (TASK_739 per-asset sign(r_fwd)), '
                             'hybrid (reserved). Value/path still use top-k.')
    parser.add_argument('--sign-epsilon', type=float, default=None,
                        help='Dead-zone threshold on |r_fwd| for dense_sign oracle '
                             '(default: 1e-4). Tune to match 5-15%% per-asset trade density.')

    # Value target mode (TASK_719, TASK_777)
    parser.add_argument('--value-target-mode', type=str, default=None,
                        choices=['oracle', 'nstep', 'td0', 'mixture'],
                        help='Value target: oracle (719a), nstep (719b), td0 '
                             '(bootstrap), mixture (777: α·oracle + (1−α)·nstep)')
    parser.add_argument('--nstep-horizon', type=int, default=None,
                        help='N for n-step TD returns (default: 20)')
    parser.add_argument('--td-discount', type=float, default=None,
                        help='Discount for TD value targets (default: 0.95)')
    # TASK_777: mixture value-target controls. Only consumed when
    # --value-target-mode=mixture. α=1.0 reproduces oracle bit-identical
    # (§7a smoke gate); α=0.0 reproduces nstep bit-identical.
    parser.add_argument('--value-mixture-alpha', type=float, default=None,
                        help='TASK_777: convex blend α for value_target_mode=mixture. '
                             'td_target_raw = α·oracle + (1−α)·nstep_target. '
                             'Must be in [0, 1]. Default: 0.5 (symmetric blend). '
                             'Ignored unless --value-target-mode mixture.')
    parser.add_argument('--oracle-horizon', type=int, default=None,
                        help='TASK_777/756b: override oracle_value_horizon '
                             '(default: 400 = episode length). Effective for '
                             'oracle and mixture modes; ignored for nstep/td0.')
    # TASK_784/756c: EWMA smoothing on oracle_cum_pnl per-step deltas before
    # the remaining-PnL value target is gathered. λ=0 (default) → no-op,
    # canon bit-identical. λ>0 smooths peakiness; targets the per-seed STD
    # 2.28pp variance signature from runs/746_value_oracle/result.md.
    parser.add_argument('--oracle-smoothing', type=float, default=None,
                        help='TASK_784: EWMA λ on oracle_cum_pnl deltas before '
                             'oracle value-target gather. Must lie in [0, 1). '
                             'Default 0.0 = canon (smoothing OFF, bit-identical). '
                             'Effective for --value-target-mode oracle.')

    # Data diversity (TASK_716 follow-up)
    parser.add_argument('--n-slots', type=int, default=17,
                        help='Concurrent segments in RAM pool (default: 17)')
    parser.add_argument('--max-runs-per-group', type=int, default=1,
                        help='Runs per date range — increase for more asset diversity (default: 1)')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')

    # Modes
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate checkpoint + pool loading, then exit')
    parser.add_argument('--test-init', action='store_true',
                        help='Test dynamics initialization + recurrent_inference')
    parser.add_argument('--test-mcts', action='store_true',
                        help='Test MCTS search with Stage A checkpoint')

    # Benchmark
    parser.add_argument('--benchmark-every', type=int, default=0,
                        help='Run benchmark every N epochs (0=disabled)')
    parser.add_argument('--benchmark-gpu', type=int, default=1,
                        help='GPU index for benchmark')

    # TASK_729b: Per-asset encoder width overrides (V2 encoder)
    parser.add_argument('--embed-dim', type=int, default=None,
                        help='Per-asset embedding width. Must match Phase 1 checkpoint.')
    parser.add_argument('--encoder-heads', type=int, default=None,
                        help='Cross-asset attention head count. Must match Phase 1 checkpoint.')
    # TASK_729c: Encoder family selector (must match Phase 1 checkpoint)
    parser.add_argument('--encoder-version', type=int, default=None, choices=[2, 3],
                        help='Encoder family: 2=V2 flatten+Dense, 3=V3 minGRU temporal.')
    # TASK_741 (Mint): Post-pool per-asset scalar injection via FiLM.
    parser.add_argument('--per-asset-injection',
                        choices=['none', 'cheat', 'xs', 'both'],
                        default=None,
                        help='TASK_741 (Mint): FiLM-inject per-asset scalars '
                             'on the post-pool feature surface. xs is '
                             'production-safe; cheat is diagnostic only.')

    # TASK_749 (Titans-MAG): neural long-term memory module on the encoder
    # temporal axis (paper: Behrouz et al. 2025). 9 flags below.
    parser.add_argument('--titans-memory', action='store_true',
                        help='TASK_749: enable Titans-style neural memory '
                             'inside V3 encoder, gated against minGRU. Default OFF; '
                             'gate init large-negative so step-0 is byte-identical.')
    parser.add_argument('--titans-variant', type=str, default=None,
                        choices=['mag'],
                        help='TASK_749: Titans variant. Only "mag" implemented '
                             '("mac", "mal" reserved for follow-up tasks).')
    parser.add_argument('--titans-memory-depth', type=int, default=None,
                        help='TASK_749: depth L_M of memory MLP (default 2).')
    parser.add_argument('--titans-memory-hidden', type=int, default=None,
                        help='TASK_749: hidden width of memory MLP '
                             '(default: same as embed_dim).')
    parser.add_argument('--titans-persistent-tokens', type=int, default=None,
                        help='TASK_749: N_p persistent memory tokens '
                             'prepended pre-scan (default 4, paper §2.3).')
    parser.add_argument('--titans-surprise-momentum-init', type=float, default=None,
                        help='TASK_749: init bias for η_t sigmoid (default 0.9 → '
                             'retain past surprise).')
    parser.add_argument('--titans-decay-init', type=float, default=None,
                        help='TASK_749: init bias for α_t sigmoid (default 0.01 → '
                             'near-zero forgetting at step 0).')
    parser.add_argument('--titans-lr-init', type=float, default=None,
                        help='TASK_749: init bias for θ_t sigmoid (default 0.01 → '
                             'small inner-loop step).')
    parser.add_argument('--titans-gate-init', type=float, default=None,
                        help='TASK_749: init bias for MAG gate sigmoid '
                             '(default -5.0 → mix≈0.7%%, byte-identical baseline). '
                             '-3 → mix≈4.7%% for faster gate engagement.')
    parser.add_argument('--titans-lr', type=float, default=None,
                        help='TASK_749 fix 5: dedicated outer-LR for Titans '
                             'params (gate + memory MLP + persistent tokens + '
                             'α/η/θ projections). Distinct from --titans-lr-init '
                             '(inner-loop θ_t bias). When None (default), Titans '
                             'inherits encoder_lr. Set to ~100× encoder_lr '
                             '(e.g. 3e-3) so the MAG gate sigmoid can move in '
                             'feasible epoch counts.')
    parser.add_argument('--reset-titans-memory', action='store_true',
                        help='TASK_749: drop titans subtree from checkpoint at '
                             'load (forces fresh init). Use on first run with a '
                             'pre-TASK_749 ckpt.')

    return parser.parse_args()
