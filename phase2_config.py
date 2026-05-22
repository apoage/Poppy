"""
Phase 2 Configuration — Dynamics Training + MCTS Integration.

TASK_716: Extends EasyConfig with Phase 2 parameters for Stage A (dynamics warmup)
and Stage B (MCTS ExIt).
"""

from dataclasses import dataclass, field
from typing import Optional
from easy_config import EasyConfig


@dataclass
class Phase2Config(EasyConfig):
    """Phase 2 config — inherits all Phase 1 fields, adds dynamics/MCTS params."""

    # === Stage Selection ===
    stage: str = "A"  # "A" = dynamics warmup, "B" = MCTS ExIt

    # === Freeze Control ===
    freeze_encoder: bool = True
    freeze_policy: bool = True     # Stage A: frozen. Stage B: unfrozen
    freeze_value: bool = True      # Stage A: frozen. Stage B: unfrozen
    freeze_dynamics: bool = False   # Always trainable in Phase 2

    # === Dynamics Training (Stage A) ===
    dynamics_lr: float = 3e-4
    consistency_weight: float = 1.0
    reward_weight: float = 1.0
    reward_scale: float = 100.0    # Per-step rewards ~0.001, need 100x to spread across 401 bins

    # === Reward Shaping (TASK_762) ===
    # Additional per-trade fee penalty (basis points) applied on top of transaction_cost
    # during reward target construction (train_phase2.py:632). 0.0 = canon (no extra fee).
    # Sweep range 0–10 bps; tests whether explicit cost penalty breaks the over-trade
    # collapse signature seen across single-mechanism regressions.
    fee_penalty_bps: float = 0.0

    # === Action-Space Restructure (TASK_767) ===
    # When True, the two limit-order actions (LimitSell=3, LimitBuy=4) are
    # dropped from the Phase-2 action space: their policy logits are masked to
    # -inf before softmax and MCTS never expands them — collapsing the live
    # action space from 64×5=320 to an effective 64×3=192. Default False = canon
    # (5-action, bit-identical to pre-767 behavior; all guard branches are dead).
    disable_limit_orders: bool = False

    # === Policy Entropy Regularization (TASK_763) ===
    # Bonus on policy_loss: -entropy_coef * H(policy) + 5*entropy_coef * relu(min_entropy - H).
    # Defaults 0.0 = canon (no regularization). entropy_coef sweep 0.005–0.1;
    # min_entropy floor at 3.0 for joint MCTS (ln(320) ≈ 5.77). Joint MCTS only —
    # factored search has different entropy semantics (per-asset, ln(5) ≈ 1.61).
    entropy_coef: float = 0.0
    min_entropy: float = 0.0

    # === MCTS (Stage B) ===
    mcts_sims: int = 32
    max_considered_actions: int = 16
    epsilon_greedy: float = 0.1
    use_gumbel: bool = True
    mcts_discount: float = 0.95  # TASK_723 Fix 2: must match td_discount (was 0.997)

    # === ExIt Loss Weights (Stage B) ===
    exit_policy_weight: float = 1.0
    exit_value_weight: float = 1.0  # TASK_723 Fix 5: match policy weight (was 0.5)
    exit_consistency_weight: float = 0.5
    bc_weight: float = 0.05        # Oracle BC regularizer
    bc_decay_steps: int = 20000    # TASK_723 Fix 4: slower decay ~50 epochs (was 5000)
    bc_weight_min: float = 0.03   # TASK_723 Fix 4: keep oracle signal meaningful (was 0.005)

    # === Class-Balanced BC (TASK_726) ===
    # Oracle is k=4/64 selected → 93.75% HOLD by construction. Without per-asset
    # trade/hold weighting, the BC gradient fits the HOLD marginal and never learns
    # the per-asset BUY/SELL conditional. Ported from easy_trainer.py:170-202.
    bc_class_balanced: bool = True     # Adaptive trade/hold weight balancing
    bc_max_trade_weight: float = 64.0  # Cap on trade weight boost (64 = full k/n balance)
    bc_return_weighted: bool = False   # Deferred: return-magnitude weighted boost (Change 3)

    # === Auxiliary Direction Head (TASK_727) ===
    # Per-asset sign-of-future-return probe. Provides 64 BCE labels/sample vs
    # oracle's 4-asset selection — 16× denser direction gradient. Forces latent
    # to preserve per-asset direction, which class-balanced BC alone can't do.
    use_direction_head: bool = True
    direction_horizon: int = 20             # K steps ahead (match oracle_lookahead)
    direction_hidden_dim: int = 256         # Head capacity
    direction_loss_weight: float = 0.2      # Loss weight; dense signal so start modest

    # === Direction → Policy Structural Bias (TASK_728) ===
    # Zero-init residual projection of direction_logits into policy-logit space.
    # Gives the policy head an explicit per-asset direction feature instead of
    # hoping it extracts direction from the bottlenecked latent.
    use_direction_policy_bias: bool = True

    # === Per-Asset Direction + Policy Routing (TASK_729a) ===
    # Route per-asset encoder features directly to DirectionHead and a new
    # PerAssetPolicyHead. Bypasses the 64-asset → state_dim global bottleneck
    # that capped dir_acc at 0.59 and Bal.Acc at 34%.
    # NOTE: DirectionHead signature changed — old checkpoints' direction_head
    # params are shape-incompatible; use --reset-direction-head to drop them.
    use_per_asset_policy_head: bool = True
    per_asset_policy_hidden_dim: int = 128

    # === Cross-Sectional Rank Oracle (TASK_730) ===
    # Per-asset rank-score head + MSE loss on normalized ranks of
    # future_cum_returns. Replaces the 4-of-64 HOLD-dominant top-k signal with
    # a dense per-asset target every step. Shares per_asset_feats with the
    # other 729a heads (no extra encoder pass).
    use_rank_head: bool = True
    rank_hidden_dim: int = 128
    rank_loss_weight: float = 1.0
    rank_loss_type: str = "mse"  # "mse" on normalized ranks (listwise/pairwise held as stretch)

    # === Flow-Matching Dynamics Loss (TASK_734, Path α-v1) ===
    # NOTE: provably equivalent to MSE under single-step supervised setup
    # (see notes §48). Flag retained as documented no-op; do not rely on it.
    use_flow_matching: bool = False
    flow_matching_velocity_weight: float = 0.8

    # === Auxiliary Log-Return Prediction Head (TASK_735, α-v3) ===
    # New prediction head filling the magnitude-awareness gap left by
    # direction_head (sign only) and rank_head (ordering only). Predicts per-
    # asset log-returns at horizons [1, 5, 10, 20], supervised with per-
    # horizon MSE against actual log-returns from the training window.
    use_aux_price_head: bool = True
    aux_price_hidden_dim: int = 256
    aux_price_horizons: tuple = (1, 5, 10, 20)
    aux_price_loss_weight: float = 0.1

    # === Bella Distillation Prediction Head (TASK_740, Path β-v2) ===
    # Mirrors aux_price_head but supervised with Bella-generated log-returns
    # precomputed offline (see tools/precompute_bella_predictions.py).
    # Chosen space: log-returns. Scale-stable across assets and aligned with
    # aux_price_head's convention — a single MSE criterion is meaningful.
    # When the manifest lacks a shard for a given period, the batch is filled
    # with sentinel NaN and the loss is masked out sample-by-sample.
    use_bella_distill_head: bool = False
    bella_distill_hidden_dim: int = 256
    bella_pred_horizons: tuple = (1, 5, 10, 20)
    bella_distill_loss_weight: float = 0.1
    bella_preds_manifest: str = "config/bella_preds_manifest.json"

    # === Bella Trajectory-Shape Distillation Head (TASK_754, β-v3) ===
    # Successor to TASK_740 — permutation-invariant 14-float trajectory-shape
    # descriptor (cross moments + horizon-1 moments + last-horizon moments
    # + SVD spectrum), supervised by per-component-weighted MSE on the
    # latent state. Mutually exclusive with use_bella_distill_head at
    # config-resolve time (enforced in main()).
    use_bella_shape_head: bool = False
    bella_shape_descriptor_dim: int = 14
    bella_shape_hidden_dim: int = 256
    bella_shape_loss_weight: float = 0.1
    bella_shape_manifest: str = "config/bella_shape_manifest.json"
    # Per-component weights for (mean, std, skew, kurt, s1, s1_ratio). Default
    # mirrors §4.1 of the TASK_754 spec; the cross/h1/hlast groups all share
    # (mean=w0, std=w1, skew=w2, kurt=w3), the SVD pair gets (s1=w4, s1_ratio=w5).
    bella_shape_component_weights: Optional[tuple] = None
    bella_shape_mode: str = "moments"  # {moments, lowrank, contrastive}; only "moments" implemented

    # === Bella as Data Augmentation (TASK_753, β-v4) ===
    # Successor framing to TASK_740/754: instead of distilling Bella's
    # predictions into a head, use Bella to *generate* the raw OHLCV the
    # encoder sees, side-by-side with the historical SegmentPool feed.
    # No new heads, no new loss surfaces — only the source of `hm.cache_data`
    # changes. Mutually exclusive with use_bella_distill_head and
    # use_bella_shape_head at config-resolve time (enforced in main()).
    bella_augment: bool = False
    bella_augment_ratio: float = 0.25      # P(slot is Bella-sourced) at slot init in 'append' mode
    bella_augment_mode: str = "append"     # {append, replace, interleave}; only "append" implemented
    bella_augment_regime_mode: str = "uniform"  # {uniform, match, unconditional}
    bella_augment_checkpoint: str = "checkpoints/diff_teacher/generator_best/generator"
    bella_augment_k_samples: int = 1       # Bella samples per generate_on_device call (1 = independent)
    # 0 = use generator's native seq_length (resolves spec §5 default of 1440 against
    # actual ckpt's 400). For values > native, BellaMarket chains chunks. Default
    # 800 (2 chained chunks) — enough headroom for episode_length=400 + window=64
    # + lookahead=20 + buffer ≈ 489 step pool slots.
    bella_augment_seq_length: int = 800
    bella_augment_refresh_epochs: int = 0  # 0 = generate once at pool init; >0 = refresh every N epochs
    bella_augment_volatility_scale: float = 0.015
    bella_augment_asset_policy: str = "remap"  # {remap, canonical, random}
    bella_augment_log_every: int = 100     # Log slot-build stats every N _make_bella_slot calls

    # === Cross-Sectional Channel Toggle (TASK_738 ablation) ===
    # True (default) → encoder emits 32 channels (27 per-asset + 5 cross-sectional).
    # False → legacy 27-channel encoder. Needed to load pre-TASK_732 Phase 1
    # checkpoints (jax_ckpt_easy_phase1_v3). Must match the Phase 1 base.
    use_cross_sectional: bool = True

    # === Oracle Mode (TASK_739) ===
    # Valid values:
    #   "topk_path"  — greedy top-k hindsight path (pre-TASK_739 behavior).
    #                  93.75% HOLD by construction (k=4 of 64 assets per step).
    #   "dense_sign" — per-asset sign(r_fwd) with ε dead-zone. 64 labels/step
    #                  across all assets; HOLD becomes a learned class.
    #   "hybrid"     — RESERVED (top-k drives value, dense-sign drives BC).
    #                  Not yet implemented; TODO(TASK_739 follow-up).
    # Only the BC target changes — oracle_selections and oracle_cum_pnl
    # (used for value/path) still come from the top-k solver.
    oracle_mode: str = "topk_path"
    sign_epsilon: float = 1e-4

    # === PopArt Value Normalization (TASK_718) ===
    use_popart: bool = True
    popart_beta: float = 0.0003  # EMA decay — faster than Poppy's 0.0001 (fewer total steps)
    td_discount: float = 0.95    # Discount for TD value targets (lower than mcts_discount=0.997
                                  # to keep returns in [-10, 10] support: V ≈ r/(1-γ) ≈ 2.0)
    reset_value_head: bool = True # Re-init value head at Stage B start (breaks Phase 1 saturation)
    oracle_value_horizon: int = 400  # Steps ahead for oracle remaining PnL (match episode length)
    nstep_horizon: int = 20          # N for n-step TD returns (TASK_719b)

    # === Value Target Mode (TASK_719) ===
    value_target_mode: str = "oracle"  # "oracle" = oracle remaining PnL, "td0" = TD(0) bootstrap

    # === Stage B Learning Rates ===
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    encoder_lr: float = 0.0        # Frozen initially, optionally 3e-5 after 200 epochs
    encoder_unfreeze_epoch: Optional[int] = None  # Epoch to unfreeze encoder (None = never)

    # === Factored Per-Asset Search (TASK_724) ===
    use_factored_search: bool = False  # True = factored, False = joint MCTS
    factored_temperature: float = 1.0  # Q-value → policy softmax temperature
    factored_sub_batch: int = 4096     # Sub-batch size for VRAM management

    # === Stage B Batch Size ===
    # Override: smaller batch for MCTS bottleneck (32 sims × 256 ≈ 8s/step)
    stage_b_batch_size: int = 256

    # === TASK_733 Passthrough (cheat-channel flag) ===
    # Mirrors EasyConfig.use_cheat_channel. Phase-2-side logic is NOT wired;
    # this exists purely so a Phase-1 cheat checkpoint's config round-trips
    # through Phase 2 load without unknown-field errors. Production Phase 2
    # must keep this False; the benchmark safety rail trips if it's True.
    use_cheat_channel: bool = False

    # === TASK_741 (Mint) Passthrough ===
    # Mirrors EasyConfig.use_per_asset_injection. Phase 2 honors these flags
    # (encoder picks up the FiLM subtree at model creation). Round-trip safe.
    use_per_asset_injection: bool = False
    per_asset_inject_xs: bool = False
    per_asset_inject_cheat: bool = False

    # === TASK_749 (Titans-MAG) ===
    # Plug a Titans-style neural long-term memory module (MAG variant) into
    # the V3 encoder's temporal compression stage. Memory updates inside the
    # forward pass via surprise-momentum gradient descent with weight decay
    # (paper eq. 13). Gated against the existing minGRU branch with a
    # large-negative-init sigmoid gate so step-0 output is byte-identical to
    # the V3-minGRU baseline (Phase-1 ckpts load unchanged).
    use_titans_memory: bool = False
    titans_variant: str = "mag"            # only "mag" implemented; "mac"/"mal" reserved
    titans_memory_depth: int = 2           # MLP depth L_M (paper §2.2: 2 ≫ 1 on long seqs)
    titans_memory_hidden: Optional[int] = None  # None → embed_dim
    titans_persistent_tokens: int = 4      # N_p prepended persistent memory tokens (paper §2.3)
    titans_surprise_momentum_init: float = 0.9   # initial bias for η_t sigmoid (retain past surprise)
    titans_decay_init: float = 0.01        # initial bias for α_t sigmoid (near-zero forgetting)
    titans_lr_init: float = 0.01           # initial bias for θ_t sigmoid (small inner-loop step)
    # Gate init = -5 → sigmoid(g) ≈ 6.7e-3 (memory contributes <1% of output at
    # step 0, well within numerical tolerance of "byte-identical" baseline) AND
    # sigmoid'(g) ≈ 6.6e-3 (large enough for gradient to actually move the gate
    # within reasonable epoch counts). Stricter -10 saturates the sigmoid so
    # hard that the gate doesn't move in <500 epochs at encoder_lr=3e-5.
    titans_gate_init: float = -5.0
    # TASK_749 fix 5: dedicated outer-LR for Titans params (gate + memory MLP +
    # persistent tokens + α/η/θ projections). Encoder LR (3e-5) is too low to
    # move the MAG gate in reasonable epoch counts (~1e-7 per step at -3 init,
    # ~1000× below the 1e-3 movement threshold). When None, Titans inherits
    # encoder_lr (legacy behavior, keeps prior smokes reproducible). When set,
    # an `optax.multi_transform` 'titans' group is created with this LR.
    titans_lr: Optional[float] = None

    # === Checkpointing ===
    phase2_checkpoint_dir: str = "jax_ckpt_easy_phase2"

    def get_stage_defaults(self):
        """Apply stage-specific default overrides. Call after construction."""
        if self.stage == "A":
            self.freeze_encoder = True
            self.freeze_policy = True
            self.freeze_value = True
            self.freeze_dynamics = False
        elif self.stage == "B":
            self.freeze_encoder = True  # Unfrozen later via encoder_unfreeze_epoch
            self.freeze_policy = False
            self.freeze_value = False
            self.freeze_dynamics = False


def get_phase2_config(phase1_config_path: str = None) -> Phase2Config:
    """Create Phase 2 config, optionally loading Phase 1 config as base."""
    if phase1_config_path:
        import json
        with open(phase1_config_path) as f:
            p1_data = json.load(f)
        # Only carry over Phase 1 fields that exist in EasyConfig
        easy_fields = {f.name for f in EasyConfig.__dataclass_fields__.values()}
        p1_filtered = {k: v for k, v in p1_data.items() if k in easy_fields}
        return Phase2Config(**p1_filtered)
    return Phase2Config()
