"""
Shared CONTRACT for the candle-language tokenizer panel (TASK_789/791/792/795, branch C).

THIS FILE IS THE INTERFACE CONTRACT. Every panel member (L1-L5), the streamer, the
eval battery, and the L4 trainer import constants and shapes from here. Do not fork
the constants; do not redefine the batch shape elsewhere.

Design source of truth: docs/candle_language/01_design_stage1_candle_language.md (§2.1),
build plan 04_jax_build_plan.md, leak/OOM audit 05_leak_oom_audit.md.

--------------------------------------------------------------------------------
VOL-UNIT SUBSTRATE (design §2.1) — 6 causal, scale-free per-bar channels.
Channel order is FIXED (index -> name in CHANNELS):
    0 mv    = log(c_t/c_{t-1}) / sigma_t        vol-unit inter-bar move (token-identity MAGNITUDE, fork F1=A)
    1 gap   = log(o_t/c_{t-1}) / sigma_t        inter-bar gap (minute bars gap)
    2 body  = log(c_t/o_t)     / sigma_t        intrabar direction/strength, vol-unit
    3 range = log(h_t/l_t)     / sigma_t        bar extent, vol-unit
    4 uw    = (h_t - max(o,c)) / (h_t-l_t+eps)  upper-wick FRACTION in [0,1] (NO vol norm; scale-free)
    5 lw    = (min(o,c) - l_t) / (h_t-l_t+eps)  lower-wick FRACTION in [0,1]
where sigma_t = causal EWMA(|log(c/c_{t-1})|, halflife=H), floored at sigma_floor,
re-anchored every DAY_MINUTES (UTC-day) per design §2.5.

RESCALE INVARIANCE (eval A0): a pure price rescale c->lambda*c leaves every log-ratio
AND sigma_t unchanged, and the wick fractions are ratios of price differences -> unchanged.
So the 6-channel feature row is byte-identical under rescale -> identical tokens. This is the
load-bearing property the whole "transfer-by-construction" claim rests on.

--------------------------------------------------------------------------------
BATCH CONTRACT (db_stream.iter_batches -> all consumers). Fixed-shape numpy, B == cfg.batch ALWAYS:
    X        : (B, W, N_FEAT)  float32   vol-unit features; all-valid; sliced WITHIN a dt==60 run only
    C        : (B, W)          float32   raw close path over the window (for A2 de-norm price fidelity)
    asset_id : (B,)            int32     bookkeeping ONLY (no cross-asset coupling enters the model)
    end_ts   : (B,)            int64     end-bar timestamp (chrono split / A8 purge)
    valid    : (B,)            bool       False for PAD rows in the final tail batch -> mask the loss
The tail batch is PADDED to cfg.batch (never emitted short) so the jit step never retraces
(audit finding 3c). Windows NEVER straddle a gap and are NEVER forward-filled (history.py:375-389
forward-fill is FORBIDDEN for branch C; gaps are missing rows = dt!=60, not flat bars).
--------------------------------------------------------------------------------
"""
from dataclasses import dataclass, field
import os

# ---- substrate (FIXED) ----
CHANNELS = ("mv", "gap", "body", "range", "uw", "lw")
N_FEAT = len(CHANNELS)            # 6
BAR_SECONDS = 60                  # MINUTE granularity (reframe correction #1: NOT hourly)
DAY_MINUTES = 1440               # sigma re-anchor cadence (design §2.5; 24/7 corpus -> UTC-day, not session)
EPS = 1e-12                      # log-ratio numerator/denominator guard
WICK_EPS = 1e-9                  # wick-fraction denominator guard
DIV_EPS = 1e-10                  # generic division guard (CLAUDE.md JAX pitfall: jnp.maximum(x,1e-10))

# ---- defaults ----
DEFAULT_WINDOW = 64
DEFAULT_FSQ_LEVELS = (8, 8, 8, 5, 5)   # implicit codebook = prod = 12800
DEFAULT_EWMA_HALFLIFE = 64             # minutes
DEFAULT_SIGMA_FLOOR = 1e-4             # floor on per-bar |logret| vol (flat-regime blow-up guard)
DEFAULT_BATCH = 1024
DEFAULT_PREFETCH = 3                   # queue.Queue(maxsize) — THE RAM bound + backpressure (HARD-REQ #2)
PAGE_ROWS = 262144                     # paged-cursor page size; never fetchall a multi-million-row asset

# Base alphabet cardinality (design §2.2: product quantization, fitted ONCE on pooled vol-unit corpus)
ALPHABET = dict(move=8, gap=3, bodyrange=4, wick=3)   # base-symbol product cardinality = 8*3*4*3 = 288

# ---- corpus (1193 assets in DB; default = wide-price-span liquid subset) ----
# BUILD = fit/merge the vocab here. HOLDOUT = A8 cross-asset transfer test at EXTREME price levels.
# Row-count + price survey 2026-06-03: build ~$1–$30k, holdout ~$0.007–$3.5 (7 orders of magnitude span).
DEFAULT_BUILD_ASSETS = (1087, 1091, 353, 359, 586, 1019, 589, 384, 309)
DEFAULT_HOLDOUT_ASSETS = (365, 1118, 1121, 711, 61)


def db_path() -> str:
    """DB path resolution: $CANDLE_DB env override, else REPO-ROOT-RELATIVE (the DB sits at
    repo root on both the workstation and gpu-server — `/home/lukas/Trader/` vs
    `/home/operator/Trader/`). This auto-resolves on either host without an env var; NEVER hardcode
    a single host's path into a shipped module (the data.py:28 / run_gate.py:45 anti-pattern that
    bit the 2026-06-03 first server launch). A --db CLI flag still overrides via CandleConfig(db=...)."""
    env = os.environ.get("CANDLE_DB")
    if env:
        return env
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "trading_data.db")


@dataclass(frozen=True)
class CandleConfig:
    """Single config object threaded through every module. frozen -> safe as a static arg."""
    window: int = DEFAULT_WINDOW
    fsq_levels: tuple = DEFAULT_FSQ_LEVELS
    ewma_halflife: int = DEFAULT_EWMA_HALFLIFE
    sigma_floor: float = DEFAULT_SIGMA_FLOOR
    batch: int = DEFAULT_BATCH
    prefetch: int = DEFAULT_PREFETCH
    page_rows: int = PAGE_ROWS
    build_assets: tuple = DEFAULT_BUILD_ASSETS
    holdout_assets: tuple = DEFAULT_HOLDOUT_ASSETS
    db: str = field(default_factory=db_path)

    @property
    def codebook_size(self) -> int:
        prod = 1
        for L in self.fsq_levels:
            prod *= int(L)
        return prod
