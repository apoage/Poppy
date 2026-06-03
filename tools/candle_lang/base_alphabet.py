"""
Base alphabet — data-adaptive product quantization of the vol-unit substrate (design 01 §2.2).

Each per-bar vol-unit tuple (6 channels, config.CHANNELS order) is mapped to a single integer
**base symbol** via a product alphabet (a 1d-SAX generalization). Four independent axes are
quantized, then combined mixed-radix into one symbol id:

    axis        source channel(s)              cardinality (defaults from config.ALPHABET)
    ----        ----------------               ------------------------------------------
    move        mv      (idx 0)                ALPHABET['move']      = 8   signed quantile bins
    gap         gap     (idx 1)                ALPHABET['gap']       = 3   signed quantile bins
    bodyrange   |body|+|range| (idx 2,3)       ALPHABET['bodyrange'] = 4   magnitude quantile bins
    wick        uw - lw (idx 4,5)              ALPHABET['wick']      = 3   asymmetry quantile bins

    N_SYMBOLS = 8 * 3 * 4 * 3 = 288 (defaults).

Breakpoints are fitted **once** on the POOLED vol-unit corpus with `np.quantile`. The vol-unit
distribution is fat-tailed (NOT Gaussian), so we never assume Gaussian SAX breakpoints — quantile
edges adapt to the empirical distribution and put roughly equal mass per bin. The single shared bin
table is what makes the same relative shape land on the same base symbol across assets ($0.007 vs
$105,747), because the substrate is already rescale-invariant (volunit.py / eval A0).

A card-K axis stores K-1 interior edges (the empirical quantiles at the K-1 internal fractions). At
encode time `np.digitize(value, edges)` returns a bin index in [0, K-1] — exactly K bins. Edges are
made strictly increasing (tiny jitter on ties from a degenerate / low-variance pooled distribution)
so `np.digitize` is well defined and every value lands in a valid bin.

NO GPU / JAX needed — pure numpy, CPU. Shared by every symbolic panel member (L1/L2/L3/L5) and by the
eval battery's A1 MDL accounting.
"""
import numpy as np

from . import config as C  # noqa: E402  (package-relative)


# Axis order is FIXED — it defines the mixed-radix combination and must never change once a table
# is fitted (the symbol ids are positionally meaningful). The radix of each axis is its cardinality.
_AXES = ("move", "gap", "bodyrange", "wick")


# --------------------------------------------------------------------------- #
# per-axis scalar extraction from the 6-channel substrate
# --------------------------------------------------------------------------- #
def _axis_values(feats: np.ndarray) -> dict:
    """Reduce the 6-channel vol-unit features to one scalar per quantization axis.

    feats: (..., N_FEAT) float — channels in config.CHANNELS order
        0 mv, 1 gap, 2 body, 3 range, 4 uw, 5 lw
    Returns dict(axis_name -> array of axis scalars with feats' leading shape).
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.shape[-1] != C.N_FEAT:
        raise ValueError(f"feats last dim must be N_FEAT={C.N_FEAT}, got {feats.shape[-1]}")
    mv = feats[..., 0]
    gap = feats[..., 1]
    body = feats[..., 2]
    rng = feats[..., 3]
    uw = feats[..., 4]
    lw = feats[..., 5]
    return {
        "move": mv,                       # signed vol-unit inter-bar move (token-identity magnitude)
        "gap": gap,                       # signed vol-unit inter-bar gap
        "bodyrange": np.abs(body) + np.abs(rng),  # non-negative joint bar-energy magnitude
        "wick": uw - lw,                  # signed wick asymmetry in [-1, 1]
    }


def _interior_edges(values: np.ndarray, card: int) -> np.ndarray:
    """K-1 interior quantile edges for a card-K axis on the pooled `values`.

    Quantile fractions are the K-1 internal boundaries (i/K for i in 1..K-1) so each bin carries
    ~equal empirical mass (fat-tail-aware: edges follow the data, not a Gaussian assumption).
    Edges are forced strictly increasing — a degenerate / near-constant pooled axis (e.g. all-zero
    wick) would otherwise produce duplicate edges and ill-defined digitize bins.
    """
    card = int(card)
    if card < 1:
        raise ValueError(f"axis cardinality must be >= 1, got {card}")
    if card == 1:
        return np.empty((0,), dtype=np.float64)
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        # No data to fit (should not happen on a real pooled corpus): fall back to symmetric
        # unit edges so encode_symbols is still well defined.
        fracs = np.arange(1, card, dtype=np.float64) / card
        edges = (fracs - 0.5) * 2.0
        return _ensure_increasing(edges)
    fracs = np.arange(1, card, dtype=np.float64) / card
    edges = np.quantile(v, fracs)
    return _ensure_increasing(np.asarray(edges, dtype=np.float64))


def _ensure_increasing(edges: np.ndarray) -> np.ndarray:
    """Make `edges` strictly increasing by nudging non-increasing entries upward by a tiny epsilon
    scaled to the data magnitude — keeps np.digitize boundaries well defined on degenerate axes."""
    edges = np.array(edges, dtype=np.float64, copy=True)
    if edges.size <= 1:
        return edges
    scale = max(1.0, float(np.max(np.abs(edges))))
    bump = 1e-9 * scale
    for i in range(1, edges.size):
        if not (edges[i] > edges[i - 1]):
            edges[i] = edges[i - 1] + bump
    return edges


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def fit_alphabet(feats_pooled: np.ndarray, cfg: "C.CandleConfig") -> dict:
    """Fit per-axis quantile breakpoints on the POOLED vol-unit distribution (design §2.2).

    feats_pooled : (N, N_FEAT) float   pooled vol-unit features across the build assets
    cfg          : CandleConfig        (cardinalities come from config.ALPHABET, not cfg)

    Returns a frozen-style `table` dict:
        {
          "axes"        : ("move","gap","bodyrange","wick"),   # FIXED order (mixed-radix layout)
          "cardinalities": {axis: int},                        # = config.ALPHABET
          "edges"       : {axis: np.ndarray(card-1,) float64}, # strictly-increasing interior edges
          "n_symbols"   : int,                                 # product of cardinalities
          "n_fit"       : int,                                 # rows used to fit (bookkeeping)
        }

    Cardinalities are read from config.ALPHABET (the contract: move 8 / gap 3 / bodyrange 4 / wick 3,
    product = 288). `cfg` is accepted for interface symmetry / future per-config overrides.
    """
    feats_pooled = np.asarray(feats_pooled, dtype=np.float64)
    if feats_pooled.ndim != 2 or feats_pooled.shape[1] != C.N_FEAT:
        raise ValueError(
            f"feats_pooled must be (N, {C.N_FEAT}), got {feats_pooled.shape}"
        )
    cards = {ax: int(C.ALPHABET[ax]) for ax in _AXES}
    axis_vals = _axis_values(feats_pooled)
    edges = {ax: _interior_edges(axis_vals[ax], cards[ax]) for ax in _AXES}

    n_symbols = 1
    for ax in _AXES:
        n_symbols *= cards[ax]

    return {
        "axes": _AXES,
        "cardinalities": cards,
        "edges": edges,
        "n_symbols": int(n_symbols),
        "n_fit": int(feats_pooled.shape[0]),
    }


def N_SYMBOLS(table: dict) -> int:
    """Total base-symbol cardinality = product of per-axis cardinalities (== 288 for defaults)."""
    if "n_symbols" in table:
        return int(table["n_symbols"])
    prod = 1
    for ax in table["axes"]:
        prod *= int(table["cardinalities"][ax])
    return int(prod)


def encode_symbols(feats: np.ndarray, table: dict) -> np.ndarray:
    """Encode vol-unit feature rows to base-symbol ids (design §2.2).

    feats : (T, N_FEAT) float   vol-unit features, config.CHANNELS order
    table : output of fit_alphabet

    Per axis: `np.digitize(value, edges)` -> bin index in [0, card-1] (edges has card-1 entries, so
    digitize yields card bins). Axes are combined mixed-radix in the FIXED `table["axes"]` order:
        id = ((b_move * card_gap + b_gap) * card_bodyrange + b_bodyrange) * card_wick + b_wick
    -> a single integer in [0, N_SYMBOLS(table)).

    Returns symbols : (T,) int64.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2 or feats.shape[1] != C.N_FEAT:
        raise ValueError(f"feats must be (T, {C.N_FEAT}), got {feats.shape}")
    axes = table["axes"]
    cards = table["cardinalities"]
    edges = table["edges"]
    axis_vals = _axis_values(feats)

    T = feats.shape[0]
    symbols = np.zeros(T, dtype=np.int64)
    for ax in axes:
        card = int(cards[ax])
        e = np.asarray(edges[ax], dtype=np.float64)
        # right=False: value < e[0] -> bin 0; value >= e[-1] -> bin card-1. NaNs (if any slipped
        # through upstream finite filtering) digitize to the top bin; clip keeps ids in range.
        bins = np.digitize(axis_vals[ax], e, right=False)
        bins = np.clip(bins, 0, card - 1).astype(np.int64)
        symbols = symbols * card + bins
    return symbols


if __name__ == "__main__":
    # CPU self-test: fit on a synthetic fat-tailed pooled corpus, check id range + rescale path.
    rng = np.random.default_rng(0)
    N = 200_000
    # Student-t-ish heavy tails for the vol-unit magnitude axes; wick fractions in [0,1].
    mv = rng.standard_t(df=3, size=N)
    gap = 0.3 * rng.standard_t(df=3, size=N)
    body = rng.standard_t(df=4, size=N)
    rngc = np.abs(rng.standard_t(df=4, size=N)) + 0.1
    uw = rng.random(N)
    lw = rng.random(N)
    feats = np.stack([mv, gap, body, rngc, uw, lw], axis=1).astype(np.float32)

    cfg = C.CandleConfig()
    table = fit_alphabet(feats, cfg)
    n = N_SYMBOLS(table)
    print(f"N_SYMBOLS = {n}  (expected 288)")
    print("cardinalities:", table["cardinalities"])
    for ax in table["axes"]:
        print(f"  edges[{ax}] (card={table['cardinalities'][ax]}):", np.round(table["edges"][ax], 4))

    syms = encode_symbols(feats, table)
    print(f"symbols shape {syms.shape} dtype {syms.dtype} "
          f"min {syms.min()} max {syms.max()}  (must be in [0,{n}))")
    assert syms.min() >= 0 and syms.max() < n, "symbol ids out of range!"

    # Per-axis bin occupancy should be near-uniform (quantile edges) for the unbounded axes.
    occ = np.bincount(syms, minlength=n)
    print(f"distinct symbols used: {(occ > 0).sum()} / {n}")

    # Rescale-invariance sanity (the table operates on already-rescale-invariant substrate, so the
    # SAME feature rows must yield byte-identical symbols on a second pass).
    syms2 = encode_symbols(feats.copy(), table)
    print(f"deterministic re-encode identical: {np.array_equal(syms, syms2)}")
