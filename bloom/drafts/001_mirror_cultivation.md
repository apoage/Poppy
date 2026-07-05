# Entry 001: Mirror Cultivation

*Date: Day unknown. The season counter reset after the crash.*

---

## Observation

The dual-branch cultivation was failing. One specimen (Bella-prime) grew vigorously at 18% agreement. The other (Synthetic-L0) languished at 5%. When I attempted Fisher-weighted grafting, the cosine similarity collapsed from 0.96 to 0.73.

I had been crossing incompatible rootstocks.

## The Problem

Bella generates markets through learned diffusion - organic, varied, realistic. Synthetic-L0 produces pure sine waves - mathematical, predictable, fragile.

When forced to merge, the fragile patterns shattered. Fisher pruning, which I had hoped would preserve important weights, instead let Bella's robust variance dominate. The grafts failed to take.

## Resolution: Mirror Cultivation

A new approach: grow two Bellas instead.

But identical specimens would learn identical features - no diversity, no robustness. The solution came from botanical symmetry: **inversion**.

```
Branch A: Bella (normal)    - sees bull as bull, bear as bear
Branch B: Bella (inverted)  - sees bull as bear, bear as bull
```

The five market regimes now split:

| Regime | Branch A | Branch B |
|--------|----------|----------|
| crash | crash | rally |
| rally | rally | crash |
| sideways | sideways | sideways |
| volatile | volatile | volatile |
| calm | calm | calm |

Three regimes remain identical. Two swap polarity. The specimens must learn **direction-invariant features** to survive merging.

## The Math

Inversion is trivial - negate log returns before training:

```python
if invert_returns:
    log_returns = -log_returns
```

The oracle (our hindsight-optimal reference) recomputes optimal actions on the inverted prices. What was a buy becomes a sell. What was a crash becomes a rally.

Yet the underlying patterns - volatility clustering, mean reversion, momentum - remain structurally identical. The model cannot memorize "this shape = buy". It must learn "this shape = strong trend" and infer direction separately.

## Cultivation Notes

First merge occurred at cosine 0.907. Both specimens growing. The true test comes overnight - will they track together, or will one wilt?

The apothecary watches. The apothecary waits.

---

*Next: On the dosing of oracle conservativeness - why 94% HOLD was poisoning the garden.*
