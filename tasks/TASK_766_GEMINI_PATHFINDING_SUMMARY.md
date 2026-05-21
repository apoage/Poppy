# TASK_766_GEMINI_PATHFINDING_SUMMARY

**Status**: GEMINI-DELIVERED
**Created**: 2026-05-21

## 1. Verdict on the Existing Roadmap
The operator's trigger calibration on 760 is correct, but the roadmap's framing of the Iris-DEX pivot is dangerously disconnected.

On Q1 (760 trigger): Declaring structural pivot at +0.35pp mean is correct. The promotion gate is +3pp. The noise floor at 3-seed is ~5pp. The best seed was +0.68pp. Even if the remaining two seeds of a 5-seed extension were exceptionally good (+6pp each), the mean would barely scratch the promotion gate. 760 is a textbook ceiling-bounded result; extending to 5 seeds burns ~18h of server wall to confirm what the first 3 seeds already proved—the mechanism is exhausted.

On Q4 (Iris-DEX framing): Framing 765 as a "standalone alpha-positive question" rather than an experiment is premature and risks throwing away months of tooling. If you fully pivot to DEX without a bridge, you won't know if the improvement comes from the 12s bar interval, the DEX asset universe, or the on-chain data purity. You must isolate the data variable. 

## 2. Recommended Re-ordering
**The operator's ordering (762 → 763 → 764) is correct for CLI-side work, but 764 should not be viewed as the final savior.**

1. **762 (Reward shaping)** (~15h) must remain first. It is the cheapest intervention and attacks the immediate symptom (trade explosion) of the current wave. 
2. **763 (Entropy reg)** (~19h) is second. It attacks the same symptom via the policy distribution.
3. **764 (Oracle k-reformulation)** (~35h) remains third. While it attacks the 5-month-old `[[bull-blindness]]` root cause, breaking the Phase 1 imprint contract is highly disruptive. It should only be run after the cheap Phase 2 stabilizers (762/763) have been tested.

## 3. Highest-EV Addition
**Action Space Restructure (Drop Limit Orders)**. 
The current 320-dim action space (64 assets × 5 actions) includes `LimitSell` and `LimitBuy`. These actions dilute the policy distribution and heavily rely on the simulator's limit order execution fidelity, which is notoriously noisy. By restructuring to a 192-dim space (64 assets × 3 actions: Sell/Hold/Buy), we reduce the MCTS exploration branching factor exponentially and condense the policy's probability mass. This is a Phase 2-only change that directly attacks the "wasting capacity / over-trading" hypothesis without breaking Phase 1. 

## 4. EV-per-Server-Hour Ranking
1. **Direction 1 (TASK_762: Reward Shaping)**: ~15h. High EV. Cleanly falsifies the noise-trade penalty hypothesis.
2. **New Direction (TASK_767: Action Space Restructure)**: ~12h. High EV. Phase 2 only. Radically shrinks the search space.
3. **Direction 2 (TASK_763: Entropy Reg)**: ~19h. Medium EV. Mechanism-aligned but tuning is notoriously finicky.
4. **Direction 3 (TASK_764: Oracle k-reformulation)**: ~35h. Medium EV. Highest single-component ceiling, but costs Phase 1 comparability.
5. **Direction 4 (TASK_765: Incompatible Dataset)**: ~300h+. Low EV/hr. High absolute ceiling, but a massive wall-time sink. Must be reframed as a bridge experiment.
