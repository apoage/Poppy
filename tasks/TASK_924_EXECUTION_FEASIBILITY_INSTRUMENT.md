> **STATUS (2026-09-28): SHIPPED (run 57, tag `task924-exec-feasibility`).** G1 PASS on B1's snapshot
> (byte-identical prefix of the final capture); G2/G3 pass. Pre-registered verdict UNBLOCKED_WITH_NUMBERS
> but the gate is NON-DISCRIMINATING (recorder suppresses one-sided rows; AS(1h) = spread, not markout).
> Gap-clipped frozen-mid BTC 30.42% (gap-blind 43.39%); time-weighted survival p50 BTC 32.5 s. Path
> correction: the capture is at srv `~/Trader/runs/907_live_fill_probe/`, not `~/captures/`. Option B
> go/no-go → **D87** (operator).

# TASK_924 — execution-feasibility instrument over the closed capture (Option B's pre-gate)

**Status**: PROPOSED 2026-09-28 (planner; operator handed unattended time off
`docs/profitability_paths_2026-09-28.md` — this is Option B's ungated half).
**Size XS/S. Risk: LOW — new module in its own directory, READ-ONLY over the capture
artifact; zero canon surface, zero DB access, no live anything.**
**Verify: full.**

## Why this row exists

Option B of the recomposition (planner repurposed from direction-selection to learned
optimal execution) lives or dies on maker feasibility on this venue. 907 B1 already
measured the warning: BTC/USD mid frozen ~46.5% of the time. But the B2 re-cut that would
measure this properly is **D81-gated (do NOT patch the 907 probe)**. This row gets the
DECISION NUMBERS with zero probe-patching: a new, descriptive microstructure instrument
over the ALREADY-CLOSED capture artifact, feeding option B's go/no-go with data instead of
opinion.

## The wall (named, binding)

**D81**: do NOT modify `experiments/907_live_fill_probe/{capture.py,recorder.py,
health_probe.py,render.py,cli.py,estimands.py,shadow_engine.py,reference_fill_model.py}`
or re-litigate any 907 markout gate. This row IMPORTS/READS via the shipped streaming
reader; it never edits that tree. If a needed read is impossible without editing those
files, HALT with a note — that is a D81 question, not an implementation choice.

## What (all read-only over `~/captures/capture_907_main.jsonl` on gpu-server, via the
shipped 914 streaming reader, O(1) memory, wrapped `safe_run` ≤4G)

Pre-registered outputs (SPEC.md sha-pinned before the first full pass; fixture/smoke JSONL
first):

1. **Rel-spread surface**: hourly distribution of relative half-spread (bp) per pair,
   book-present rows only; median/quartiles by hour-of-day, with the D74 gap-exclusion
   (wall-clock-locked stall gaps are recorded gaps — reuse the capture's own gap marks,
   do not re-derive).
2. **Mid-staleness**: frozen-mid fraction + streak-duration distribution (per pair), on
   the SAME definition B1 used (46.5% BTC/USD must reproduce within tolerance — an
   instrument that can't reproduce the shipped number is wrong, not novel).
3. **Book presence**: fraction of event rows with two-sided book / one-sided / print-only;
   depth-at-best sample statistics.
4. **Quote-update rate**: book-refresh events per active minute; the maker-relevant
   "how long does a stale quote actually persist" survival curve.
5. **AS(1h) first estimate**: mean |half-spread| over 1h windows with live two-sided book,
   reported against the **≤ 2 bp** Phase-B gate line — labelled DESCRIPTIVE: it bounds
   what passive fills COULD cost, it is not 907's markout gate and does not unlock Phase B
   by itself.

Artifact: `experiments/924_execution_feasibility/{SPEC.md,results.json,RESULTS.md}` —
RESULTS.md carries the go/no-go table for Option B (maker viability by pair/hour) and the
explicit "what this does NOT establish" list.

## Gates (binary)

- **G1 REPRO**: the instrument reproduces B1's frozen-mid 46.5% (BTC/USD) within a
  stated tolerance computed in the SPEC; a mismatch is a BLOCKER, not a footnote.
- **G2 READ-ONLY PROOF**: suite proves no write anywhere outside the experiment dir
  (srv-side: the capture file's sha256 + mtime before/after, asserted).
- **G3 RSS**: peak RSS < 4 G through the full 1.44 GB pass (the 914 reader's contract).
- **G4 PRE-REGISTRATION**: SPEC sha recorded before the first full-pass run; any output
  added after seeing data is labelled POST-HOC and excluded from the go/no-go table.

## Out of scope

Any markout/fill simulation (that is 907's gated lane); any venue comparison; any
order-book-based SIGNAL (the parked L2-mining row is the operator's pre-registration call);
any DB read; any write to srv outside the experiment dir.

## Consequences (pre-committed)

- Two-sided book present < ~50% of active time across pairs, or AS(1h) descriptive ≫ 2 bp
  everywhere → Option B's maker premise is measured-dead on this venue; the recomposition
  record says so and B closes to "taker-only planner", which the 916 arithmetic already
  prices.
- Otherwise → B's build row (planner-as-executor spec) is unblocked WITH numbers, and the
  next row cites this artifact's tables.
