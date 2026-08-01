> **[AUTHORIZED 2026-08-01 — D31 RULED by the OPERATOR: opt (a) as amended. Attribution
> resolved: the "unattributed actor" was the planner session writing while the factory was
> armed; the orchestrator's quarantine was correct and is preserved in the D31 row. This is now
> a QUEUED task, SERIAL after TASK_868.]**

# TASK_869 — Re-base-wave semantics: UTC window boundaries + masked daily teacher floor (D31 (a3))

**Status**: READY 2026-08-01 (D31 RULED by the OPERATOR). SERIAL after TASK_868
(same surface, same re-base wave — these two ship together before any reference is re-measured).
**Size: S. Risk: low-medium** (both changes move recorded numbers by design; neither may land
outside the wave).

## Why (run-23 recommended rows A + B, ruled into the D31 re-base wave)

Every minute-panel number re-bases under TASK_868 anyway, so the two remaining known
measurement-semantics defects land in the SAME wave — one re-base, not three:

- **(B→scope 1) `_to_epoch` resolves `--start/--end` in LOCAL time**, so a benchmark window's
  extent depends on which machine ran it — measured: a 308-row CEST offset on the OOS window;
  the reason several counts differ by one between parties (1734 vs 1735 panel rows, 2 vs 3 stale
  tails, 90 vs 91 daily fabricated rows). A determinism hazard for a read-of-record operator
  that runs on both the workstation and `gpu-server`.
- **(A→scope 2) The daily-framed teacher floor walks UNMASKED** — same defect class as D30
  rider (a); fixing it moves shipped daily-pinned numerics, which is why it was a planner call
  and is only sanctioned inside this wave.

## Scope

1. **`_to_epoch` → UTC**: window boundaries resolve in UTC everywhere (loaders, floors,
   scorecard manifest windows, fixture smokes). The artifact records the resolution
   (`window_utc: [start_epoch, end_epoch]`) so any two machines produce identical windows.
   Grandfathering: refuse nothing, but any artifact comparison across the boundary must show
   the window epochs, not just the date strings.
2. **Mask the daily teacher floor** exactly as the minute floor was masked (D30 rider (a) +
   865's `_ffill_interior_only` semantics as ruled in D31): price-validity threaded, causal,
   disclosed. The rider-(a)-is-inert finding on the minute path does NOT transfer — measure,
   don't assume; if the daily mask is also inert, it ships as defence-in-depth with an honest
   docstring, same as 865 did.
3. **Numeric attribution**: every recorded numeric that moves (fixture baselines, smoke
   expectations, scorecard cards) is re-derived with the per-field diff attributed to (1) or
   (2) explicitly — the 864 promote gate is the write path for tracked baselines.

## Acceptance (gates)

- (i) Same command, TZ=Europe/Prague vs TZ=UTC: identical artifacts byte-for-byte (the
  308-row-class offset is gone); demonstrated on the OOS window.
- (ii) Daily floor mask: mutation-style proof both ways (a synthetic stale daily span changes
  the unmasked walk and not the masked one — or the inertness is measured and documented).
- (iii) All suites green; every moved numeric attributed; daily-pin PV conservation holds.

## Out of scope

Alias union / interior mask (TASK_868); re-measuring references (planner 861 leg 0);
any change to window SEMANTICS beyond timezone (start/end inclusivity etc. stay as-is).
