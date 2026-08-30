> **STATUS (2026-08-30): SHIPPED** — D68's four safety pins are mechanism now.
> 3 cycles, 5 verification passes (security + fail-closed + regression, then a fail-closed
> re-verify). 271 passed; selftest 12/12. All three gates closed, each site mutation-tested
> RED-before/green-after.
>
> **The story worth reading before touching the fence again:** all three cycles failed on the
> SAME defect in different clothes — *a file that should be scanned was silently not scanned*.
> Cycle 1 the glob was non-recursive (`venue/order_client.py` invisible). Cycle 2's `_SKIP_DIRS`
> fix matched at ANY depth (`venue/build/order_client.py` invisible again). Cycle 3 split it into
> `ALWAYS_SKIP` (caches, any depth) + `TOP_LEVEL_ONLY_SKIP` (ambiguous names, lane root only) and
> made symlinked dirs a REFUSAL. Each blind spot survived the previous cycle's own mutation
> testing because that testing planted poison exactly where the mechanism already worked.
>
> **Remainders, deliberately not fixed:** (a) the fence is a literal-substring grep and is NOT
> proof against deliberate obfuscation — now stated in its docstring instead of overclaimed; an
> AST-based fence is past D68's ruled shape and is a planner call; (b) `getattr( os` (with a
> space) or a line-wrapped variant escapes it; (c) the symlink refusal is lane-wide, so one stray
> symlink blocks the whole lane's CI; (d) case-insensitive matching will false-positive on
> plausible order-book identifiers (`AddOrderBook`, `add_order_level`) — deliberate, and the
> documented correct response is to rename or add a reviewed exception, NEVER to narrow the
> tokens or restore a per-file allowlist, which is how the cycle-1 defect got in; (e) the fence
> covers `.py` only — a credential or order frame in a `.jsonl`/`.json`/`.toml` fixture (the lane
> has one) is out of scope.

# TASK_910 — pin the 907 lane's safety invariants as mechanism, not prose (D68)

**Status**: READY 2026-08-27 (planner; seeded by D68's ruling, option (a) — the lane's most
important safety property is currently a docstring, and phase 2 is the row that sends real
money).
**Size: XS. Risk: LOW — purely additive; touches no number, bound, or canon.**

## Scope (D68's four fixes, all demonstrated-needed in run 44)

1. **Pin the no-order-path invariant.** Tests that go RED when: (i) anything other than the two
   public subscribe frames is ever sent (count ALL frames, not just `method == 'subscribe'`
   ones — cycle 4's coupling test filtered before counting, which is the demonstrated hole);
   (ii) the connect target is anything but the pinned public `WS_URL` literal.
2. **Re-raise the refusal classes above the broad catch.** `recorder.py:297`'s
   `except Exception` must re-raise `NotImplementedError`/`Phase2RefusalError`/`RailViolation`
   first — RE-RAISE, do not narrow the clause (902 measured that narrowing an `except` proves
   illusory here).
3. **The credential fence covers the NEXT increment, not just today's files:** glob `*.py` over
   the lane directory (no filename allowlist), and add `os.environ`, `getenv`, `add_order`,
   `AddOrder` to `FORBIDDEN`.
4. **Caps are MANDATORY at arming (D68(b) in shape):** `--budget-cap-usd` AND
   `--max-order-usd` both required to arm; `check_total_cap` called unconditionally; caps
   enforced at the order boundary, not only at parse. The VALUES stay the operator's at arm
   time. (The phase-2 placement increment itself remains its own future row; this row only
   makes the contract mandatory-and-enforced so no future increment can arm weakly.)

## Acceptance (gates)

- (i) Each of the four run-44 demonstrations now goes RED (mutation-tested at each site).
- (ii) 907's own suite stays green (173+ passed); selftest 12/12.
- (iii) The 907 task file's gate (ii) sentence — currently claiming the caps fire — is replaced
  by the true statement (done planner-side at ruling; this row verifies the new wording against
  the code).

## Out of scope

Live order placement (phase 2's own row). Any change to phase-1 capture behavior.
