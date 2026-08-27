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
