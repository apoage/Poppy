# TASK_822 — Workstation stability investigation (sync floods survive RAM swap)

**Status**: ACTIVE (2026-07-05 ~23:35). Operator called full interrupt: "there cannot be pair
mismatch since I removed two sticks — run full investigation and stress test."

## Evidence

Tonight's boots (all times CEST):
| window | config / load | ended by |
|---|---|---|
| 21:13–22:54 | mixed 4-DIMM, 821 attempt-1 training ~19 min | **sync flood** |
| 22:55–23:06 | mixed, attempt-2 loading | reset (operator → BIOS, DRAM 2400) |
| 23:14–23:20 | mixed @2400 | reset (operator → RAM swap) |
| 23:25–23:30 | **matched 2×16GB FURY, 32GB**, attempt-4 loading | **sync flood** |

**Decisive:** MCE records bit-identical across the two RAM configs —
`Bank 0: baa00000000c0135  SYND 4d000014  IPID 1000b000000000` (crash #1 mixed, CPU 9;
crash #4 matched, CPU 1). Zen3 Bank 0 = **Load-Store unit**. Same fault, different DIMMs
→ **RAM kit hypothesis falsified as root cause** (June's diagnosis incomplete — mixed kit
was at most an aggravator). June history: ≥5 floods incl. AT IDLE, same banks (0 + 22).

## Hypotheses (ranked)

1. **Zen low-load voltage instability** ("Ryzen idle crash") — fits June's at-idle floods;
   canonical fix = BIOS *Power Supply Idle Control → Typical Current Idle*.
2. **SoC/IMC voltage marginality under bandwidth load** — fits tonight's crashes during the
   memory-heavy data-load/training phases; fix = VSOC ≈1.05–1.10V, VDDG bump.
3. Degrading 5800X (LS/IMC path) — if BIOS knobs don't hold, CPU swap/RMA.
4. Board (slot/trace/power delivery), 5. PSU transients, 6. BIOS/AGESA bug (3634, 2025-09).

## Stress ladder (attribute the domain; each stage in screen, logging to disk)

- **S1 RAM verify (RUNNING 23:36)** — `tools/ram_stress_verify.py --gb 18`, checksum-verified
  pattern loop, log `runs/822_stability/ram_stress.log`. Corruption before crash → DIMM/IMC;
  crash without corruption → fabric/voltage; ≥45 min clean → escalate.
- **S2 CPU+cache** — `stress-ng --matrix 0 --cache 0 --cpu 0 -t 30m` (needs sudo install).
- **S3 GPU** — JAX matmul burn, MEM_FRACTION=0.7, 20 min.
- **S4 combined** — S1(12GB) + S2 + S3 together ≈ worst-case real workload.
- Post-any-crash: reset reason + `ras-mc-ctl --errors` diff.

## FINDINGS 23:45 (operator dmidecode + ras-mc-ctl)

- **Sticks are in DIMM_A1 + DIMM_B1 — the WRONG slots** for 2-DIMM on PRIME B550-PLUS
  (daisy-chain topology wants **A2+B2**). Electrically worst 2-DIMM config; at 3200 MT/s a
  credible sync-flood source on its own. Softens the CPU-fault reading: LS-bank MCE can be
  downstream of bad data from a marginal memory path; BOTH crash configs tonight were
  non-optimal topologies.
- **BIOS reverted DRAM to 3200 MT/s** after the swap (the 2400 setting did not survive).
- **rasdaemon never recorded** (mc_event table missing — daemon wasn't running with --record
  since June). No decoded MCE history exists; fix = `systemctl enable --now rasdaemon`.
- S1 RAM-verify: clean (3+ passes, 0 corruption @18GB); S2 CPU/cache: in window.

**Revised action order:** (1) move sticks to A2+B2 at next power-off; (2) Power Supply Idle
Control → Typical Current Idle; (3) keep 3200 post-slot-fix, drop to 3000 only if floods recur;
(4) enable rasdaemon.

## Operator BIOS checklist (do on next reboot, highest value first)

1. **Power Supply Idle Control → Typical Current Idle** (fixes hypothesis 1 outright).
2. Verify DRAM speed post-swap — BIOS may have reverted to XMP/3200 or auto; confirm what
   the matched pair actually runs at now (`sudo dmidecode -t 17`).
3. VSOC → 1.05–1.10 V (if DRAM ≥3200); optionally VDDG IOD 0.95–1.00 V.
4. Test-disable Core Performance Boost for one session (rules out boost droop).
5. Check ASUS for BIOS newer than 3634 (2025-09) — AGESA stability fixes.

## Pending root commands (operator, one line)

`sudo apt install -y stress-ng memtester && sudo ras-mc-ctl --errors | tail -40 && sudo dmidecode -t 17 | grep -E 'Locator|Configured Memory Speed|Size|Part Number'`

## Paused by this interrupt

821 attempt-5 relaunch (planner clean cell), --seed agent, decode agent — all held until the
box survives the ladder. Paper timer (02:21, 4G cap) left armed — trivial load, useful canary.
