# workstation syncflood

> Workstation (workstation) recurring hard crashes — root cause is AMD data-fabric sync flood from a marginal mixed 4-DIMM RAM config; crashes even at idle.

*Migrated from local agent memory 2026-08-31 (`project_workstation_syncflood.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
Workstation `workstation` (Ryzen 7 5800X, ASUS PRIME B550-PLUS, BIOS 3634 2025-09-10) hard-crashes recurringly. Diagnosed 2026-06-26 after the 06-25 23:35 reboot.

**Signature:** kernel on next boot logs `x86/amd: Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event`. Previous boot's journal ends mid-stream with no shutdown sequence = involuntary hard reset (bypasses kernel, so kdump can't catch it). Recurred ≥5× in June (08, 09, 20, 21, 25); the non-sync-flood reboots were ACPI/clean (user-initiated). Each crash logs the SAME two MCE banks — `Bank 0` (Load-Store) + `Bank 22` — with UC+PCC (uncorrected, fatal) bits; TIME field advances per crash → reproducible IMC/memory fault, not random.

**Root cause:** marginal RAM config — 4 dual-rank DIMMs from TWO different kits: 2× Kingston FURY Beast DDR4-3200 16GB (KF3200C16D4, slots A1/B1) + 2× Kingston HyperX DDR4-2666 8GB (KHX2666C15D4, slots A2/B2). 8 ranks on a single-CCD 5800X IMC; AMD only guarantees DDR4-2133 with 4×DR. Board auto-downclocked all to 2666 @ 1.2V JEDEC but still unstable. Mixing kits is the core problem.

**Fix (priority order):** (1) pull the two 8GB HyperX sticks, run only the matched 2× 16GB FURY Beast pair in recommended 2-DIMM slots (A2+B2) = 32GB validated config — highest-priority, free. (2) If keeping 4 DIMMs: 2133 + raise VSOC ~1.10V / VDDG IOD-CCD in BIOS (band-aid). `rasdaemon` now installed+enabled; next fault decodes into `sudo ras-mc-ctl --errors`.

**Ruled out:** sdb (WD Red 4TB, storage disk/.git drive) healthy 36°C — the "112°C" smartd alert was a normalized-VALUE misread. sdc (Seagate 8TB) healthy; ata3 cold-boot IDENTIFY timeout is benign slow-spinup. nvme1 (KC3000 OS drive) healthy but 160 unsafe-shutdowns = crash body-count. **nvme0** has a latent issue (`smartctl` → Host Pathing Error 0x370; kernel cmdline already carries `nvme_core.default_ps_max_latency_us=0` APST workaround) — NOT the crash cause, but worth watching.

**Why it matters:** these crashes fire AT IDLE (desktop/Firefox only, no CUDA) — a SEPARATE, more fundamental failure mode than the earlier GPU-hang crashes. The box is unstable independent of GPU load.

**FULL FIX APPLIED 2026-07-06 ~00:15 (final story — supersedes the kit-mismatch diagnosis):**
The stick pull ALONE did NOT fix it — the matched pair sync-flooded again 23:30 (2026-07-05),
~5 min into a memory-heavy load. Forensics: **MCE bit-identical across both RAM configs**
(Bank 0 LS, status `baa00000000c0135`, SYND `4d000014`) → not the kit. The real compound cause:
(a) sticks were placed in **A1+B1 — the electrically worst 2-DIMM slots** on this daisy-chain
board (A2+B2 is validated), (b) BIOS silently reverted DRAM to 3200 after the swap, (c) the
Zen low-load idle-voltage instability (explains June's at-idle floods). Fix set: sticks →
**A2+B2**, BIOS **3634→3641 (AGESA ComboV2 PI 1.2.0.12)**, **Power Supply Idle Control →
Typical Current Idle**, VSOC/VDDG set per `runs/822_stability/BIOS_CHEATSHEET.md`. Validated:
stress ladder PASSED (19 checksum passes 0 corruption + 25-min stress-ng 0 failures, full
boost 84°C) and a 6h13m 2000-epoch GPU training run completed clean (first ever — attempts
1-4 all died). See `tasks/TASK_822_WORKSTATION_STABILITY.md` + `lore/notes_2026-07-06.md`.
RESOURCE CONSEQUENCE (as of 2026-07-06): 32GB total — single-job caps ≤18G, aggregate ≤~22G,
keep ≥8G free. **SUPERSEDED 2026-08-01 — do NOT size jobs from these numbers.** The operator
changed the modules and runs a custom-calculated profile; MemTotal is now 56,444,032 kB ≈ 53.8 GiB.
Current caps: single job ≤32G, aggregate ≤40G, keep ≥12G free. Everything above this line
describes the OLD 4-DIMM/32GB config and is kept as forensic history only.
**rasdaemon: still BROKEN** (0.8.3 vs kernel 6.17, boot-time buffer-overflow ABRT, known
upstream bug; apt has no newer) — raw MCEs still reach the kernel journal (sufficient for
diagnosis); real fix = build rasdaemon ≥0.8.4 from source. Open watch item.

**2026-07-05 22:55 recurrence (+1 data point):** same signature (`0x08000800` sync flood), ~19 min
into the 821 GPU training run + 3 capped CPU agents. NO oom-killer entries — all workloads were
inside their cgroup caps; operator initially guessed "something oom", forensics say RAM fault
again. Consequence policy going forward: long runs on this box must assume random hard resets —
preserve logs per attempt (runs/<id>/attemptN_crashed/), prefer resumable/checkpointed designs,
and the matched-2×16GB fix keeps gaining votes.

**How to apply:** if the workstation reboots unexpectedly, check `journalctl -b 0 -k | grep "reset reason"` first — if sync flood, it's this RAM fault, not software/GPU. See `feedback_no_gpu_jobs_on_workstation` and `feedback_sandbox_experiments` for the rules that
now govern workstation compute. **The old corollary — "GPU compute → server only; workstation
untrustworthy" — was RETIRED by operator override 2026-07-05:** the requirement is correct
allocation (cgroup RAM cap + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`, never bare), not abstinence.
The RAM fault this file diagnoses was fixed 2026-07-06 and the modules replaced 2026-08-01.
