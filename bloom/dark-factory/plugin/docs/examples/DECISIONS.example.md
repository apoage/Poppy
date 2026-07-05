<!-- Live file from videoED, the project dark-factory was extracted from — kept verbatim as a worked example. -->
# DECISIONS — the single blocked-on-Lukas surface

One line per open call. **Lukas: answer inline** (edit this file or tell either session — the answer
gets recorded here + in the owning task). The implementer loop checks this file before starting
blocked work and NEVER guesses past an open row; the planner seeds new rows instead of scattering
open questions across task files (task files link here). Answered rows move to DECIDED with a date.

## OPEN

| id | decision | options (rec first) | owning task |
|----|----------|----------------------|-------------|
| D3 | **Moon landing (in-app delivery render)** — build now or gate on demand | **gate: build only when a real project wants an in-app final** (validate on Blue first) · alt: build after the ledger regardless | `2026-07-02-landings.md` |
| D6 | **Project portability** — `app/preview.py` uses absolute `clips[].path` from sync.json (fixture builder works around it); fix by rebasing paths on load? | **yes, queue a small task** (rebase against the project root on read; additive) · alt: accept non-portable projects | (unqueued — latent finding) |
| D7 | **Code-sign + notarize the .app?** Unsigned build → Gatekeeper "damaged/can't open" + App Translocation path mismatches | Lukas 2026-07-03: distribution "very limited" for now → **deprioritized; assistant/docs path short-term**; revisit when distribution grows. NOTE: unsigned+quarantined is the prime suspect for the field "path mismatches" — the explorer should confirm (translocation), since that finding raises D7 back up | `2026-07-03-install-assistant.md` |

## DECIDED

| date | id | decision | answer |
|------|----|----------|--------|
| 2026-07-04 | D2 | L2 creative payoff path | **Resolved via L3**: Lukas queued L3 (2026-07-04) — the payoff proof is L3-remix's acceptance (a steered run producing an APPLIED gate-passing reorder on real content; see `2026-07-04-l3-remix.md`). The CJF 2026 re-run stays an optional bonus validation whenever the project folder surfaces (NOT on Blue; processed at v0.8.0b). |
| 2026-07-03 | D8 | Voiceclone-pickup backend + insert policy | **Backend = Chatterbox** (mlx-audio Turbo, MIT; MegaTTS3 OUT — gated clone encoder), Qwen3-TTS a drop-in A/B via `pick_voice_backend()` on Lukas's real voice at build time. **Insert policy = draft-always-MARKED** (a marked draft IS a proposal; propose/dispose holds; low volume; strongest demo). Chatterbox's built-in watermark enforces synthetic-always-marked at the audio layer beneath the ledger/marker. |
| 2026-07-03 | D9 | Next build direction | **voiceclone-pickup** (backend decided, shovel-ready, best demo). pro-formats Rung 2 (TC-jam) optional filler; agent-mcp-surface if consolidating; fps-unify stays blocked on 60fps footage. |
| 2026-07-02 | D1 | Vision runtime for visual-index Phase 1 | **Apple Vision (pyobjc, on-device)** for classify + face/human/hand + saliency + OCR — decided on the FACT that MLX is not installed in the venv but `objc`/`Quartz` are and all Vision request classes load with zero model downloads. MLX-CLIP/VLM demoted to a later optional richer local engine behind the same `label_frame()` seam; the true-vision `api` tier later still. Shipped v0.8.6a. |
| 2026-07-03 | D5 | Footage-library storage + retrieval surface | **library.json `footage[]` + CLI query** now (SQLite + MCP when the library grows) — Lukas "both please" 2026-07-03. NOTE: the existing `library.json` (YouTube Long Form Assets) is a MOTION-GRAPHICS/asset pack, not b-roll footage — no real footage library exists yet, so Phase 3 ships the mechanics + is validated on a constructed footage set; real payoff awaits a real b-roll library. |
| 2026-07-02 | D4 | Visual-index knobs (grid/sampling/vocab) | **3×3 grid** default (`--grid` configurable); **`--every 2.0`, `--max-frames 400`**; scene-change **0.3 on the CLI** but **disabled on the app/interactive path (`--scene-thresh 1.0`)** so an app run isn't a full 4K decode per clip (Lukas: don't wait for processing to test UI) — both settings-overridable. Starter vocab (person/face/hands/screen/hardware/product/text/background/empty) + `screenshot`→`screen` whole-token key. Shipped v0.8.6a. |
