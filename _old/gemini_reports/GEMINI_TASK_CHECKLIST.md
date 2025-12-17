# Gemini Task Checklist

## High Priority
- [x] Fix "Catastrophic Forgetting" in Rainbow Table (Implemented Checkpoint Transfer).
- [x] Fix Shape Mismatch in Rainbow Table (Implemented Asset Padding to 64).
- [ ] **Audit Path Handling for Portability**:
    - [ ] Scan codebase for hardcoded absolute paths.
    - [ ] Ensure all config files (JSON) store relative paths.
    - [ ] Ensure scripts (`train_jax_fast.py`, `run_scheduled_training.py`) use wrappers (`os.path.abspath`) only at runtime when interfacing with strict libraries (Orbax, SQLite).
    - [ ] Goal: Repository should be cloneable to a new machine/path without breaking.
- [ ] Implement "Universal Agent" (Transformer Backbone) for v2.0.

## Medium Priority
- [ ] Implement Multi-Horizon Curriculum (3h, 24h, 7d).
- [ ] Add Validation Split (10% temporal holdout).

## Low Priority
- [ ] Live Data Integration (WebSocket).